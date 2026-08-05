from __future__ import annotations

import argparse
import json
from pathlib import Path

from .character_rag import CharacterRagPipeline, CharacterRagPromptBuilder
from .embeddings import make_embedding_model
from .llm import GenerationOptions, make_llm
from .pipeline import RagPipeline
from .prompt import PromptBuilder
from .roles import RoleProfile
from .store import VectorIndex


def main() -> None:
    parser = argparse.ArgumentParser(prog="simple-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="build a local vector index")
    index_parser.add_argument("--data", default="data")
    index_parser.add_argument("--index", default=".rag_index/index.json")
    index_parser.add_argument("--embedding-model")
    index_parser.add_argument("--device")
    index_parser.add_argument("--chunk-size", type=int, default=900)
    index_parser.add_argument("--overlap", type=int, default=120)

    search_parser = subparsers.add_parser("search", help="search an existing index")
    _add_query_args(search_parser)
    search_parser.add_argument("--index", default=".rag_index/index.json")
    search_parser.add_argument("--embedding-model")
    search_parser.add_argument("--device")
    search_parser.add_argument("--top-k", type=int, default=5)

    prompt_parser = subparsers.add_parser("prompt", help="compile a role RAG prompt")
    _add_query_args(prompt_parser)
    _add_chat_common_args(prompt_parser)

    chat_parser = subparsers.add_parser("chat", help="run retrieval and local generation")
    _add_query_args(chat_parser)
    _add_chat_common_args(chat_parser)
    _add_model_args(chat_parser)
    chat_parser.add_argument("--model")

    character_prompt_parser = subparsers.add_parser(
        "character-rag-prompt",
        help="compile a character prompt with retrieved external data",
    )
    _add_query_args(character_prompt_parser)
    _add_character_rag_common_args(character_prompt_parser)
    character_prompt_parser.add_argument("--debug", action="store_true")

    character_chat_parser = subparsers.add_parser(
        "character-rag-chat",
        help="run character retrieval and local generation",
    )
    _add_query_args(character_chat_parser)
    _add_character_rag_common_args(character_chat_parser)
    _add_model_args(character_chat_parser)
    character_chat_parser.add_argument("--model")
    character_chat_parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    if args.command == "index":
        handle_index(args)
    elif args.command == "search":
        handle_search(args)
    elif args.command == "prompt":
        handle_prompt(args)
    elif args.command == "chat":
        handle_chat(args)
    elif args.command == "character-rag-prompt":
        handle_character_rag_prompt(args)
    elif args.command == "character-rag-chat":
        handle_character_rag_chat(args)


def handle_index(args: argparse.Namespace) -> None:
    embedding_model = make_embedding_model(args.embedding_model, device=args.device)
    index = VectorIndex.build(
        args.data,
        embedding_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    index.save(args.index)
    print(json.dumps({"index": args.index, "chunks": len(index.items)}, ensure_ascii=False, indent=2))


def handle_search(args: argparse.Namespace) -> None:
    embedding_model = make_embedding_model(args.embedding_model, device=args.device)
    index = VectorIndex.load(args.index)
    results = index.search(args.query, embedding_model, top_k=args.top_k)
    print(
        json.dumps(
            [
                {
                    "score": round(result.score, 6),
                    "source": result.chunk.source_path,
                    "start": result.chunk.start,
                    "end": result.chunk.end,
                    "text": result.chunk.text[:500],
                }
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


def handle_prompt(args: argparse.Namespace) -> None:
    pipeline = _build_pipeline(args, model_name_or_path=None)
    response = pipeline.answer(args.query, RoleProfile.load(args.role), top_k=args.top_k)
    print(json.dumps({"messages": response.prompt.messages, "sources": response.prompt.sources}, ensure_ascii=False, indent=2))


def handle_chat(args: argparse.Namespace) -> None:
    pipeline = _build_pipeline(args, model_name_or_path=args.model)
    options = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
    )
    response = pipeline.answer(args.query, RoleProfile.load(args.role), top_k=args.top_k, options=options)
    print(response.answer)
    print("\n--- sources ---")
    print(json.dumps(response.prompt.sources, ensure_ascii=False, indent=2))


def handle_character_rag_prompt(args: argparse.Namespace) -> None:
    pipeline = _build_character_rag_pipeline(args, model_name_or_path=None)
    response = pipeline.answer(
        args.query,
        args.npc,
        story_cutoff=args.cutoff,
        top_k=args.top_k,
        max_character_chars=args.max_character_chars,
    )
    payload = {
        "messages": response.prompt.messages,
        "sources": response.prompt.sources,
    }
    if args.debug:
        payload["character_debug"] = response.prompt.character_debug
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def handle_character_rag_chat(args: argparse.Namespace) -> None:
    pipeline = _build_character_rag_pipeline(args, model_name_or_path=args.model)
    options = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
    )
    response = pipeline.answer(
        args.query,
        args.npc,
        story_cutoff=args.cutoff,
        top_k=args.top_k,
        max_character_chars=args.max_character_chars,
        options=options,
    )
    print(response.answer)
    print("\n--- sources ---")
    print(json.dumps(response.prompt.sources, ensure_ascii=False, indent=2))
    if args.debug:
        print("\n--- character debug ---")
        print(json.dumps(response.prompt.character_debug, ensure_ascii=False, indent=2))


def _build_pipeline(args: argparse.Namespace, model_name_or_path: str | None) -> RagPipeline:
    embedding_model = make_embedding_model(args.embedding_model, device=args.device)
    llm = _make_llm_from_args(args, model_name_or_path)
    return RagPipeline.from_index_file(
        args.index,
        embedding_model,
        llm,
        prompt_builder=PromptBuilder(max_context_chars=args.max_context_chars),
    )


def _build_character_rag_pipeline(
    args: argparse.Namespace,
    model_name_or_path: str | None,
) -> CharacterRagPipeline:
    embedding_model = make_embedding_model(args.embedding_model, device=args.device)
    llm = _make_llm_from_args(args, model_name_or_path)
    return CharacterRagPipeline.from_index_file(
        args.index,
        embedding_model,
        llm,
        prompt_builder=CharacterRagPromptBuilder(max_context_chars=args.max_context_chars),
        character_root=args.character_root,
        evidence_root=args.evidence_root,
    )


def _make_llm_from_args(args: argparse.Namespace, model_name_or_path: str | None):
    return make_llm(
        model_name_or_path,
        device=args.device,
        backend=getattr(args, "model_backend", None),
        gguf_n_ctx=getattr(args, "gguf_n_ctx", 4096),
        gguf_n_gpu_layers=getattr(args, "gguf_n_gpu_layers", 0),
        gguf_chat_format=getattr(args, "gguf_chat_format", None),
    )


def _add_query_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query", required=True)


def _add_chat_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index", default=".rag_index/index.json")
    parser.add_argument("--role", default="configs/roles/default.json")
    parser.add_argument("--embedding-model")
    parser.add_argument("--device")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=5000)


def _add_character_rag_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index", default=".rag_index/index.json")
    parser.add_argument("--character-root", default="character_system")
    parser.add_argument("--evidence-root")
    parser.add_argument("--npc", default="lu_jiangxian")
    parser.add_argument("--cutoff")
    parser.add_argument("--embedding-model")
    parser.add_argument("--device")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=5000)
    parser.add_argument("--max-character-chars", type=int, default=4500)


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-backend", choices=["transformers", "gguf"])
    parser.add_argument("--gguf-n-ctx", type=int, default=4096)
    parser.add_argument("--gguf-n-gpu-layers", type=int, default=0)
    parser.add_argument("--gguf-chat-format")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--no-sample", action="store_true")


if __name__ == "__main__":
    main()
