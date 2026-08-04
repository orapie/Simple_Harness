from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    chat_parser.add_argument("--model")
    chat_parser.add_argument("--max-new-tokens", type=int, default=512)
    chat_parser.add_argument("--temperature", type=float, default=0.7)
    chat_parser.add_argument("--top-p", type=float, default=0.9)
    chat_parser.add_argument("--no-sample", action="store_true")

    args = parser.parse_args()
    if args.command == "index":
        handle_index(args)
    elif args.command == "search":
        handle_search(args)
    elif args.command == "prompt":
        handle_prompt(args)
    elif args.command == "chat":
        handle_chat(args)


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


def _build_pipeline(args: argparse.Namespace, model_name_or_path: str | None) -> RagPipeline:
    embedding_model = make_embedding_model(args.embedding_model, device=args.device)
    llm = make_llm(model_name_or_path, device=args.device)
    return RagPipeline.from_index_file(
        args.index,
        embedding_model,
        llm,
        prompt_builder=PromptBuilder(max_context_chars=args.max_context_chars),
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


if __name__ == "__main__":
    main()
