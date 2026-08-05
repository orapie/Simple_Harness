from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerationOptions:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


class LocalLLM:
    def generate(self, messages: list[dict[str, str]], options: GenerationOptions) -> str:
        raise NotImplementedError


class PromptEchoLLM(LocalLLM):
    def generate(self, messages: list[dict[str, str]], options: GenerationOptions) -> str:
        return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)


class TransformersLLM(LocalLLM):
    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        torch_dtype: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "local LLM inference requires: python -m pip install torch transformers"
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def generate(self, messages: list[dict[str, str]], options: GenerationOptions) -> str:
        prompt = self._format_messages(messages)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=options.max_new_tokens,
                do_sample=options.do_sample,
                temperature=options.temperature,
                top_p=options.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][encoded["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "\n\n".join(
            f"<|{message['role']}|>\n{message['content']}" for message in messages
        ) + "\n\n<|assistant|>\n"


class GGUFLLM(LocalLLM):
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        chat_format: str | None = None,
        verbose: bool = False,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "GGUF inference requires: python -m pip install -e '.[gguf]'"
            ) from exc

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"GGUF model file does not exist: {path}")
        kwargs = {
            "model_path": str(path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": verbose,
        }
        if chat_format:
            kwargs["chat_format"] = chat_format
        self.model = Llama(**kwargs)

    def generate(self, messages: list[dict[str, str]], options: GenerationOptions) -> str:
        result = self.model.create_chat_completion(
            messages=messages,
            max_tokens=options.max_new_tokens,
            temperature=options.temperature if options.do_sample else 0.0,
            top_p=options.top_p,
        )
        message = result["choices"][0]["message"]
        return str(message.get("content", "")).strip()


def infer_model_backend(model_name_or_path: str | None, backend: str | None = None) -> str:
    if not model_name_or_path:
        return "echo"
    if backend:
        normalized = backend.lower()
        if normalized not in {"transformers", "gguf"}:
            raise ValueError("model backend must be one of: transformers, gguf")
        return normalized
    if Path(model_name_or_path).suffix.lower() == ".gguf":
        return "gguf"
    return "transformers"


def make_llm(
    model_name_or_path: str | None,
    device: str | None = None,
    backend: str | None = None,
    gguf_n_ctx: int = 4096,
    gguf_n_gpu_layers: int = 0,
    gguf_chat_format: str | None = None,
) -> LocalLLM:
    selected = infer_model_backend(model_name_or_path, backend)
    if selected == "gguf":
        if model_name_or_path is None:
            raise ValueError("GGUF backend requires a model path")
        return GGUFLLM(
            model_name_or_path,
            n_ctx=gguf_n_ctx,
            n_gpu_layers=gguf_n_gpu_layers,
            chat_format=gguf_chat_format,
        )
    if selected == "transformers":
        return TransformersLLM(model_name_or_path, device=device)
    return PromptEchoLLM()
