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


class HarnessManagedLLM(LocalLLM):
    def __init__(
        self,
        harness_root: str | Path,
        harness_models_root: str | Path | None = None,
        model_id: str | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        chat_format: str | None = None,
    ) -> None:
        try:
            from harness_logic.registry import HarnessModelRegistry
            from harness_logic.store import LlamaModelStore
        except ImportError as exc:
            raise RuntimeError("Harness backend requires the harness_logic package") from exc

        self.harness_root = Path(harness_root).expanduser()
        self.harness_models_root = Path(harness_models_root).expanduser() if harness_models_root else None
        self.model_store = LlamaModelStore(self.harness_root, model_root=self.harness_models_root)
        if model_id:
            model = HarnessModelRegistry.find_legacy_model(model_id)
            if model is None:
                known = ", ".join(model.id for model in HarnessModelRegistry.available_model_infos())
                raise ValueError(f"Unknown harness model id: {model_id}. Known ids: {known}")
            self.model_id = model.id
            llm_path = self.model_store.model_dir_for(model) / model.gguf_file_name
        else:
            files = self.model_store.get_selected_model_files()
            self.model_id = files.model.id
            llm_path = files.artifact_files["llm"]
        if not llm_path.is_file():
            raise FileNotFoundError(
                "Harness model artifact is missing: "
                f"{llm_path}. Run `./run.sh harness download-plan` to see download sources, "
                "then place the GGUF at that path."
            )
        self.model_path = llm_path
        self.llm = GGUFLLM(
            str(llm_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            chat_format=chat_format,
        )

    def generate(self, messages: list[dict[str, str]], options: GenerationOptions) -> str:
        return self.llm.generate(messages, options)


def infer_model_backend(model_name_or_path: str | None, backend: str | None = None) -> str:
    if not model_name_or_path:
        if backend and backend.lower() == "harness":
            return "harness"
        return "echo"
    if backend:
        normalized = backend.lower()
        if normalized not in {"transformers", "gguf", "harness"}:
            raise ValueError("model backend must be one of: transformers, gguf, harness")
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
    harness_root: str | Path = "harness_logic/data",
    harness_models_root: str | Path | None = "models",
    harness_model_id: str | None = None,
) -> LocalLLM:
    selected = infer_model_backend(model_name_or_path, backend)
    if selected == "harness":
        return HarnessManagedLLM(
            harness_root,
            harness_models_root=harness_models_root,
            model_id=harness_model_id or model_name_or_path,
            n_ctx=gguf_n_ctx,
            n_gpu_layers=gguf_n_gpu_layers,
            chat_format=gguf_chat_format,
        )
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
