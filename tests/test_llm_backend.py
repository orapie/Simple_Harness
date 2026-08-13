from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from simple_rag.llm import HarnessManagedLLM, infer_model_backend, make_llm


class LLMBackendTest(unittest.TestCase):
    def test_infers_echo_without_model(self) -> None:
        self.assertEqual("echo", infer_model_backend(None))

    def test_infers_gguf_from_file_suffix(self) -> None:
        self.assertEqual("gguf", infer_model_backend("/models/minicpm.gguf"))

    def test_infers_transformers_for_directory_like_path(self) -> None:
        self.assertEqual("transformers", infer_model_backend("/models/minicpm"))

    def test_explicit_backend_overrides_suffix(self) -> None:
        self.assertEqual("transformers", infer_model_backend("/models/model.gguf", "transformers"))

    def test_explicit_harness_backend_can_use_selected_model_without_model_arg(self) -> None:
        self.assertEqual("harness", infer_model_backend(None, "harness"))

    def test_harness_backend_resolves_model_artifact_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_root = root / "models"
            model_path = models_root / "llama-3.2-1b-instruct" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
            model_path.parent.mkdir(parents=True)
            model_path.write_text("fake gguf", encoding="utf-8")

            with patch("simple_rag.llm.GGUFLLM") as gguf_cls:
                llm = make_llm(
                    None,
                    backend="harness",
                    harness_root=root / "state",
                    harness_models_root=models_root,
                    harness_model_id="llama-3.2-1b-instruct",
                )

            self.assertIsInstance(llm, HarnessManagedLLM)
            self.assertEqual("llama-3.2-1b-instruct", llm.model_id)
            self.assertEqual(model_path, llm.model_path)
            gguf_cls.assert_called_once_with(
                str(model_path),
                n_ctx=4096,
                n_gpu_layers=0,
                chat_format=None,
            )


if __name__ == "__main__":
    unittest.main()
