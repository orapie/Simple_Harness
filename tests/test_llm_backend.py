from __future__ import annotations

import unittest

from simple_rag.llm import infer_model_backend


class LLMBackendTest(unittest.TestCase):
    def test_infers_echo_without_model(self) -> None:
        self.assertEqual("echo", infer_model_backend(None))

    def test_infers_gguf_from_file_suffix(self) -> None:
        self.assertEqual("gguf", infer_model_backend("/models/minicpm.gguf"))

    def test_infers_transformers_for_directory_like_path(self) -> None:
        self.assertEqual("transformers", infer_model_backend("/models/minicpm"))

    def test_explicit_backend_overrides_suffix(self) -> None:
        self.assertEqual("transformers", infer_model_backend("/models/model.gguf", "transformers"))


if __name__ == "__main__":
    unittest.main()
