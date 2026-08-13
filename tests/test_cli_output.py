from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from simple_rag.cli import handle_chat
from simple_rag.embeddings import HashEmbeddingModel
from simple_rag.store import VectorIndex


class CliOutputTest(unittest.TestCase):
    def test_chat_prints_compiled_prompt_before_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            (data / "notes.txt").write_text(
                "世界杯商业化争议集中在高票价和球迷现场体验。",
                encoding="utf-8",
            )
            role_path = root / "role.json"
            role_path.write_text(
                json.dumps(
                    {
                        "id": "tester",
                        "name": "测试角色",
                        "persona": "冷静回答。",
                        "style": ["用中文"],
                        "rules": ["依据资料"],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            embedding = HashEmbeddingModel(dimensions=64)
            index = VectorIndex.build(data, embedding, chunk_size=80, overlap=10)
            index_path = root / "index.json"
            index.save(index_path)

            args = argparse.Namespace(
                query="有哪些商业化争议？",
                index=str(index_path),
                role=str(role_path),
                embedding_model=None,
                device=None,
                top_k=1,
                max_context_chars=5000,
                model=None,
                model_backend=None,
                gguf_n_ctx=4096,
                gguf_n_gpu_layers=0,
                gguf_chat_format=None,
                max_new_tokens=64,
                temperature=0.7,
                top_p=0.9,
                no_sample=False,
            )

            output = io.StringIO()
            with redirect_stdout(output):
                handle_chat(args)

            text = output.getvalue()
            self.assertLess(text.index("--- prompt ---"), text.index("--- answer ---"))
            self.assertLess(text.index("--- answer ---"), text.index("--- sources ---"))
            self.assertIn('"messages"', text)
            self.assertIn("世界杯商业化争议", text)


if __name__ == "__main__":
    unittest.main()
