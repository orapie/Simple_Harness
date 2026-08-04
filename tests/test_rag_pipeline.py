from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simple_rag.embeddings import HashEmbeddingModel
from simple_rag.llm import GenerationOptions, PromptEchoLLM
from simple_rag.pipeline import RagPipeline
from simple_rag.roles import RoleProfile
from simple_rag.store import VectorIndex


class RagPipelineTest(unittest.TestCase):
    def test_build_search_and_prompt_with_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            (data / "notes.txt").write_text(
                "西班牙在决赛后夺冠。商业计划引发球迷对票价的讨论。",
                encoding="utf-8",
            )
            embedding = HashEmbeddingModel(dimensions=64)
            index = VectorIndex.build(data, embedding, chunk_size=80, overlap=10)
            index_path = root / "index.json"
            index.save(index_path)

            pipeline = RagPipeline.from_index_file(index_path, embedding, PromptEchoLLM())
            role = RoleProfile(
                id="tester",
                name="测试角色",
                persona="冷静回答。",
                style=["用中文"],
                rules=["依据资料"],
            )
            response = pipeline.answer(
                "决赛后有什么商业争议？",
                role,
                top_k=1,
                options=GenerationOptions(max_new_tokens=64),
            )

            self.assertIn("测试角色", response.answer)
            self.assertIn("商业计划", response.answer)
            self.assertEqual(len(response.prompt.sources), 1)


if __name__ == "__main__":
    unittest.main()
