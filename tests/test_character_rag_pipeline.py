from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simple_rag.character_rag import CharacterRagPipeline
from simple_rag.embeddings import HashEmbeddingModel
from simple_rag.llm import GenerationOptions, PromptEchoLLM
from simple_rag.store import VectorIndex


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CharacterRagPipelineTest(unittest.TestCase):
    def test_character_prompt_can_include_external_rag_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            (data / "world_cup.txt").write_text(
                "世界杯商业化争议集中在高票价、转播权益和球迷现场体验。",
                encoding="utf-8",
            )

            embedding = HashEmbeddingModel(dimensions=64)
            index = VectorIndex.build(data, embedding, chunk_size=80, overlap=10)
            index_path = root / "index.json"
            index.save(index_path)

            pipeline = CharacterRagPipeline.from_index_file(
                index_path,
                embedding,
                PromptEchoLLM(),
                character_root=PROJECT_ROOT / "character_system",
                evidence_root=PROJECT_ROOT / "data" / "novel_test",
            )
            response = pipeline.answer(
                "用陆江仙的口吻解释世界杯商业化争议。",
                "lu_jiangxian",
                story_cutoff="evt-010",
                top_k=1,
                options=GenerationOptions(max_new_tokens=64),
            )

            self.assertEqual("system", response.prompt.messages[0]["role"])
            self.assertEqual("user", response.prompt.messages[1]["role"])
            self.assertIn("我是陆江仙", response.prompt.messages[0]["content"])
            self.assertIn("外部检索资料使用规则", response.prompt.messages[0]["content"])
            self.assertIn("世界杯商业化争议", response.prompt.messages[1]["content"])
            self.assertIn("高票价", response.answer)
            self.assertEqual(1, len(response.prompt.sources))


if __name__ == "__main__":
    unittest.main()
