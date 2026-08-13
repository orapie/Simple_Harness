import tempfile
import unittest
from pathlib import Path

from harness_logic.store import LlamaModelStore


class StoreTests(unittest.TestCase):
    def test_select_model_and_artifact_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LlamaModelStore(Path(tmp))
            store.set_selected_model("llama-3.2-1b-instruct")
            files = store.get_selected_model_files()
            self.assertEqual("llama-3.2-1b-instruct", files.model.id)
            self.assertEqual(["llm"], list(files.artifact_files))
            self.assertFalse(store.is_selected_model_downloaded())

    def test_touching_required_artifact_makes_text_model_downloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LlamaModelStore(Path(tmp))
            store.set_selected_model("llama-3.2-1b-instruct")
            files = store.get_selected_model_files()
            llm_path = files.artifact_files["llm"]
            llm_path.parent.mkdir(parents=True)
            llm_path.write_text("demo", encoding="utf-8")
            self.assertTrue(store.is_selected_model_downloaded())

    def test_model_root_can_be_separate_from_state_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "state"
            model_root = Path(tmp) / "models"
            store = LlamaModelStore(root, model_root=model_root)
            store.set_selected_model("llama-3.2-1b-instruct")

            files = store.get_selected_model_files()

            self.assertEqual(
                model_root / "llama-3.2-1b-instruct" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                files.artifact_files["llm"],
            )
            self.assertEqual(root / ".harness_state.json", store.prefs.path)


if __name__ == "__main__":
    unittest.main()
