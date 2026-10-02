import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

from data import all_letters, letterToIndex, lineToTensor, load_data, n_letters
from evaluate import evaluate, predict
from model import RNN
from train import train


class RegressionTests(unittest.TestCase):
    def test_single_character_training_ignores_unused_gradients(self):
        rnn = RNN(n_letters, 4, 2)
        output, loss = train(rnn, torch.tensor([0]), lineToTensor("A"), torch.nn.NLLLoss())
        self.assertEqual(tuple(output.shape), (1, 2))
        self.assertTrue(torch.isfinite(torch.tensor(loss)))

    def test_prediction_normalizes_unicode_and_caps_topk(self):
        rnn = RNN(n_letters, 4, 2)
        actual = predict(rnn, ["one", "two"], "Émile")
        expected = predict(rnn, ["one", "two"], "Emile", n_predictions=2)
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 2)

    def test_empty_and_unsupported_names_fail_clearly(self):
        with self.assertRaises(ValueError):
            lineToTensor("")
        with self.assertRaises(ValueError):
            letterToIndex("🙂")
        with self.assertRaises(ValueError):
            predict(RNN(n_letters, 4, 2), ["one", "two"], "🙂")
        with self.assertRaises(ValueError):
            evaluate(RNN(n_letters, 4, 2), torch.zeros(0, 1, n_letters))

    def test_loading_skips_empty_names_and_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "empty.txt").write_text("\n🙂\n", encoding="utf-8")
            (root / "valid.txt").write_text("\nÉmile\n🙂\nA\n", encoding="utf-8")
            lines, categories = load_data(str(root / "*.txt"))
            self.assertEqual(categories, ["valid"])
            self.assertEqual(lines, {"valid": ["Emile", "A"]})
            with self.assertRaises(ValueError):
                load_data(str(root / "missing-*.txt"))

    def test_evaluation_does_not_build_gradient_graph(self):
        output = evaluate(RNN(n_letters, 4, 2), lineToTensor("Ab"))
        self.assertFalse(output.requires_grad)

    def test_hidden_state_matches_model_dtype(self):
        self.assertEqual(RNN(n_letters, 4, 2).double().initHidden().dtype, torch.float64)

    def test_importing_examples_does_not_start_training(self):
        root = Path(__file__).resolve().parents[1]
        for filename in ("main.py", "__init__.py"):
            spec = importlib.util.spec_from_file_location("example_module", root / filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertTrue(callable(module.main))
            self.assertFalse(hasattr(module, "model"))
            self.assertFalse(hasattr(module, "rnn"))


if __name__ == "__main__":
    unittest.main()
