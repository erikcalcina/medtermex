import unittest
from test.assets.common import TEST_CASES_EXACT, TEST_CASES_OVERLAP, TEST_CASES_RELAXED

from common.evaluate import evaluate_ner_performance

# =====================================
# Test Common
# =====================================


class TestCommonEvaluate(unittest.TestCase):
    def test_evaluate_ner_performance_default(self):
        for (
            true_ents,
            pred_ents,
            expected_precision,
            expected_recall,
            expected_f1,
        ) in TEST_CASES_EXACT:
            with self.subTest(true_ents=true_ents, pred_ents=pred_ents):
                precision, recall, f1 = evaluate_ner_performance(true_ents, pred_ents)
                self.assertAlmostEqual(expected_precision, precision)
                self.assertAlmostEqual(expected_recall, recall)
                self.assertAlmostEqual(expected_f1, f1)

    def test_evaluate_ner_performance_exact(self):
        for (
            true_ents,
            pred_ents,
            expected_precision,
            expected_recall,
            expected_f1,
        ) in TEST_CASES_EXACT:
            with self.subTest(true_ents=true_ents, pred_ents=pred_ents):
                precision, recall, f1 = evaluate_ner_performance(
                    true_ents, pred_ents, match_type="exact"
                )
                self.assertAlmostEqual(expected_precision, precision)
                self.assertAlmostEqual(expected_recall, recall)
                self.assertAlmostEqual(expected_f1, f1)

    def test_evaluate_ner_performance_relaxed(self):
        for (
            true_ents,
            pred_ents,
            expected_precision,
            expected_recall,
            expected_f1,
        ) in TEST_CASES_RELAXED:
            with self.subTest(true_ents=true_ents, pred_ents=pred_ents):
                precision, recall, f1 = evaluate_ner_performance(
                    true_ents, pred_ents, match_type="relaxed"
                )
                self.assertAlmostEqual(expected_precision, precision)
                self.assertAlmostEqual(expected_recall, recall)
                self.assertAlmostEqual(expected_f1, f1)

    def test_evaluate_ner_performance_overlap(self):
        for (
            true_ents,
            pred_ents,
            expected_precision,
            expected_recall,
            expected_f1,
        ) in TEST_CASES_OVERLAP:
            with self.subTest(true_ents=true_ents, pred_ents=pred_ents):
                precision, recall, f1 = evaluate_ner_performance(
                    true_ents, pred_ents, match_type="overlap"
                )
                self.assertAlmostEqual(expected_precision, precision)
                self.assertAlmostEqual(expected_recall, recall)
                self.assertAlmostEqual(expected_f1, f1)
