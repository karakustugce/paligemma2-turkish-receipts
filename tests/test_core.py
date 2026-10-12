import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from pg2receipts.core import load_manifest, paired_stats, parse_amount, score_answer, wilson


def paired_rows(patterns):
    rows = []
    for i, (a, b) in enumerate(patterns):
        for r, correct in [(224, a), (448, b)]:
            rows.append({"doc_id": str(i), "resolution": r, "correct": correct,
                         "sha256": str(i), "true_total": "100.00"})
    return rows


class ScientificChecks(unittest.TestCase):
    def test_turkish_and_international_money_formats(self):
        for raw in ["1234,56", "1.234,56", "1234.56", "1,234.56", "₺ 1.234,56", "1234,56 TL"]:
            self.assertEqual(parse_amount(raw), Decimal("1234.56"))

    def test_ambiguous_or_multiple_numbers_do_not_get_guessed(self):
        for raw in ["1.234", "12,5", "100 veya 120", "Toplam 120,00", "UNREADABLE", "NaN", "-10,00"]:
            self.assertIsNone(parse_amount(raw))

    def test_decimal_cent_difference_is_a_wrong_total(self):
        self.assertFalse(score_answer("100,01", "100.00")["correct"])
        self.assertEqual(score_answer("100,01", "100.00")["error_type"], "wrong_total")
        self.assertIsNone(score_answer("UNREADABLE", "100.00")["absolute_error"])

    def test_incomplete_pairs_and_duplicates_are_rejected(self):
        rows = paired_rows([(True, False), (False, True)])
        with self.assertRaises(ValueError): paired_stats(rows[:-1])
        with self.assertRaises(ValueError): paired_stats(rows + [rows[0]])

    def test_ground_truth_changes_across_models_are_rejected(self):
        rows = paired_rows([(True, True)])
        rows[1]["true_total"] = "101.00"
        with self.assertRaises(ValueError): paired_stats(rows)

    def test_paired_counts_and_exact_test_on_known_discordance(self):
        rows = paired_rows([(False, True)]*5 + [(True, True)]*5)
        result = paired_stats(rows, bootstrap_samples=1000)
        self.assertEqual(result["only_448_correct"], 5)
        self.assertEqual(result["accuracy_difference_448_minus_224"], .5)
        self.assertAlmostEqual(result["mcnemar_exact_p"], .0625)

    def test_zero_observed_errors_does_not_mean_zero_population_uncertainty(self):
        lo, hi = wilson(50, 50)
        self.assertLess(lo, 1.0)
        self.assertAlmostEqual(hi, 1.0)
        result = paired_stats(paired_rows([(True, True)]*30), bootstrap_samples=1000)
        self.assertEqual(result["mcnemar_exact_p"], 1.0)

    def test_question_languages_are_paired_within_the_same_model(self):
        rows = []
        for i in range(30):
            for language in ["en", "tr"]:
                rows.append({"doc_id": str(i), "resolution": 224,
                             "prompt_language": language, "correct": i < (20 if language == "en" else 25),
                             "sha256": str(i), "true_total": "100.00"})
        result = paired_stats(rows, condition_key="prompt_language", condition_values=("en", "tr"))
        self.assertEqual(result["only_tr_correct"], 5)
        self.assertAlmostEqual(result["accuracy_difference_tr_minus_en"], 5/30)

    def test_a_language_pair_cannot_hide_missing_documents(self):
        rows = [{"doc_id": "one", "prompt_language": "en", "correct": True,
                 "sha256": "one", "true_total": "100.00"}]
        with self.assertRaises(ValueError):
            paired_stats(rows, condition_key="prompt_language", condition_values=("en", "tr"))

    def test_duplicate_images_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); images = root / "images"; images.mkdir()
            for i in range(30): (images / f"{i}.jpg").write_bytes(f"fixture-{i}".encode())
            manifest = root / "manifest.csv"
            lines = ["doc_id,image_path,true_total"] + [f"doc_{i},{i}.jpg,100.00" for i in range(30)]
            manifest.write_text("\n".join(lines))
            self.assertEqual(len(load_manifest(manifest, images)), 30)
            draft_lines = ["doc_id,image_path,true_total,annotation_status"] + [f"doc_{i},{i}.jpg,100.00,draft_assistant_visual_review" for i in range(30)]
            manifest.write_text("\n".join(draft_lines))
            with self.assertRaises(ValueError): load_manifest(manifest, images)
            manifest.write_text("\n".join(lines))
            (images / "1.jpg").write_bytes((images / "0.jpg").read_bytes())
            with self.assertRaises(ValueError): load_manifest(manifest, images)
            lines[1] = "doc_0,../outside.jpg,100.00"
            manifest.write_text("\n".join(lines))
            with self.assertRaises(ValueError): load_manifest(manifest, images)


if __name__ == "__main__":
    unittest.main()
