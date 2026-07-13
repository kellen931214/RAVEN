import unittest

from utils.bit_accuracy import (
    METRIC_SCHEMA_VERSION,
    STATUS_ERROR,
    STATUS_NOT_AVAILABLE,
    STATUS_OK,
    add_metric_schema,
    cache_has_current_metric_schema,
    extract_bit_accuracy,
    format_bit_accuracy,
    format_bit_accuracy_rows,
    format_staged_bit_accuracy_rows,
    run_bit_decoder,
    summarize_bit_accuracy,
)


class BitAccuracyTests(unittest.TestCase):
    def test_gs_real_value_is_formatted_to_four_places(self):
        metric = extract_bit_accuracy({"bit_accuracies": [0.812345]})
        self.assertEqual(metric.status, STATUS_OK)
        self.assertEqual(format_bit_accuracy(metric), "0.8123")

    def test_detection_only_providers_are_not_available(self):
        for provider_output in (
            {"p_values": [0.01]},
            {"l1_dist": [12.0]},
            {"l1_dist": [13.0]},
            {"l1_dist": [14.0]},
        ):
            with self.subTest(provider_output=provider_output):
                metric = extract_bit_accuracy(provider_output)
                self.assertIsNone(metric.value)
                self.assertEqual(metric.status, STATUS_NOT_AVAILABLE)
                self.assertEqual(format_bit_accuracy(metric), "N/A")

    def test_before_and_after_aggregate_only_valid_values(self):
        missing = extract_bit_accuracy({})
        before = summarize_bit_accuracy(
            [extract_bit_accuracy({"bit_accuracies": [1.0]}), missing]
        )
        after = summarize_bit_accuracy(
            [extract_bit_accuracy({"bit_accuracies": [0.5]}), missing]
        )
        self.assertEqual(before["value"], 1.0)
        self.assertEqual(after["value"], 0.5)
        self.assertEqual(before["valid_count"], 1)
        self.assertEqual(after["valid_count"], 1)

    def test_table_and_csv_format_before_and_after_independently(self):
        rows = format_staged_bit_accuracy_rows(
            [
                {
                    "before_bit_accuracy": 0.812345,
                    "before_bit_accuracy_status": STATUS_OK,
                    "after_bit_accuracy": None,
                    "after_bit_accuracy_status": STATUS_NOT_AVAILABLE,
                }
            ]
        )
        self.assertEqual(rows[0]["before_bit_accuracy"], "0.8123")
        self.assertEqual(rows[0]["after_bit_accuracy"], "N/A")

        error_rows = format_bit_accuracy_rows(
            [
                {
                    "bit_accuracy": None,
                    "bit_accuracy_status": STATUS_ERROR,
                    "bit_accuracy_error": "decoder failed",
                }
            ]
        )
        self.assertEqual(error_rows[0]["bit_accuracy"], "ERROR")

    def test_genuine_zero_is_valid_but_none_is_not(self):
        summary = summarize_bit_accuracy(
            [extract_bit_accuracy({"bit_accuracies": [0.0]}), extract_bit_accuracy({})]
        )
        self.assertEqual(summary["value"], 0.0)
        self.assertEqual(summary["display"], "0.0000")
        self.assertEqual(summary["valid_count"], 1)

    def test_decoder_exception_is_error_and_counted(self):
        def fail():
            raise RuntimeError("decoder failed")

        metric = run_bit_decoder(fail)
        summary = summarize_bit_accuracy([metric])
        self.assertEqual(metric.status, STATUS_ERROR)
        self.assertEqual(summary["display"], "ERROR")
        self.assertEqual(summary["error_count"], 1)

    def test_old_cache_schema_is_not_reused(self):
        self.assertFalse(cache_has_current_metric_schema({"bit_accuracy": 0.0}))
        self.assertFalse(
            cache_has_current_metric_schema(
                {"metric_schema_version": METRIC_SCHEMA_VERSION - 1}
            )
        )
        current = add_metric_schema({"bit_accuracy": None})
        self.assertTrue(cache_has_current_metric_schema(current))


if __name__ == "__main__":
    unittest.main()
