from autoinsight.grounding import check_numbers

FACTS = {"rate": 0.474, "n": 5000, "auc": 0.8349, "text": "median 14.00 vs 31.00", "d": -0.89, "money": 1555.72}


def test_real_numbers_pass_with_rounding_and_percent():
    text = "Churn is 47.4% (n=5,000); ROC AUC 0.83; median 14.00 vs 31.00; d = 0.89; revenue 1,555.72."
    r = check_numbers(text, FACTS)
    assert r.ungrounded == [] and r.rate == 1.0


def test_invented_numbers_are_flagged():
    r = check_numbers("Churn rose to 61.2% and 12,345 customers left; AUC 0.97.", FACTS)
    assert set(r.ungrounded) == {"61.2%", "12,345", "0.97"}
    assert r.rate < 0.5


def test_structural_integers_and_years_are_ignored():
    r = check_numbers("Top 3 drivers over 5-fold CV in 2026; 1. first 2) second", FACTS)
    assert r.checked == 0 and r.rate == 1.0


def test_code_spans_and_urls_are_ignored():
    r = check_numbers("See `x = 98765` and https://example.com/v1234567", FACTS)
    assert r.checked == 0
