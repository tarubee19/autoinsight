"""Missing-value indicator columns added during cleaning are derived by the predictor, never asked from users."""

import pytest

from autoinsight.pipeline import run_pipeline
from autoinsight.predictor import Predictor

from .conftest import make_settings


@pytest.fixture(scope="module")
def house_predictor(tmp_path_factory, samples):
    tmp = tmp_path_factory.mktemp("indicator")
    res = run_pipeline(samples / "house_prices.csv", "What drives the sale price of a house?", settings=make_settings(tmp))
    assert res.ok, res.error
    return Predictor.load(res.run_dir / "artifacts")


def test_indicator_is_a_model_input_but_not_a_caller_field(house_predictor):
    p = house_predictor
    assert p.derived == {"lot_size__was_missing": "lot_size"}
    assert "lot_size__was_missing" in {f["name"] for f in p.all_features}
    assert "lot_size__was_missing" not in {f["name"] for f in p.features}
    assert "lot_size__was_missing" not in {f["name"] for f in p.public_schema()["features"]}


def test_indicator_follows_whether_the_source_field_was_supplied(house_predictor):
    given, _ = house_predictor._coerce({"sqft": 1500, "lot_size": 5000})
    omitted, notes = house_predictor._coerce({"sqft": 1500})
    assert given["lot_size__was_missing"] == 0.0
    assert omitted["lot_size__was_missing"] == 1.0
    assert "lot_size" in notes["defaulted_fields"] and "lot_size__was_missing" not in notes["defaulted_fields"]
    _, ignored = house_predictor._coerce({"sqft": 1500, "lot_size__was_missing": 1})
    assert ignored["ignored_fields"] == ["lot_size__was_missing"]  # callers cannot set it directly


def test_predictions_and_explanations_work_with_indicators(house_predictor):
    out = house_predictor.predict([{"sqft": 2100, "bathrooms": 2}, {"sqft": 2100, "bathrooms": 2, "lot_size": 6000}])
    assert all(r["prediction"] > 0 for r in out)
    assert all("__was_missing" not in f["feature"] for r in out for f in r["top_factors"])
