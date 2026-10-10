"""Serve predictions from the model trained for a run.

Safety notes
* ``joblib.load`` unpickles, so it must only ever read files written by this application. The run
  id is validated and the path is pinned inside ``runs_dir`` (see ``jobs.JobManager.run_path``).
* Inputs are validated against ``schema.json``: unknown fields are ignored, missing fields fall
  back to the training default (and are reported), values far outside the training range or
  unseen categories are flagged so callers know the prediction is an extrapolation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

MISSING_INDICATOR_SUFFIX = "__was_missing"


class PredictionError(ValueError):
    """Invalid prediction input (message is safe to show to the caller)."""


@dataclass
class Predictor:
    bundle: dict
    schema: dict

    @classmethod
    def load(cls, artifacts_dir: Path) -> Predictor:
        model_path, schema_path = artifacts_dir / "model.joblib", artifacts_dir / "schema.json"
        if not model_path.is_file() or not schema_path.is_file():
            raise FileNotFoundError("No trained model exists for this run.")
        return cls(joblib.load(model_path), json.loads(schema_path.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------------------------------
    @property
    def all_features(self) -> list[dict]:
        """Every model input, including indicator columns the cleaning step derived."""
        return self.schema["features"]

    @property
    def derived(self) -> dict[str, str]:
        """``{indicator column: source column}`` for ``<col>__was_missing`` flags added during cleaning."""
        names = {f["name"] for f in self.all_features}
        return {
            n: n[: -len(MISSING_INDICATOR_SUFFIX)]
            for n in names
            if n.endswith(MISSING_INDICATOR_SUFFIX) and n[: -len(MISSING_INDICATOR_SUFFIX)] in names
        }

    @property
    def features(self) -> list[dict]:
        """The fields a caller supplies: all model inputs except the derived indicators."""
        return [f for f in self.all_features if f["name"] not in self.derived]

    def public_schema(self) -> dict:
        """Schema for clients (API + UI): the derived indicators are computed here, never asked for."""
        return {**self.schema, "features": self.features}

    def _coerce(self, record: dict) -> tuple[dict, dict]:
        """Return (clean_row, notes) for one input record."""
        notes: dict[str, list] = {"defaulted_fields": [], "ignored_fields": [], "out_of_range": [], "unseen_categories": []}
        known = {f["name"]: f for f in self.features}
        notes["ignored_fields"] = sorted(k for k in record if k not in known)
        row = {}
        for name, spec in known.items():
            value = record.get(name)
            if value is None or (isinstance(value, str) and not value.strip()):
                row[name] = spec["default"]
                notes["defaulted_fields"].append(name)
                continue
            if spec["kind"] == "numeric":
                try:
                    number = float(str(value).replace(",", "")) if isinstance(value, str) else float(value)
                except (TypeError, ValueError) as exc:
                    raise PredictionError(f"Field '{name}' must be a number, got {value!r}") from exc
                span = (spec["max"] - spec["min"]) or 1.0
                if number < spec["min"] - 0.25 * span or number > spec["max"] + 0.25 * span:
                    notes["out_of_range"].append({"field": name, "value": number, "training_range": [spec["min"], spec["max"]]})
                row[name] = number
            else:
                text = str(value).strip()
                if text not in spec["choices"]:
                    notes["unseen_categories"].append({"field": name, "value": text})
                row[name] = text
        # A "was missing" flag is a fact about the input, not an input: it is 1 exactly when the caller left
        # the source field blank (the model then sees the training default, as it did during training).
        for flag, source in self.derived.items():
            row[flag] = 1.0 if source in notes["defaulted_fields"] else 0.0
        return row, notes

    def predict(self, records: list[dict], explain: bool = True) -> list[dict]:
        if not records:
            raise PredictionError("Provide at least one record.")
        if len(records) > 1000:
            raise PredictionError("At most 1000 records per request.")
        rows, notes = zip(*(self._coerce(r) for r in records), strict=True)
        frame = pd.DataFrame(list(rows), columns=self.bundle["feature_order"])
        pipe = self.bundle["pipeline"]
        classification = self.bundle["task_type"] == "classification"
        scores = pipe.predict_proba(frame)[:, 1] if classification else pipe.predict(frame)

        out = []
        for i, (row, note) in enumerate(zip(rows, notes, strict=True)):
            result: dict = {"warnings": {k: v for k, v in note.items() if v}}
            if classification:
                p = float(scores[i])
                thr, base = float(self.bundle["threshold"]), float(self.bundle["base_rate"])
                result.update(
                    probability=round(p, 4),
                    prediction=int(p >= thr),
                    label=str(self.bundle["positive_label"]) if p >= thr else f"not {self.bundle['positive_label']}",
                    risk_band="high" if p >= thr else ("medium" if p >= base else "low"),
                    threshold=round(thr, 4),
                    base_rate=round(base, 4),
                )
            else:
                result["prediction"] = round(float(scores[i]), 4)
            if explain and i < 50:  # explanations cost ~40 model calls per feature per row
                result["top_factors"] = self._explain(pipe, row, float(scores[i]), classification)
            out.append(result)
        return out

    def _explain(self, pipe, row: dict, score: float, classification: bool) -> list[dict]:
        """Local explanation without extra dependencies.

        For every feature we replace its value by values observed in the training data (40 stored
        samples) and average the model output. ``effect`` = score - that average, i.e. how much
        *this* value moves the prediction compared with a typical value of the feature. Positive
        effects push the score up. (A one-feature-at-a-time what-if, not a full Shapley value.)
        """
        names = self.bundle["feature_order"]
        # Derived "was missing" flags are explained through their source feature; on its own a flag means little to a reader.
        explained = [n for n in names if n not in self.derived]
        marginals = self.bundle.get("marginals") or {}
        blocks, sizes = [], []
        for name in explained:
            values = marginals.get(name) or [self._default(name)]
            block = pd.DataFrame([row] * len(values), columns=names)
            block[name] = values
            blocks.append(block)
            sizes.append(len(values))
        batch = pd.concat(blocks, ignore_index=True)
        preds = pipe.predict_proba(batch)[:, 1] if classification else pipe.predict(batch)
        effects, start = [], 0
        for name, size in zip(explained, sizes, strict=True):
            typical = float(preds[start : start + size].mean())
            start += size
            effects.append({"feature": name, "value": row[name], "effect": round(float(score) - typical, 4)})
        effects = [e for e in effects if abs(e["effect"]) > 1e-4]
        effects.sort(key=lambda e: -abs(e["effect"]))
        return effects[:5]

    def _default(self, name: str):
        return next(f["default"] for f in self.all_features if f["name"] == name)
