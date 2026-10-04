"""Dependency-free LightGBM inference: serve a trained booster from JSON with numpy only.

LightGBM needs the system library libgomp, which serverless runtimes (Vercel) lack.
Export once with `scripts/export_deploy_assets.py`; predictions match LightGBM to float precision.
"""

import json
from pathlib import Path

import numpy as np


class PortableLGBM:
    """Evaluate a LightGBM regression forest dumped with Booster.dump_model()."""

    def __init__(self, trees: list, expm1: bool = False):
        self.trees = trees
        self.expm1 = expm1  # model was trained on log1p(target)

    @classmethod
    def from_model(cls, model) -> "PortableLGBM":
        """Build from a fitted LGBMRegressor, or a TransformedTargetRegressor wrapping one."""
        log_target = hasattr(model, "regressor_")
        booster = (model.regressor_ if log_target else model).booster_
        return cls([t["tree_structure"] for t in booster.dump_model()["tree_info"]], expm1=log_target)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"expm1": self.expm1, "trees": self.trees}), encoding="utf8")

    @classmethod
    def load(cls, path: Path) -> "PortableLGBM":
        d = json.loads(Path(path).read_text(encoding="utf8"))
        return cls(d["trees"], d["expm1"])

    @staticmethod
    def _leaf(node: dict, row: np.ndarray) -> float:
        while "leaf_value" not in node:
            x = row[node["split_feature"]]
            if np.isnan(x):
                go_left = node["default_left"]
            else:
                go_left = x <= node["threshold"]
            node = node["left_child"] if go_left else node["right_child"]
        return node["leaf_value"]

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=float)
        raw = np.array([sum(self._leaf(t, row) for t in self.trees) for row in X])
        return np.expm1(raw) if self.expm1 else raw
