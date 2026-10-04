"""Write public/options.json (dropdown values for the web form) from the trained preprocessor."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from berlinrentml.config import MODELS_DIR
from berlinrentml.modeling.artifacts import load_model_artifacts

_, prep, _ = load_model_artifacts("final_model", MODELS_DIR)
enc = prep.preprocessor.named_transformers_["cat"][-1]
opts = {f: sorted(str(v) for v in c) for f, c in zip(prep.categorical_features, enc.categories_)}
out = Path(__file__).parent.parent / "public" / "options.json"
out.write_text(json.dumps({k: opts[k] for k in ("geo_bln", "geo_plz", "condition", "heatingType")}, ensure_ascii=False), encoding="utf8")
print(f"wrote {out}")
