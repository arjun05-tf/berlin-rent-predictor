"""Compare logged API requests against the training data. Usage: python scripts/check_drift.py"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from berlinrentml.config import PROCESSED_DATA_DIR
from berlinrentml.monitoring import psi

LOG = Path(__file__).parent.parent / "logs" / "requests.jsonl"
FEATURES = ["livingSpace", "rooms", "floor"]


def main():
    if not LOG.exists():
        sys.exit(f"No request log at {LOG}. Send some /predict calls first.")
    live = pd.DataFrame(json.loads(line) for line in LOG.read_text(encoding="utf8").splitlines())
    train = pd.read_csv(PROCESSED_DATA_DIR / "berlin_processed.csv")
    print(f"{len(live)} logged requests vs {len(train):,} training rows\n")
    for f in FEATURES:
        if f in live and live[f].notna().any():
            v = psi(train[f], live[f])
            print(f"{f:<12} PSI {v:.3f}  {'STABLE' if v < 0.1 else 'MODERATE' if v < 0.25 else 'SHIFTED'}")


if __name__ == "__main__":
    main()
