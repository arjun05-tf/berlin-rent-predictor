"""Download dataset from Kaggle."""

import sys
from pathlib import Path
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from berlinrentml.config import RAW_DATA_DIR


def main():
    """Download dataset using Kaggle API."""
    print("=" * 80)
    print("BerlinRentML - Dataset Download")
    print("=" * 80)

    print("\n[1/3] Checking Kaggle CLI...")
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"✓ Kaggle CLI installed: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Kaggle CLI not found.")
        print("\nInstall with: pip install kaggle")
        print("\nThen configure credentials:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New API Token'")
        print("3. Place kaggle.json in:")
        print("   - Linux/Mac: ~/.kaggle/kaggle.json")
        print("   - Windows: %USERPROFILE%\\.kaggle\\kaggle.json")
        return

    print("\n[2/3] Downloading dataset...")
    dataset_name = "corrieaar/apartment-rental-offers-in-germany"

    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset_name,
                "-p",
                str(RAW_DATA_DIR),
            ],
            check=True,
        )
        print(f"✓ Downloaded to {RAW_DATA_DIR}/")
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print("\nManual download:")
        print(f"1. Visit: https://www.kaggle.com/datasets/{dataset_name}")
        print("2. Click 'Download'")
        print(f"3. Extract CSV to: {RAW_DATA_DIR}/")
        return

    print("\n[3/3] Extracting archive...")
    import zipfile

    zip_path = RAW_DATA_DIR / "apartment-rental-offers-in-germany.zip"

    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        print(f"✓ Extracted to {RAW_DATA_DIR}/")

        # Remove zip file
        zip_path.unlink()
        print("✓ Cleaned up zip file")
    else:
        print("⚠ No zip file found, may already be extracted")

    # Check for CSV files
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    if csv_files:
        print(f"\n✅ Found {len(csv_files)} CSV file(s):")
        for f in csv_files:
            print(f"  - {f.name}")
    else:
        print("\n⚠ No CSV files found in data/raw/")

    print("\n" + "=" * 80)
    print("✅ Dataset download complete!")
    print("=" * 80)
    print("\nNext step: Run training")
    print("  python scripts/train.py")


if __name__ == "__main__":
    main()
