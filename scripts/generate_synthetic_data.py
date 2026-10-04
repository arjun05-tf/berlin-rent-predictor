"""Generate synthetic Berlin rental dataset for demonstration."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from berlinrentml.config import RAW_DATA_DIR

# Set seed for reproducibility
np.random.seed(42)


def generate_synthetic_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic Berlin rental data for demonstration.

    This simulates realistic patterns but is NOT real data.
    """
    print(f"Generating {n_samples:,} synthetic listings...")

    # Berlin districts
    districts = [
        "Mitte", "Friedrichshain-Kreuzberg", "Pankow", "Charlottenburg-Wilmersdorf",
        "Spandau", "Steglitz-Zehlendorf", "Tempelhof-Schöneberg", "Neukölln",
        "Treptow-Köpenick", "Marzahn-Hellersdorf", "Lichtenberg", "Reinickendorf",
    ]

    # Postal code ranges by district
    plz_by_district = {
        "Mitte": ["10115", "10117", "10119", "10178", "10179"],
        "Friedrichshain-Kreuzberg": ["10243", "10245", "10247", "10249", "10997", "10999"],
        "Pankow": ["13187", "13189", "13347", "13349", "13353"],
        "Charlottenburg-Wilmersdorf": ["10585", "10587", "10589", "10623", "10625"],
        "Spandau": ["13581", "13583", "13585", "13587", "13589"],
        "Steglitz-Zehlendorf": ["12163", "12165", "12167", "12169", "12203"],
        "Tempelhof-Schöneberg": ["10777", "10779", "10781", "10783", "10823"],
        "Neukölln": ["12043", "12045", "12047", "12049", "12051"],
        "Treptow-Köpenick": ["12435", "12437", "12439", "12459", "12487"],
        "Marzahn-Hellersdorf": ["12619", "12621", "12623", "12627", "12679"],
        "Lichtenberg": ["10315", "10317", "10318", "10319", "10365"],
        "Reinickendorf": ["13403", "13405", "13407", "13409", "13435"],
    }

    # Base rent by district (approximate median, €/m²)
    base_rent_per_m2 = {
        "Mitte": 16.0,
        "Friedrichshain-Kreuzberg": 15.5,
        "Pankow": 13.0,
        "Charlottenburg-Wilmersdorf": 14.5,
        "Spandau": 10.0,
        "Steglitz-Zehlendorf": 13.5,
        "Tempelhof-Schöneberg": 13.0,
        "Neukölln": 11.0,
        "Treptow-Köpenick": 11.5,
        "Marzahn-Hellersdorf": 8.5,
        "Lichtenberg": 10.5,
        "Reinickendorf": 11.0,
    }

    data = []

    for _ in range(n_samples):
        # Select district
        district = np.random.choice(districts)
        plz = np.random.choice(plz_by_district[district])

        # Living space (m²)
        living_space = np.random.lognormal(mean=4.2, sigma=0.4)
        living_space = np.clip(living_space, 20, 200)

        # Rooms (correlated with size)
        rooms = max(1, int(living_space / 30) + np.random.choice([-1, 0, 1]))
        if rooms == 1:
            rooms = np.random.choice([1.0, 1.5, 2.0], p=[0.3, 0.5, 0.2])

        # Construction year
        year_weights = [0.1, 0.2, 0.3, 0.25, 0.15]
        year_bins = [
            np.random.randint(1900, 1945),
            np.random.randint(1945, 1970),
            np.random.randint(1970, 1990),
            np.random.randint(1990, 2010),
            np.random.randint(2010, 2025),
        ]
        year = np.random.choice(year_bins, p=year_weights)

        # Floor
        floor = np.random.choice(list(range(0, 8)), p=[0.15, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05])

        # Amenities
        has_kitchen = np.random.choice([True, False], p=[0.8, 0.2])
        has_balcony = np.random.choice([True, False], p=[0.6, 0.4])
        has_garden = np.random.choice([True, False], p=[0.2, 0.8])
        has_cellar = np.random.choice([True, False], p=[0.7, 0.3])

        # Condition
        condition = np.random.choice(
            ["modernized", "well_kept", "need_of_renovation", "first_time_use"],
            p=[0.3, 0.5, 0.15, 0.05]
        )

        # Heating type
        heating_type = np.random.choice(
            ["central_heating", "district_heating", "gas_heating", "oil_heating"],
            p=[0.5, 0.3, 0.15, 0.05]
        )

        # Calculate base rent
        base_price = base_rent_per_m2[district]

        # Adjustments
        if year >= 2010:
            base_price *= 1.15  # New buildings
        elif year >= 1990:
            base_price *= 1.05
        elif year < 1945:
            base_price *= 0.95  # Old buildings

        if condition == "modernized":
            base_price *= 1.1
        elif condition == "need_of_renovation":
            base_price *= 0.85
        elif condition == "first_time_use":
            base_price *= 1.2

        if has_balcony:
            base_price *= 1.05
        if has_garden:
            base_price *= 1.08

        if floor == 0:
            base_price *= 0.95  # Ground floor
        elif floor >= 5:
            base_price *= 0.95  # High floor without elevator (assumed)

        # Add noise
        base_price *= np.random.normal(1.0, 0.1)

        # Calculate rent
        base_rent = base_price * living_space
        base_rent = max(300, base_rent)  # Minimum rent

        # Service charge (Nebenkosten)
        service_charge = living_space * np.random.uniform(1.5, 3.5)

        # Total rent
        total_rent = base_rent + service_charge

        data.append({
            "regio1": "Berlin",
            "geo_bln": district,
            "geo_plz": plz,
            "livingSpace": round(living_space, 1),
            "rooms": rooms,
            "floor": floor,
            "yearConstructed": int(year),
            "hasKitchen": has_kitchen,
            "hasBalcony": has_balcony,
            "hasGarden": has_garden,
            "cellar": has_cellar,
            "condition": condition,
            "heatingType": heating_type,
            "baseRent": round(base_rent, 2),
            "serviceCharge": round(service_charge, 2),
            "totalRent": round(total_rent, 2),
        })

    df = pd.DataFrame(data)
    return df


def main():
    """Generate and save synthetic dataset."""
    print("=" * 80)
    print("BerlinRentML - Synthetic Data Generator")
    print("=" * 80)
    print("\nWARNING: This generates SYNTHETIC data for demonstration only.")
    print("Real results require downloading the actual ImmobilienScout24 dataset.")
    print("=" * 80)

    # Generate data
    df = generate_synthetic_data(n_samples=5000)

    # Save
    output_path = RAW_DATA_DIR / "immo_data.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df):,} synthetic listings to: {output_path}")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nSample statistics:")
    print(f"  Base rent range: €{df['baseRent'].min():.0f} - €{df['baseRent'].max():.0f}")
    print(f"  Mean base rent: €{df['baseRent'].mean():.2f}")
    print(f"  Living space range: {df['livingSpace'].min():.0f} - {df['livingSpace'].max():.0f} m²")
    print(f"  Districts: {df['geo_bln'].nunique()}")

    print("\n" + "=" * 80)
    print("Next step: Run training")
    print("  python scripts/train.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
