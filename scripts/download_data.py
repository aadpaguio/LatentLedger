#!/usr/bin/env python
"""Verify local parquet files exist."""

from pathlib import Path

# UPDATE THESE PATHS TO WHERE YOUR PARQUETS ARE!
PARQUET_PATHS = {
    'churn': Path('/Users/arnaldpaguio/Documents/Portofolio /LatentLedger/data/age.parquet'),      # ← CHANGE!
    'default': Path('/Users/arnaldpaguio/Documents/Portofolio /LatentLedger/data/default.parquet'),  # ← CHANGE!
    'hsbc': Path('/Users/arnaldpaguio/Documents/Portofolio /LatentLedger/data/hsbc.parquet'),        # ← CHANGE!
    'age': Path('/Users/arnaldpaguio/Documents/Portofolio /LatentLedger/data/age.parquet'),          # ← CHANGE!
}

def verify_parquets():
    """Verify all parquet files exist."""
    print("Verifying parquet files...")
    
    all_found = True
    for dataset_name, path in PARQUET_PATHS.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 ** 2)
            print(f"✓ {dataset_name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {dataset_name}: NOT FOUND at {path}")
            all_found = False
    
    if all_found:
        print("\n✓ All parquet files found and ready!")
        return True
    else:
        print("\n✗ Some parquet files missing. Check paths above.")
        return False

if __name__ == "__main__":
    success = verify_parquets()
    exit(0 if success else 1)