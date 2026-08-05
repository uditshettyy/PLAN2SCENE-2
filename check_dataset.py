import os
from pathlib import Path

base = Path("data/cubicasa5k")
print(f"Looking at: {base.absolute()}")

if not base.exists():
    print("❌ data/cubicasa5k NOT FOUND")
    exit(1)

items = list(base.iterdir())
print(f"Contents: {[p.name for p in items]}")

# Check structure
if (base / "train").exists():
    print("✅ Found train/ split folder")
    train_items = list((base / "train").iterdir())[:5]
    print(f"  Sample train items: {[p.name for p in train_items]}")
    
    # Check one house
    first_house = [p for p in (base / "train").iterdir() if p.is_dir()][0]
    print(f"  First house: {first_house.name}")
    print(f"  House contents: {[p.name for p in first_house.iterdir()]}")
    
elif (base / "model.svg").exists():
    print("⚠️  Flat structure (no train/val/test splits)")
else:
    # Maybe split is defined by text files
    for txt in ["train.txt", "val.txt", "test.txt"]:
        if (base / txt).exists():
            print(f"✅ Found {txt}")
            with open(base / txt) as f:
                lines = f.readlines()
                print(f"   {len(lines)} entries")
                print(f"   First 3: {lines[:3]}")