"""Verify the entire Phase 1 data pipeline works end-to-end."""

import sys
from data.dataset import FloorPlanDataset


def test_cubicasa():
    print("=" * 60)
    print("TEST 1: CubiCasa5k Data Pipeline")
    print("=" * 60)

    try:
        ds = FloorPlanDataset.from_cubicasa5k(
            "data/cubicasa5k",
            split="train",
            target_size=(512, 512)
        )
        print(f"✅ Dataset loaded: {len(ds)} samples")

        if len(ds) == 0:
            print("❌ Dataset is empty!")
            return False

        img, mask = ds[0]
        print(f"✅ Image shape: {img.shape}")
        print(f"✅ Mask shape: {mask.shape}")
        print(f"✅ Image dtype: {img.dtype}, range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"✅ Mask unique classes: {sorted(mask.unique().tolist())}")

        # Check class distribution
        for cid in sorted(mask.unique().tolist()):
            count = (mask == cid).sum().item()
            pct = count / mask.numel() * 100
            print(f"   Class {cid}: {count:,} pixels ({pct:.1f}%)")

        print("\n🎯 CubiCasa5k pipeline: PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Plan2Scene — Phase 1 Data Pipeline Verification\n")

    ok1 = test_cubicasa()

    print("\n" + "=" * 60)
    if ok1:
        print("RESULT: ALL TESTS PASSED ✅")
        print("Phase 1 is complete. Ready for Phase 2 (Model Training).")
    else:
        print("RESULT: TESTS FAILED ❌")
        print("Fix errors above before proceeding.")
    print("=" * 60)

    return 0 if ok1 else 1


if __name__ == "__main__":
    sys.exit(main())