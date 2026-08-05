"""Quick verification that the new parser extracts real classes."""

from data.cubicasa_parser import parse_cubicasa_sample
import numpy as np

image, mask, meta = parse_cubicasa_sample(
    "data/cubicasa5k/high_quality_architectural/6044/F1_scaled.png",
    "data/cubicasa5k/high_quality_architectural/6044/model.svg",
    target_size=(512, 512)
)

print(f"Elements parsed: {meta['num_elements']}")
print(f"Label counts: {meta['label_counts']}")
print(f"\nClass distribution in mask:")
for cid, count in sorted(meta['class_distribution'].items()):
    if count > 0:
        pct = count / mask.size * 100
        print(f"  Class {cid}: {count:,} px ({pct:.1f}%)")

unique = sorted(np.unique(mask))
print(f"\nUnique classes in mask: {unique}")

if len(unique) > 1:
    print("\n✅ PARSER IS WORKING — mask contains multiple classes!")
else:
    print("\n❌ Still only background — parser needs more fixes.")