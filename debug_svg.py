"""Debug SVG parsing for a single CubiCasa5k sample."""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

# Pick the first house from train split
root_dir = Path("data/cubicasa5k")
split_file = root_dir / "train.txt"

with open(split_file) as f:
    first_line = f.readline().strip()
    # Remove leading/trailing slashes
    rel_path = first_line.strip('/')
    house_dir = root_dir / rel_path

print(f"House directory: {house_dir}")
print(f"Exists: {house_dir.exists()}")

# List all files
print(f"\nFiles in house dir:")
for f in sorted(house_dir.iterdir()):
    print(f"  {f.name}")

# Find SVG
svg_path = house_dir / "model.svg"
print(f"\nSVG path: {svg_path}")
print(f"SVG exists: {svg_path.exists()}")

if not svg_path.exists():
    print("❌ No SVG found!")
    exit(1)

# Parse raw SVG
tree = ET.parse(svg_path)
root = tree.getroot()

print(f"\nSVG root tag: {root.tag}")

# List ALL elements (first 50)
print(f"\nFirst 50 elements in SVG:")
count = 0
for elem in root.iter():
    tag = elem.tag
    if '}' in tag:
        tag = tag.split('}')[1]
    attrs = dict(elem.attrib)
    # Only print key attrs
    short_attrs = {k: v for k, v in attrs.items() if k in ['points', 'class', 'id', 'd', 'label']}
    print(f"  <{tag}> {short_attrs}")
    count += 1
    if count >= 50:
        print("  ... (truncated)")
        break

# Count polygons specifically
polygons = list(root.iter())
polygon_tags = [e for e in polygons if 'polygon' in e.tag.lower()]
path_tags = [e for e in polygons if 'path' in e.tag.lower()]

print(f"\nTotal <polygon> elements: {len(polygon_tags)}")
print(f"Total <path> elements: {len(path_tags)}")

# Try to extract labels from polygons
if polygon_tags:
    print(f"\nFirst 10 polygon labels:")
    for i, poly in enumerate(polygon_tags[:10]):
        label = poly.get('class') or poly.get('id') or 'NO LABEL'
        points = poly.get('points', '')[:60]
        print(f"  [{i}] label='{label}' points='{points}...'")

# Check if CubiCasa uses <g> groups with class attributes
group_tags = [e for e in polygons if 'g' in e.tag.lower()]
print(f"\nTotal <g> (group) elements: {len(group_tags)}")
if group_tags:
    print("First 5 groups:")
    for g in group_tags[:5]:
        print(f"  class={g.get('class')} id={g.get('id')}")
        # List children
        children = list(g)
        print(f"    children: {len(children)}")
        for c in children[:3]:
            tag = c.tag.split('}')[1] if '}' in c.tag else c.tag
            print(f"      <{tag}> class={c.get('class')} id={c.get('id')}")

print("\nDone.")