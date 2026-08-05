"""
Plan2Scene — CubiCasa5k SVG Annotation Parser (Fixed v2)

CubiCasa5k SVG structure:
  <g class="Space Outdoor Terrace">
    <polygon points="..."/>
  </g>
  <g class="Wall">
    <line x1="..." y1="..." x2="..." y2="..."/>
  </g>
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image, ImageDraw
import logging

from data.class_mapping import map_label, NUM_CLASSES, DEFAULT_CLASS, SKIP_LABELS

logger = logging.getLogger(__name__)

SVG_NS = "http://www.w3.org/2000/svg"

# Line thickness for rasterization
LINE_THICKNESS = {1: 8, 7: 6, 8: 6, 9: 6}


# ── Skip non-architectural group classes ──
SKIP_CLASS_KEYWORDS = [
    "Dimension", "Visual", "TextLabel", "NameLabel",
    "DimensionMark", "SpaceDimensionsLabel", "Model",
    "Floor", "Floorplan", "defs"
]


def _should_skip_group(class_attr: str) -> bool:
    if not class_attr:
        return True
    return any(kw in class_attr for kw in SKIP_CLASS_KEYWORDS)


def _extract_label_from_class(class_attr: str) -> Optional[str]:
    parts = class_attr.strip().split()
    if not parts:
        return None
    # Remove generic type prefixes
    if parts[0] in ("Space", "Wall", "Door", "Window", "Stair", "Stairs", "Railing"):
        label = " ".join(parts[1:]) if len(parts) > 1 else parts[0]
    else:
        label = " ".join(parts)
    return label if label else None


def _parse_polygon_points(points_str: str) -> List[Tuple[float, float]]:
    points = []
    for pt in points_str.strip().split():
        try:
            coords = pt.split(",")
            if len(coords) == 2:
                x, y = float(coords[0]), float(coords[1])
                points.append((x, y))
        except (ValueError, IndexError):
            continue
    return points


def _parse_line_element(elem) -> List[Tuple[float, float]]:
    try:
        x1 = float(elem.get("x1", 0))
        y1 = float(elem.get("y1", 0))
        x2 = float(elem.get("x2", 0))
        y2 = float(elem.get("y2", 0))
        return [(x1, y1), (x2, y2)]
    except (ValueError, TypeError):
        return []


def _parse_path_element(elem) -> List[Tuple[float, float]]:
    d = elem.get("d", "")
    if not d:
        return []
    points = []
    current = None
    tokens = d.replace(",", " ").split()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("M", "m", "L", "l"):
            i += 1
            if i + 1 < len(tokens):
                try:
                    x = float(tokens[i])
                    y = float(tokens[i + 1])
                    i += 2
                    if tok in ("m", "l") and current is not None:
                        x += current[0]
                        y += current[1]
                    current = (x, y)
                    points.append(current)
                except ValueError:
                    i += 1
            else:
                i += 1
        else:
            try:
                x = float(tok)
                if i + 1 < len(tokens):
                    y = float(tokens[i + 1])
                    i += 2
                    if current is not None:
                        current = (x, y)
                        points.append(current)
                    else:
                        i += 1
                else:
                    i += 1
            except ValueError:
                i += 1
    return points


def parse_svg_annotations(svg_path: str) -> List[Dict]:
    if not os.path.exists(svg_path):
        raise FileNotFoundError(f"SVG not found: {svg_path}")

    tree = ET.parse(svg_path)
    root = tree.getroot()

    elements = []

    for g in root.iter():
        tag = g.tag
        if "}" in tag:
            tag = tag.split("}")[1]
        if tag != "g":
            continue

        class_attr = g.get("class", "")
        if _should_skip_group(class_attr):
            continue

        label = _extract_label_from_class(class_attr)
        if not label:
            continue

        class_id = map_label(label)
        if class_id == -1:
            continue  # skip non-architectural

        for child in g:
            child_tag = child.tag
            if "}" in child_tag:
                child_tag = child_tag.split("}")[1]

            points = []
            geom_type = None

            if child_tag == "polygon":
                pts_str = child.get("points", "")
                points = _parse_polygon_points(pts_str)
                geom_type = "polygon"
            elif child_tag == "line":
                points = _parse_line_element(child)
                geom_type = "line"
            elif child_tag == "path":
                points = _parse_path_element(child)
                geom_type = "path"
            elif child_tag == "polyline":
                pts_str = child.get("points", "")
                points = _parse_polygon_points(pts_str)
                geom_type = "polyline"

            if len(points) < 2:
                continue

            elements.append({
                "label": label,
                "class_id": class_id,
                "points": points,
                "geom_type": geom_type,
            })

    logger.info(f"Parsed {len(elements)} architectural elements from {svg_path}")
    return elements


def rasterize_annotations(
    elements: List[Dict],
    image_size: Tuple[int, int],
    background_class: int = 0,
) -> np.ndarray:
    width, height = image_size
    mask = np.full((height, width), background_class, dtype=np.uint8)

    priority = {1: 4, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 3}
    elements_sorted = sorted(elements, key=lambda e: priority.get(e["class_id"], 5))

    for elem in elements_sorted:
        points = elem["points"]
        class_id = elem["class_id"]
        geom_type = elem["geom_type"]
        pts = [(int(round(x)), int(round(y))) for x, y in points]

        img_mask = Image.fromarray(mask)
        draw = ImageDraw.Draw(img_mask)

        try:
            if geom_type == "polygon" and len(pts) >= 3:
                draw.polygon(pts, fill=class_id)
            elif geom_type in ("line", "polyline", "path"):
                thickness = LINE_THICKNESS.get(class_id, 4)
                if len(pts) >= 2:
                    for i in range(len(pts) - 1):
                        draw.line([pts[i], pts[i + 1]], fill=class_id, width=thickness)
            mask = np.array(img_mask)
        except Exception as e:
            logger.warning(f"Failed to rasterize {elem['label']} ({geom_type}): {e}")
            continue

    return mask


def parse_cubicasa_sample(
    image_path: str,
    svg_path: str,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    elements = parse_svg_annotations(svg_path)
    mask = rasterize_annotations(elements, (orig_w, orig_h))

    if target_size and target_size != (orig_w, orig_h):
        image = image.resize(target_size, Image.BILINEAR)
        mask = Image.fromarray(mask).resize(target_size, Image.NEAREST)
        mask = np.array(mask)

    image = np.array(image)

    label_counts = {}
    for e in elements:
        lbl = e["label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    metadata = {
        "original_size": (orig_w, orig_h),
        "target_size": target_size or (orig_w, orig_h),
        "num_elements": len(elements),
        "label_counts": label_counts,
        "class_distribution": {
            int(cid): int(np.sum(mask == cid))
            for cid in range(NUM_CLASSES)
        },
    }

    return image, mask, metadata