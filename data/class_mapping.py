"""
Plan2Scene — Class Mapping Configuration
Maps CubiCasa5k labels to our unified 10-class segmentation schema.
"""

CLASSES = [
    "background",      # 0
    "wall",            # 1
    "room_living",     # 2
    "room_bedroom",    # 3
    "room_kitchen",    # 4
    "room_bath",       # 5
    "room_other",      # 6
    "door",            # 7
    "window",          # 8
    "stairs",          # 9
]

NUM_CLASSES = len(CLASSES)

COLOR_MAP = {
    0: (0, 0, 0),
    1: (255, 255, 255),
    2: (230, 126, 34),
    3: (52, 152, 219),
    4: (241, 196, 15),
    5: (46, 204, 113),
    6: (149, 165, 166),
    7: (231, 76, 60),
    8: (155, 89, 182),
    9: (243, 156, 18),
}

# ── Labels to completely skip (annotation UI, furniture, helpers) ──
SKIP_LABELS = {
    "Swing Beside",      # door swing arc (not physical)
    "Direction",         # arrow indicators
    "removeControl",     # SVG UI control
    "translateControl",  # SVG UI control
    "SelectionControls", # SVG UI control
    "BoundaryPolygon",   # building outline (we have walls)
    "InnerPolygon",      # inner room boundary duplicate
    "Bench",             # furniture
    "Hanger",            # furniture
    "Faucet",            # fixture
    "Column Rectangle",  # structural column (skip for now)
}

# ── CubiCasa label → our class ID ──
CUBICASA_TO_UNIFIED = {
    # Walls
    "Wall": 1,
    "Railing": 1,
    "Wall2": 1,

    # Living room
    "Living room": 2,
    "LivingRoom": 2,
    "Lounge": 2,

    # Bedrooms
    "Bedroom": 3,
    "Master bedroom": 3,
    "Guest room": 3,

    # Kitchen
    "Kitchen": 4,
    "Kitchen2": 4,

    # Bathrooms
    "Bathroom": 5,
    "Bath": 5,
    "Toilet": 5,
    "Sauna": 5,
    "Shower": 5,

    # Other rooms
    "Hallway": 6,
    "Hall": 6,
    "Corridor": 6,
    "Storage": 6,
    "Closet": 6,
    "Wardrobe": 6,
    "Garage": 6,
    "Balcony": 6,
    "Terrace": 6,
    "Entrance": 6,
    "Lobby": 6,
    "Entry Lobby": 6,
    "DraughtLobby": 6,
    "Office": 6,
    "Dining room": 6,
    "Utility room": 6,
    "Utility Laundry": 6,
    "Boiler room": 6,
    "Walk-in closet": 6,
    "Undefined": 6,
    "External": 6,
    "Outdoor Terrace": 6,

    # Doors
    "Door": 7,
    "Door2": 7,
    "Sliding door": 7,
    "Double door": 7,
    "Regular": 7,        # CubiCasa: regular door
    "Panel": 7,          # CubiCasa: door panel
    "PanelArea": 7,      # CubiCasa: door panel area
    "Threshold": 7,      # CubiCasa: door threshold

    # Windows
    "Window": 8,
    "Window2": 8,
    "Bay window": 8,
    "Glass": 8,          # CubiCasa: glass door/window

    # Stairs
    "Stairs": 9,
    "Staircase": 9,
    "Flight": 9,         # CubiCasa: stair flight
    "Steps": 9,          # CubiCasa: stair steps
}

DEFAULT_CLASS = 6


def map_label(label: str) -> int:
    """Map a CubiCasa5k label string to our unified class ID."""
    if not label:
        return DEFAULT_CLASS

    # Skip non-architectural elements
    if label in SKIP_LABELS:
        return -1  # signal to skip

    # Exact match
    if label in CUBICASA_TO_UNIFIED:
        return CUBICASA_TO_UNIFIED[label]

    # Case-insensitive exact match
    label_lower = label.lower()
    for key, val in CUBICASA_TO_UNIFIED.items():
        if key.lower() == label_lower:
            return val

    # Partial match
    for key, val in CUBICASA_TO_UNIFIED.items():
        if key.lower() in label_lower or label_lower in key.lower():
            return val

    return DEFAULT_CLASS


def get_color(class_id: int) -> tuple:
    return COLOR_MAP.get(class_id, (128, 128, 128))