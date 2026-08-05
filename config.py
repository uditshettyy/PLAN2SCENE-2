"""Global configuration for Plan2Scene."""

from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

IMAGE_SIZE = (512, 512)
NUM_CLASSES = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 50

for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)