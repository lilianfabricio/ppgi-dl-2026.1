from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
ANDROID_ASSETS = ROOT / "android" / "app" / "src" / "main" / "assets"

NUM_CLASSES = 2  # background, pet
IMAGE_SIZE = 256
SUBSET_TRAIN = 500
SUBSET_VAL = 100

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_WORKERS = 2

CHECKPOINT_PATH = CHECKPOINT_DIR / "unet_best.pt"
EXPORT_PATH = CHECKPOINT_DIR / "unet_mobile.pt"
