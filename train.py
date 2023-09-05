from pathlib import Path

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from src.data_augmentation import get_data_generators
from src.emotion_recognition_models import get_model

BASE_DIR = Path(__file__).resolve().parent
DATASET_NAME = "fer2013"
DATASET_DIR = BASE_DIR / "datasets" / DATASET_NAME
TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 200
BATCH_SIZE = 32

IMAGE_SIZE = 224
COLOR_MODE = "rgb"
MODEL = "mobilenetv2"

# IMAGE_SIZE = 48
# COLOR_MODE = "grayscale"
# MODEL = "goodfellow"

train_data_gen, validation_data_gen = get_data_generators(
    train_dir=TRAIN_DIR,
    validation_dir=VALIDATION_DIR,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    color_mode=COLOR_MODE,
)

model = get_model(img_size=IMAGE_SIZE, choice=MODEL)

with open(MODELS_DIR / f"{MODEL}.json", "w") as f:
    f.write(model.to_json())

checkpoint = ModelCheckpoint(
    filepath=str(MODELS_DIR / f"{MODEL}-best-accuracy.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=10,
    min_lr=0.00001,
    mode="auto",
    verbose=1,
)
callbacks = [
    TensorBoard(log_dir=MODELS_DIR / "logs"),
    checkpoint,
    reduce_lr,
]

history = model.fit(
    x=train_data_gen,
    steps_per_epoch=train_data_gen.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_data_gen,
    validation_steps=validation_data_gen.samples // BATCH_SIZE,
    callbacks=callbacks,
)

model.save(MODELS_DIR / f"{MODEL}.h5")
