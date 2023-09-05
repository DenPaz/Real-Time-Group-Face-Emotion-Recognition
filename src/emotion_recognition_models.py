from keras.applications import MobileNetV2
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from keras.models import Sequential
from keras.optimizers import Adam


def get_model(img_size, choice):
    if choice == "goodfellow":
        return goodfellow(img_size)
    elif choice == "mobilenetv2":
        return mobilenetv2(img_size)
    else:
        raise ValueError(f"Unknown model choice {choice}")


def mobilenetv2(img_size):
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )

    model = Sequential(
        [
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(7, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy", "categorical_crossentropy"],
    )

    return model


def goodfellow(img_size):
    """
    Inspired by Goodfellow, I.J., et.al. (2013). Challenged in representation learning: A report of three machine learning contests. Neural Networks, 64, 59-63. doi:10.10161i.neunet.201409005
    """

    model = Sequential()

    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            input_shape=(img_size, img_size, 1),
            padding="same",
            activation="relu",
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5, 5),
            padding="same",
            activation="relu",
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    model.add(
        Conv2D(
            filters=512,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    model.add(
        Conv2D(
            filters=512,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    model.add(Flatten())

    model.add(Dense(units=256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.25))

    model.add(Dense(units=512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.25))

    model.add(Dense(units=7, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy", "categorical_crossentropy"],
    )

    return model
