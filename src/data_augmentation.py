from keras.preprocessing.image import ImageDataGenerator


def get_data_generators(train_dir, validation_dir, batch_size, image_size, color_mode):
    train_image_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        shear_range=0.3,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    validation_image_generator = ImageDataGenerator(
        rescale=1.0 / 255,
    )

    train_data_generator = train_image_generator.flow_from_directory(
        directory=train_dir,
        batch_size=batch_size,
        target_size=(image_size, image_size),
        shuffle=True,
        class_mode="categorical",
        color_mode=color_mode,
    )
    validation_data_generator = validation_image_generator.flow_from_directory(
        directory=validation_dir,
        batch_size=batch_size,
        target_size=(image_size, image_size),
        shuffle=True,
        class_mode="categorical",
        color_mode=color_mode,
    )

    return train_data_generator, validation_data_generator
