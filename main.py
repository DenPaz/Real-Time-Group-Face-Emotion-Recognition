from pathlib import Path

import cv2
import face_recognition
import numpy as np
import tensorflow as tf
from keras.models import model_from_json


def main():
    BASE_DIR = Path(__file__).resolve().parent
    MODEL = "mobilenetv2"
    MODELS_DIR = BASE_DIR / "models"

    MODEL_WEIGHTS = MODELS_DIR / f"{MODEL}.h5"
    MODEL_JSON = MODELS_DIR / f"{MODEL}.json"
    IMAGE_SIZE = 224
    SOURCE_INPUT = 0
    SCALE_FACTOR = 2

    input_stream = cv2.VideoCapture(SOURCE_INPUT)

    fer_model = model_from_json(open(MODEL_JSON, "r").read())
    fer_model.load_weights(MODEL_WEIGHTS)

    emotions_labels = [
        "nervoso",
        "nojo",
        "medo",
        "feliz",
        "neutro",
        "triste",
        "surpreso",
    ]

    while True:
        bol, frame = input_stream.read()

        frame_resized = cv2.resize(
            frame,
            (0, 0),
            fx=1 / SCALE_FACTOR,
            fy=1 / SCALE_FACTOR,
            interpolation=cv2.INTER_AREA,
        )

        face_locations = face_recognition.face_locations(
            img=frame_resized,
            number_of_times_to_upsample=1,
            model="hog",
        )

        for face, face_location in enumerate(face_locations):
            top, right, bottom, left = face_location

            top *= SCALE_FACTOR
            right *= SCALE_FACTOR
            bottom *= SCALE_FACTOR
            left *= SCALE_FACTOR

            print(f"Face {face+1}, top: {top}, right: {right}, bottom: {bottom}, left: {left}")

            face_image = frame[top:bottom, left:right]
            face_image = cv2.resize(face_image, (IMAGE_SIZE, IMAGE_SIZE))

            face_image = tf.keras.preprocessing.image.img_to_array(face_image)

            img_pixels = np.expand_dims(face_image, axis=0)
            img_pixels /= 255

            predictions = fer_model.predict(img_pixels)
            max_index = np.argmax(predictions[0])

            emotion = emotions_labels[max_index]

            cv2.putText(
                frame,
                emotion.upper(),
                (left, top - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (255, 0, 0),
                2,
            )

        cv2.imshow("Video Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    input_stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
