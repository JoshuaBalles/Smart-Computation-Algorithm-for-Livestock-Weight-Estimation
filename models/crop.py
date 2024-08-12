# models/crop.py (do not change or remove this comment)

from datetime import datetime
import os
import cv2
import logging
import sqlite3
from ultralytics import YOLO
from PIL import Image
import piexif

logging.basicConfig(level=logging.INFO)

# Define a dictionary to map class IDs to class names
class_names = {
    0: "Chicken",
    1: "Pig",
}


def crop_objects(image_path):
    def get_captured_metadata(image_path):
        try:
            image = Image.open(image_path)
            exif_data = piexif.load(image.info["exif"])
            captured_metadata = (
                exif_data["0th"]
                .get(piexif.ImageIFD.ImageDescription, b"Unknown")
                .decode("utf-8")
            )
            return captured_metadata
        except Exception as e:
            logging.error(f"Failed to get metadata from image: {e}")
            return "Unknown"

    def extract_datetime_from_metadata(metadata):
        try:
            return datetime.strptime(metadata, "%B %d, %Y, %I-%M-%S %p")
        except ValueError:
            logging.error(f"Failed to parse date from metadata: {metadata}")
            return None

    captured_metadata = get_captured_metadata(image_path)
    logging.info(f"Extracted metadata: Captured={captured_metadata}")

    date_captured = extract_datetime_from_metadata(captured_metadata)
    date_str = date_captured.strftime("%Y-%m-%d %H:%M:%S") if date_captured else None

    model_path = os.path.join("models", "object_detection_model.pt")
    logging.info(f"Loading YOLOv8 model from {model_path}")
    model = YOLO(model_path)

    logging.info(f"Loading image from {image_path}")
    image = cv2.imread(image_path)

    logging.info("Performing inference on the image")
    results = model(image)

    if len(results[0].boxes.cls) == 0:
        logging.info("No objects detected in the image.")
    else:
        logging.info("Extracting bounding boxes and class IDs")
        boxes = results[0].boxes.xyxy.tolist()
        class_ids = results[0].boxes.cls.tolist()

        conn = sqlite3.connect("stats.db")
        cur = conn.cursor()

        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
            logging.info(f"Processing object {i + 1} with class ID {class_id}")
            x1, y1, x2, y2 = box
            x1 -= 32
            y1 -= 32
            x2 += 32
            y2 += 32
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            cropped_object = image[int(y1) : int(y2), int(x1) : int(x2)]
            class_name = class_names.get(int(class_id), "Unknown")
            filename = f"{class_name} - {captured_metadata} {i}.jpg"
            cropped_folder = "cropped"
            os.makedirs(cropped_folder, exist_ok=True)
            save_path = os.path.join(cropped_folder, filename)
            cv2.imwrite(save_path, cropped_object)
            logging.info(f"Cropped object saved to {save_path}")

            cur.execute(
                """
            INSERT INTO cropped_images (id, name, date)
            VALUES (?, ?, ?)
            """,
                (None, filename, date_str),
            )

        conn.commit()
        conn.close()
