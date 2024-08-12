import json
import logging
import os
import re
import socket
import sqlite3
import subprocess
import sys
from datetime import datetime
import time

from flask import flash
import joblib
import pandas as pd
import piexif
import plotly
import plotly.express as px
import torch
import torch.nn as nn
from flask import (
    Flask,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from libcamera import Transform
from picamera2 import Picamera2
from PIL import Image

from models import crop
from models.annotate import Annotator

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Add this line near the app initialization

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define the path to the video server script
VIDEO_SERVER_SCRIPT = os.path.join(os.getcwd(), "camera_video_server.py")

# Use the Python interpreter from the current environment
python_interpreter = sys.executable

# Global variable to hold the video server process
video_server_process = None

# Create cropped directory if it doesn't exist
CROPPED_DIR = os.path.join(os.getcwd(), "cropped")
if not os.path.exists(CROPPED_DIR):
    os.makedirs(CROPPED_DIR)


def get_ip_address():
    # Get the IP address of the local machine
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 1))
        ip_address = s.getsockname()[0]
    except Exception:
        ip_address = "127.0.0.1"
    finally:
        s.close()
    logging.info("IP address determined: %s", ip_address)
    return ip_address


@app.context_processor
def inject_ip_address():
    ip_address = get_ip_address()
    return dict(ip_address=ip_address)


# Define the model
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.linear1 = nn.Linear(4, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x


# Load the pig scaler and model state dict
pig_checkpoint = joblib.load(os.path.join("models", "p_model_scaler.joblib"))
pig_scaler = pig_checkpoint["scaler"]
pig_model = RegressionModel()
pig_model.load_state_dict(pig_checkpoint["model"])
pig_model.eval()

# Load the chicken scaler and model state dict
chicken_checkpoint = joblib.load(os.path.join("models", "c_model_scaler.joblib"))
chicken_scaler = chicken_checkpoint["scaler"]
chicken_model = RegressionModel()
chicken_model.load_state_dict(chicken_checkpoint["model"])
chicken_model.eval()


# Function to estimate weight
def estimate_weight(features, scaler, model):
    features = scaler.transform([features])
    features = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        estimation = model(features)
    return estimation.item()


# Function to start the video server
def start_video_server():
    global video_server_process
    if video_server_process is None:
        logging.info("Starting video server...")
        video_server_process = subprocess.Popen(
            [python_interpreter, VIDEO_SERVER_SCRIPT]
        )
    else:
        logging.info("Video server already running.")


# Function to stop the video server
def stop_video_server():
    global video_server_process
    if video_server_process is not None:
        logging.info("Stopping video server...")
        video_server_process.terminate()  # Send termination signal to the process
        video_server_process.wait()  # Wait for the process to terminate
        logging.info("Video server stopped.")
        video_server_process = None
    else:
        logging.info("No video server process to stop.")


# Function to release the camera resources
def release_camera(picam2):
    try:
        picam2.stop()
        picam2.close()
        logging.info("Camera released.")
    except Exception as e:
        logging.error(f"Error releasing camera: {e}")


# Function to capture an image and save it with metadata,
# then perform object detection on the saved image
def capture_image():
    # Get current timestamp
    now = datetime.now()
    timestamp = now.strftime("%B %d, %Y, %I-%M-%S %p")

    # Initialize Picamera2
    picam2 = Picamera2()

    # Configure the camera with vertical and horizontal flip
    config = picam2.create_still_configuration(
        transform=Transform(vflip=True, hflip=False)
    )
    picam2.configure(config)

    # Start the camera
    picam2.start()

    # Capture an image into memory
    logging.info("Capturing image...")
    buffer = picam2.capture_array()

    # Save the image as JPEG with EXIF metadata
    image = Image.fromarray(buffer)
    image_path = os.path.join(os.getcwd(), "capture.jpg")

    # Create EXIF metadata
    exif_dict = {
        "0th": {},
        "Exif": {},
        "1st": {},
        "thumbnail": None,
        "GPS": {},
        "Interop": {},
    }
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = timestamp

    exif_bytes = piexif.dump(exif_dict)

    # Save the image with EXIF metadata
    image.save(image_path, "jpeg", exif=exif_bytes)
    logging.info(f"Saved image at {image_path} with metadata: Captured={timestamp}")

    # Perform object detection on the saved image
    crop.crop_objects("capture.jpg")

    # Release the camera
    release_camera(picam2)


@app.before_request
def before_request():
    if request.endpoint == "capture":
        start_video_server()


@app.after_request
def after_request(response):
    if request.endpoint != "capture":
        stop_video_server()
    return response


@app.route("/")
def index():
    # Render the main home page.
    return render_template("home.html")


@app.route("/capture", methods=["GET", "POST"])
def capture():
    if request.method == "POST":
        # Stop the video server before capturing the image
        stop_video_server()

        # Capture image
        logging.info("Capturing image...")
        # capture_image()
        time.sleep(5)

        # Restart the video server
        start_video_server()

        logging.info("Capture complete and video server restarted.")
        return redirect(url_for("results"))

    return render_template("capture.html")


from flask import flash


@app.route("/results")
def results():
    # Get all cropped images from the cropped/ directory
    cropped_dir = os.path.join(os.getcwd(), "cropped")
    images = [
        {"filename": f, "basename": os.path.splitext(f)[0]}
        for f in os.listdir(cropped_dir)
        if f.endswith((".jpg", ".jpeg", ".png"))
    ]

    # Extract datetime from the basename and sort images by datetime in descending order
    def extract_datetime(basename):
        match = re.search(r"(\w+ \d{2}, \d{4} \d{2}-\d{2}-\d{2} (AM|PM))", basename)
        if match:
            return datetime.strptime(match.group(1), "%B %d, %Y %I-%M-%S %p")
        return datetime.min

    images.sort(key=lambda x: extract_datetime(x["basename"]), reverse=True)

    return render_template("results.html", images=images)


@app.route("/result/estimate/<basename>")
def estimate(basename):
    # Find the full filename in the cropped directory
    cropped_dir = os.path.join(os.getcwd(), "cropped")
    filename = None
    for f in os.listdir(cropped_dir):
        if os.path.splitext(f)[0] == basename:
            filename = f
            break

    if not filename:
        return "Image not found", 404

    # Fetch the estimated weight and date from the database
    conn = sqlite3.connect("stats.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT estimated_weight, date FROM cropped_images WHERE name = ?", (filename,)
    )
    result = cur.fetchone()

    if result:
        estimated_weight = result[0]
        date_captured = result[1]
    else:
        estimated_weight = 0
        date_captured = "Unknown"

    if estimated_weight == 0:
        # Perform weight estimation if the weight is 0 and debugging is disabled
        annotator = Annotator(os.path.join(cropped_dir, filename))
        annotator.annotate_and_mask()
        area = annotator.area()
        perimeter = annotator.perimeter()
        length, _ = annotator.length()
        girth, _ = annotator.girth()
        features = [area, perimeter, length, girth]

        # Select appropriate model and scaler
        if "Pig" in basename:
            logging.info(f"Using pig weight estimation model.")
            estimated_weight = round(
                estimate_weight(features, pig_scaler, pig_model), 9
            )
        elif "Chicken" in basename:
            logging.info(f"Using chicken weight estimation model.")
            estimated_weight = round(
                estimate_weight(features, chicken_scaler, chicken_model), 9
            )

        # Update the database with the estimated weight
        cur.execute(
            "UPDATE cropped_images SET estimated_weight = ? WHERE name = ?",
            (estimated_weight, filename),
        )
        conn.commit()

    conn.close()

    return render_template(
        "estimate.html",
        basename=basename,
        filename=filename,
        estimated_weight=estimated_weight,
        date_captured=date_captured,
    )


@app.route("/delete/<basename>", methods=["POST"])
def delete_image(basename):
    # Find the full filename in the cropped directory
    cropped_dir = os.path.join(os.getcwd(), "cropped")
    filename = None
    for f in os.listdir(cropped_dir):
        if os.path.splitext(f)[0] == basename:
            filename = f
            break

    if not filename:
        return "Image not found", 404

    # Delete the image file
    file_path = os.path.join(cropped_dir, filename)
    try:
        os.remove(file_path)
        logging.info(f"Deleted image: {file_path}")

        # Remove the corresponding entry from the database
        conn = sqlite3.connect("stats.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM cropped_images WHERE name = ?", (filename,))
        conn.commit()
        conn.close()
        logging.info(f"Deleted database entry for image: {filename}")

    except Exception as e:
        logging.error(f"Error deleting image: {e}")
        return f"Error deleting image: {e}", 500

    return redirect(url_for("results"))


@app.route("/cropped/<filename>")
def get_cropped_image(filename):
    return send_from_directory(CROPPED_DIR, filename)


@app.route("/manage")
def manage():
    conn = sqlite3.connect("stats.db")
    cur = conn.cursor()

    # Fetch chicken data
    cur.execute(
        "SELECT name, estimated_weight FROM cropped_images WHERE name LIKE '%Chicken%'"
    )
    chickens = cur.fetchall()

    # Fetch pig data
    cur.execute(
        "SELECT name, estimated_weight FROM cropped_images WHERE name LIKE '%Pig%'"
    )
    pigs = cur.fetchall()

    conn.close()

    # Prepare chicken data for the table and pie chart
    chicken_data = []
    chicken_sizes = {"Off": 0, "Regular": 0, "Prime": 0}
    for name, weight in chickens:
        if weight <= 1.49:
            size = "Off"
        elif weight <= 1.69:
            size = "Regular"
        else:
            size = "Prime"
        chicken_sizes[size] += 1
        chicken_data.append({"name": name, "weight": weight, "size": size})

    # Prepare pig data for the table and pie chart
    pig_data = []
    pig_sizes = {"Weaner": 0, "Grower": 0, "Finisher": 0}
    for name, weight in pigs:
        if weight <= 39:
            size = "Weaner"
        elif weight <= 65:
            size = "Grower"
        else:
            size = "Finisher"
        pig_sizes[size] += 1
        pig_data.append({"name": name, "weight": weight, "size": size})

    # Create pie charts using Plotly
    chicken_labels = [f"{size} ({count})" for size, count in chicken_sizes.items()]
    chicken_values = list(chicken_sizes.values())
    chicken_fig = px.pie(
        names=chicken_labels,
        values=chicken_values,
        title="Chicken Size Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    chicken_graph = json.dumps(chicken_fig, cls=plotly.utils.PlotlyJSONEncoder)

    pig_labels = [f"{size} ({count})" for size, count in pig_sizes.items()]
    pig_values = list(pig_sizes.values())
    pig_fig = px.pie(
        names=pig_labels,
        values=pig_values,
        title="Pig Size Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    pig_graph = json.dumps(pig_fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "manage.html",
        chickens=chicken_data,
        pigs=pig_data,
        chicken_graph=chicken_graph,
        pig_graph=pig_graph,
    )


@app.route("/result/estimate_modal/<path:basename>")
def estimate_modal(basename):
    logging.info(f"Processing modal request for: {basename}")
    cropped_dir = os.path.join(os.getcwd(), "cropped")
    filename = basename

    if not os.path.exists(os.path.join(cropped_dir, filename)):
        logging.error(f"Image not found: {basename}")
        return "Image not found", 404

    conn = sqlite3.connect("stats.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT estimated_weight, date FROM cropped_images WHERE name = ?", (filename,)
    )
    result = cur.fetchone()

    if result:
        estimated_weight = result[0]
        date_captured = result[1]
    else:
        estimated_weight = 0
        date_captured = "Unknown"

    if estimated_weight == 0:
        annotator = Annotator(os.path.join(cropped_dir, filename))
        annotator.annotate_and_mask()
        area = annotator.area()
        perimeter = annotator.perimeter()
        length, _ = annotator.length()
        girth, _ = annotator.girth()
        features = [area, perimeter, length, girth]

        if "Pig" in basename:
            estimated_weight = estimate_weight(features, pig_scaler, pig_model)
        elif "Chicken" in basename:
            estimated_weight = estimate_weight(features, chicken_scaler, chicken_model)

        cur.execute(
            "UPDATE cropped_images SET estimated_weight = ? WHERE name = ?",
            (estimated_weight, filename),
        )
        conn.commit()

    conn.close()

    return render_template(
        "estimate_modal.html",
        basename=basename,
        filename=filename,
        estimated_weight=estimated_weight,
        date_captured=date_captured,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
