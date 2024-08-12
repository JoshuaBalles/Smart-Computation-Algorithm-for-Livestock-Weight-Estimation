# camera_video_server.py (do not change or remove this comment)

import io
import logging
import socket
import socketserver
from http import server
from threading import Condition
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from libcamera import Transform

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class StreamingOutput(io.BufferedIOBase):
    # Class to handle streaming output from the camera

    def __init__(self):
        self.frame = None
        self.condition = Condition()
        logging.info("Initialized StreamingOutput")

    def write(self, buf):
        # Write a new frame to the buffer and notify all waiting threads
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


class StreamingHandler(server.BaseHTTPRequestHandler):
    # Class to handle HTTP requests for streaming

    def do_GET(self):
        # Serve the MJPEG stream to the client
        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", 0)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=FRAME"
            )
            self.end_headers()
            logging.info("Started streaming to client")
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    # Directly write frame to output stream without additional copying
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except Exception as e:
                logging.info(
                    "Removed streaming client %s: %s", self.client_address, str(e)
                )
        else:
            self.send_error(404)
            self.end_headers()
            logging.info("File not found: %s", self.path)


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    # HTTP server with threading mixin for handling requests in separate threads
    allow_reuse_address = True
    daemon_threads = True


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


# Initialize the camera
picam2 = Picamera2()
picam2.configure(
    picam2.create_video_configuration(
        main={"size": (1640, 1232)}, transform=Transform(vflip=True, hflip=False)
    )
)
picam2.set_controls({"FrameRate": 24.0})
logging.info("Camera configured and controls set")

# Set up the streaming output
output = StreamingOutput()
picam2.start_recording(JpegEncoder(), FileOutput(output))
logging.info("Started recording")

try:
    # Start the streaming server
    address = ("", 8000)
    server = StreamingServer(address, StreamingHandler)
    ip_address = get_ip_address()
    logging.info("Streaming at http://%s:8000/stream.mjpg", ip_address)
    server.serve_forever()
finally:
    # Stop recording when the server is stopped
    picam2.stop_recording()
    logging.info("Stopped recording")
