# annotate.py (do not change or remove this comment)

import os
import logging
from ultralytics.data.annotator import auto_annotate
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage.draw import polygon
from skimage.measure import find_contours, approximate_polygon

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Annotator:
    def __init__(self, image_file):
        self.image_file = image_file
        self.base_file, _ = os.path.splitext(image_file)
        self.file_name = os.path.basename(self.base_file)
        self.output_dir = "annotations"
        self.masked_dir = "masked"
        self.det_model = os.path.join(os.getcwd(), "models/object_detection_model.pt")
        self.sam_model = os.path.join(os.getcwd(), "models/mobile_sam.pt")
        self.image = self.load_image()
        self.original_shape = self.image.shape[:2]
        self.mask = None

        self.ensure_directories()

    def ensure_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.masked_dir, exist_ok=True)

    def perform_auto_annotation(self):
        auto_annotate(
            data=self.image_file,
            det_model=self.det_model,
            sam_model=self.sam_model,
            output_dir=self.output_dir,
        )

    def load_image(self):
        return mpimg.imread(self.image_file)

    def parse_annotation_data(self):
        annotation_file = os.path.join(self.output_dir, f"{self.file_name}.txt")
        with open(annotation_file, "r") as file:
            lines = file.readlines()

        # Find the line with the most characters
        longest_line = max(lines, key=len)
        segments = longest_line.split()[
            1:
        ]  # Skip the first value which is usually the class ID

        return np.array(segments, dtype=float).reshape(-1, 2)

    def create_mask(self, segments):
        if len(segments) > 2048:
            segments = approximate_polygon(segments, tolerance=0.01)[:1024]

        mask = np.zeros(self.original_shape)
        rr, cc = polygon(
            segments[:, 1] * self.original_shape[0],
            segments[:, 0] * self.original_shape[1],
        )
        mask[rr, cc] = 1
        return mask

    def apply_mask(self):
        masked_image = self.image.copy()
        masked_image[self.mask == 1] = [
            0,
            0,
            255,
        ]  # Set blue channel to 255, others to 0
        return masked_image

    def plot_and_save_image(self, masked_image, output_file):
        plt.figure(figsize=(self.original_shape[1] / 100, self.original_shape[0] / 100))
        plt.imshow(self.image)
        plt.imshow(
            masked_image,
            alpha=0.25,
            extent=[0, self.original_shape[1], self.original_shape[0], 0],
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close()

    def annotate_and_mask(self):
        self.perform_auto_annotation()
        segments = self.parse_annotation_data()
        self.mask = self.create_mask(segments)
        masked_image = self.apply_mask()
        output_file = os.path.join(self.masked_dir, f"Masked {self.file_name}.jpg")
        self.plot_and_save_image(masked_image, output_file)
        return masked_image

    def area(self):
        if self.mask is None:
            raise ValueError("No mask found. Please run annotate_and_mask() first.")
        return round(np.sum(self.mask), 4)

    def centroid(self):
        y_coords, x_coords = np.where(self.mask == 1)
        return np.mean(y_coords), np.mean(x_coords)

    def perimeter(self):
        if self.mask is None:
            raise ValueError("No mask found. Please run annotate_and_mask() first.")
        contours = find_contours(self.mask, level=0.5)
        return round(
            sum(
                np.linalg.norm(np.diff(contour, axis=0), axis=1).sum()
                for contour in contours
            ),
            4,
        )

    def length(self):
        if self.mask is None:
            raise ValueError("No mask found. Please run annotate_and_mask() first.")

        centroid_y, centroid_x = self.centroid()
        max_length = 0
        best_line = []

        for angle in np.linspace(0, np.pi, 180):
            dy, dx = np.sin(angle), np.cos(angle)
            length, line_points = self._calculate_line_length(
                centroid_y, centroid_x, dy, dx
            )

            if length > max_length:
                max_length = length
                best_line = line_points

        return round(max_length, 4), best_line

    def _calculate_line_length(self, y, x, dy, dx):
        length = 0
        line_points = []

        for direction in (1, -1):
            _y, _x = y, x
            while (
                0 <= _y < self.original_shape[0]
                and 0 <= _x < self.original_shape[1]
                and self.mask[int(_y), int(_x)] == 1
            ):
                line_points.append((_x, _y))
                length += 1
                _y += dy * direction
                _x += dx * direction

        return length, line_points

    def girth(self):
        if self.mask is None:
            raise ValueError("No mask found. Please run annotate_and_mask() first.")

        _, best_length_line = self.length()
        if not best_length_line:
            raise ValueError("No valid length line found within the mask.")

        centroid_y, centroid_x = self.centroid()
        angle_of_length = np.arctan2(
            best_length_line[-1][1] - best_length_line[0][1],
            best_length_line[-1][0] - best_length_line[0][0],
        )
        angle_perpendicular = angle_of_length + np.pi / 2

        dy, dx = np.sin(angle_perpendicular), np.cos(angle_perpendicular)
        max_girth, best_girth_line = self._calculate_line_length(
            centroid_y, centroid_x, dy, dx
        )

        return round(max_girth, 4), best_girth_line

    def visualize(self):
        if self.mask is None:
            raise ValueError("No mask found. Please run annotate_and_mask() first.")

        vis_image = self.image.copy()
        centroid_y, centroid_x = self.centroid()

        max_length, best_length_line = self.length()
        max_girth, best_girth_line = self.girth()

        plt.figure(figsize=(self.original_shape[1] / 100, self.original_shape[0] / 100))
        plt.imshow(vis_image)
        plt.imshow(self.mask, cmap="gray", alpha=0.5)
        plt.scatter([centroid_x], [centroid_y], c="red", marker="x")

        if best_length_line:
            best_length_line = np.array(best_length_line)
            plt.plot(best_length_line[:, 0], best_length_line[:, 1], "b-")

        if best_girth_line:
            best_girth_line = np.array(best_girth_line)
            plt.plot(best_girth_line[:, 0], best_girth_line[:, 1], "g-")

        plt.axis("off")
        plt.tight_layout()
        output_file = os.path.join(self.masked_dir, f"visualized-{self.file_name}.jpg")
        plt.savefig(output_file, bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close()
        logger.info(f"Visualization saved to {output_file}")
