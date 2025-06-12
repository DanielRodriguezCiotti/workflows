import base64
import io
import os

import numpy as np
from PIL import Image


def rgba_to_rgb_white_background(img):
    """
    Convert RGBA PIL Image to RGB PIL Image with white background

    Args:
        img: PIL.Image in RGBA mode
    Returns:
        PIL.Image in RGB mode
    """
    # Check if image has alpha channel
    if img.mode != "RGBA":
        return img

    # Create white background image
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))

    # Composite input image over white background
    composite = Image.alpha_composite(background, img)

    # Convert to RGB
    return composite.convert("RGB")


class ImageConverter:
    """
    A utility class for converting images between different formats:
    - str (file path)
    - numpy.ndarray
    - PIL.Image
    - bytes
    """

    @staticmethod
    def to_bytes(image_input):
        """
        Convert various image formats to bytes.

        Args:
            image_input: Can be one of:
                - str: Path to image file
                - numpy.ndarray: Image array
                - PIL.Image: PIL Image object
                - bytes: Already in bytes format

        Returns:
            bytes: Image in bytes format
        """
        if isinstance(image_input, bytes):
            return image_input

        if isinstance(image_input, str):
            if os.path.isfile(image_input):
                image_input = Image.open(image_input)
                byte_arr = io.BytesIO()
                image_input.save(byte_arr, format="PNG")
                return byte_arr.getvalue()
            else:
                try:
                    # Try if it's a base64 string
                    return base64.b64decode(image_input)
                except:
                    raise ValueError(
                        "String input must be either a valid file path or base64 encoded image"
                    )

        if isinstance(image_input, np.ndarray):
            # Convert numpy array to PIL Image
            image = Image.fromarray(image_input.astype("uint8"))
            byte_arr = io.BytesIO()
            image.save(byte_arr, format="PNG")
            return byte_arr.getvalue()

        if isinstance(image_input, Image.Image):
            byte_arr = io.BytesIO()
            image_input.save(byte_arr, format="PNG")
            return byte_arr.getvalue()

        raise ValueError(f"Unsupported input type: {type(image_input)}")

    @staticmethod
    def from_bytes(image_bytes, target_type="PIL"):
        """
        Convert bytes to the specified image format.

        Args:
            image_bytes (bytes): Image in bytes format
            target_type (str): Desired output format ('PIL', 'numpy', 'tensor', or 'bytes')

        Returns:
            Image in the specified format
        """
        if target_type.lower() == "bytes":
            return image_bytes

        # First convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        if target_type.lower() == "pil":
            return image

        if target_type.lower() == "numpy":
            return np.array(image)

        raise ValueError(f"Unsupported target type: {target_type}")

    @staticmethod
    def convert(image_input, target_type="PIL"):
        """
        Convert between any supported image formats.

        Args:
            image_input: Input image in any supported format
            target_type (str): Desired output format

        Returns:
            Image in the specified format
        """
        image_bytes = ImageConverter.to_bytes(image_input)
        return ImageConverter.from_bytes(image_bytes, target_type)


class ServerTimeoutError(Exception):
    """Custom exception for timeout errors."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"ServerTimeoutError: {self.message}"
