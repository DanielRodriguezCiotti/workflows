import os
from enum import Enum
from io import BytesIO
from typing import Literal, Optional, TypeVar
from urllib.parse import urlparse

import boto3
from PIL import Image
from prefect.logging import get_logger

logger = get_logger("s3")


class ExtensionsEnum(Enum):
    WEBP = ".webp"
    PNG = ".png"
    JPG = ".jpg"
    JPEG = ".jpeg"
    TIFF = ".tiff"
    BLOSC2 = ".blosc2"
    TXT = ".txt"
    JSON = ".json"
    CSV = ".csv"

    @staticmethod
    def str_to_extension(extension):
        if extension.lower() == "webp":
            return ExtensionsEnum.WEBP
        elif extension.lower() == "png":
            return ExtensionsEnum.PNG
        elif extension.lower() == "jpg":
            return ExtensionsEnum.JPG
        elif extension.lower() == "jpeg":
            return ExtensionsEnum.JPEG
        elif extension.lower() == "tiff":
            return ExtensionsEnum.TIFF
        elif extension.lower() == "blosc2":
            return ExtensionsEnum.BLOSC2
        elif extension.lower() == "txt":
            return ExtensionsEnum.TXT
        elif extension.lower() == "json":
            return ExtensionsEnum.JSON
        elif extension.lower() == "csv":
            return ExtensionsEnum.CSV
        else:
            raise ValueError(f"Invalid extension: {extension}")


extension_type = TypeVar(
    "extension_type",
    bound=Literal[
        ExtensionsEnum.WEBP,
        ExtensionsEnum.PNG,
        ExtensionsEnum.JPG,
        ExtensionsEnum.JPEG,
        ExtensionsEnum.TIFF,
    ],
)


def download_file(
    uri: str, output_folder: str, s3_client=None, new_name: Optional[str] = None
) -> tuple[str, str]:
    """
    Download a single file from S3.
    Args:
        uri: S3 object uri
    Returns:
        Tuple of (local_path, status)
        status: "skipped" if file already exists, "success" if file was downloaded, "fail" if error occurred
    """
    parsed_uri = urlparse(uri)
    # Validate S3 URI format
    if parsed_uri.scheme != "s3":
        raise ValueError(f"Invalid S3 URI: {uri}")
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    if new_name is not None:
        extension = os.path.splitext(os.path.basename(parsed_uri.path))[1]
        local_path = os.path.join(output_folder, new_name + extension)
    else:
        local_path = os.path.join(output_folder, os.path.basename(parsed_uri.path))
    # Check if file already exists
    if os.path.exists(local_path):
        return local_path, "skipped"
    # Create S3 client if not provided
    if s3_client is None:
        s3_client = boto3.client("s3")
    try:
        bucket_name = parsed_uri.netloc
        key = parsed_uri.path.lstrip("/")
        s3_client.download_file(
            bucket_name,
            key,
            local_path,
        )
        return local_path, "success"

    except Exception as e:
        logger.error(f"Error downloading {uri}: {e}")
        return local_path, "fail"


def upload_file(local_path: str, s3_uri: str, s3_client=None) -> tuple[str, str]:
    """
    Upload a single file to S3.
    Args:
        local_path: Local path to the file
        s3_uri: S3 object uri
    Returns:
        Tuple of (s3_uri, status)
        status: "skipped" if file already exists, "success" if file was uploaded, "fail" if error occurred
    """
    parsed_uri = urlparse(s3_uri)
    # Validate S3 URI format
    if parsed_uri.scheme != "s3":
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    # Create S3 client if not provided
    if s3_client is None:
        s3_client = boto3.client("s3")
    try:
        bucket_name = parsed_uri.netloc
        key = parsed_uri.path.lstrip("/")
        s3_client.upload_file(
            local_path,
            bucket_name,
            key,
        )
        return s3_uri, "success"
    except Exception as e:
        logger.error(f"Error uploading {local_path}: {e}")
        return s3_uri, "fail"


def download_image(uri: str, s3_client=None) -> tuple[Optional[Image.Image], str]:
    """
    Download an image from S3 and verify its validity.
    Args:
        uri: S3 object uri
    Returns:
        Tuple of (Image object, status)
        status: "skipped" if image already exists, "success" if image was downloaded and is valid, "fail" if error occurred
    """
    parsed_uri = urlparse(uri)
    # Validate S3 URI format
    if parsed_uri.scheme != "s3":
        raise ValueError(f"Invalid S3 URI: {uri}")
    # Create S3 client if not provided
    if s3_client is None:
        s3_client = boto3.client("s3")
    try:
        bucket_name = parsed_uri.netloc
        key = parsed_uri.path.lstrip("/")
        # Check if the extension is valid
        extension = os.path.splitext(key)[1].lower()
        if extension not in [e.value for e in ExtensionsEnum]:
            raise ValueError(f"Invalid file extension: {extension}")
        # Download the file into a buffer
        buffer = BytesIO()
        s3_client.download_fileobj(bucket_name, key, buffer)
        buffer.seek(0)
        # Verify the image
        image = Image.open(buffer)
        image.verify()  # This will raise an exception if the image is not valid
        return image, "success"
    except Exception as e:
        logger.error(f"Error downloading {uri}: {e}")
        return None, "fail"


def upload_image(image: Image.Image, s3_uri: str, s3_client=None) -> tuple[str, str]:
    """
    Upload an image to S3 from a buffer.
    Args:
        image: Image object to upload
        s3_uri: S3 object uri
    Returns:
        Tuple of (s3_uri, status)
        status: "success" if image was uploaded, "fail" if error occurred
    """
    parsed_uri = urlparse(s3_uri)
    # Validate S3 URI format
    if parsed_uri.scheme != "s3":
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    # Create S3 client if not provided
    if s3_client is None:
        s3_client = boto3.client("s3")
    try:
        bucket_name = parsed_uri.netloc
        key = parsed_uri.path.lstrip("/")
        # Check if the image format is valid
        if image.format is None or f".{image.format.lower()}" not in [
            e.value for e in ExtensionsEnum
        ]:
            raise ValueError(f"Invalid image format: {image.format}")
        # Save the image to a buffer
        buffer = BytesIO()
        image.save(buffer, format=image.format)
        buffer.seek(0)
        # Upload the buffer to S3
        s3_client.upload_fileobj(buffer, bucket_name, key)
        return s3_uri, "success"
    except Exception as e:
        logger.error(f"Error uploading image to {s3_uri}: {e}")
        return s3_uri, "fail"
