import yaml
from PIL import Image
from prefect import flow, task
from prefect.logging import get_logger
from prefect_aws import AwsCredentials

from client import DummyJobClient
from s3 import download_image, upload_image

aws_credentials_block = AwsCredentials.load("aws-credentials")
logger = get_logger("dummy-try-on-workflow")


@task(
    description="Pull a garment image from S3",
    name="pull-garment-image",
    retries=3,
    retry_delay_seconds=10,
)
def pull_garment_image(garment_uri: str) -> Image.Image:
    image, status = download_image(garment_uri)
    if status == "fail" or image is None:
        logger.error(f"Failed to download garment image from {garment_uri}")
        raise ValueError("Failed to download garment image")
    logger.info(f"Downloaded garment image from {garment_uri}")
    return image


@task(
    description="Generate a model from a prompt",
    name="generate-model",
    retries=3,
    retry_delay_seconds=10,
)
def generate_model(endpoint: str, prompt: str) -> Image.Image:
    client = DummyJobClient(endpoint, job_type="model_generation_job")
    result_img = client.run_job(input_data={"prompt": prompt})
    if result_img is None:
        logger.error("Job failed to produce a result")
        raise ValueError("Job failed to produce a result")
    logger.info("Generated model from prompt")
    return result_img


@task(
    description="Generate a mask from a model and a category",
    name="generate-mask",
    retries=3,
    retry_delay_seconds=10,
)
def generate_mask(endpoint: str, model: Image.Image, category: str) -> Image.Image:
    client = DummyJobClient(endpoint, job_type="mask_job")
    result = client.run_job(
        input_data={
            "model_img": model,
            "category": category,
        }
    )
    if result is None:
        logger.error("Masking job failed to produce a result")
        raise ValueError("Masking job failed to produce a result")
    logger.info("Generated mask from model and category")
    return result


@task(
    description="Generate a try-on image from a model, a mask, a garment, and a category",
    name="generate-tryon",
    retries=3,
    retry_delay_seconds=10,
)
def generate_tryon(
    endpoint: str,
    model: Image.Image,
    mask: Image.Image,
    garment: Image.Image,
    category: str,
) -> Image.Image:
    client = DummyJobClient(endpoint, job_type="tryon_job")
    result = client.run_job(
        input_data={
            "model_img": model,
            "mask_img": mask,
            "cloth_img": garment,
            "category": category,
        }
    )
    if result is None:
        logger.error("Try-on job failed to produce a result")
        raise ValueError("Try-on job failed to produce a result")
    logger.info("Generated try-on image from model, mask, garment, and category")
    return result


@task(
    description="Upload the try-on image to S3",
    name="push-tryon-to-s3",
    retries=2,
    retry_delay_seconds=10,
)
def push_tryon_to_s3(tryon_image: Image.Image, output_uri: str):
    uri, status = upload_image(tryon_image, output_uri)
    if status == "fail":
        logger.error("Failed to upload try-on image")
        raise ValueError("Failed to upload try-on image")
    logger.info(f"Uploaded try-on image to {output_uri}")
    return uri


@flow(
    description="Generate a try-on image from a garment image and a model prompt",
    name="dummy-try-on-workflow",
    retries=3,
    retry_delay_seconds=10,
    timeout_seconds=300,
)
def main_flow(
    garment_uri: str,
    model_prompt: str,
    category: str,
    output_uri: str,
    config_path: str,
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Parallel execution of garment image pull and model generation
    future_garment_image = pull_garment_image.submit(garment_uri)
    future_model_image = generate_model.submit(
        config["endpoints"]["model_generator"], model_prompt
    )
    garment_image = future_garment_image.result()
    model_image = future_model_image.result()

    # Sequential execution of mask generation and try-on generation
    mask_image = generate_mask(config["endpoints"]["masking"], model_image, category)
    tryon_image = generate_tryon(
        config["endpoints"]["tryon"],
        model_image,
        mask_image,
        garment_image,
        category,
    )
    push_tryon_to_s3(tryon_image, output_uri)


if __name__ == "__main__":
    import json
    from argparse import ArgumentParser

    with open("data/input.json", "r") as f:
        default_args = json.load(f)

    parser = ArgumentParser()
    parser.add_argument(
        "-g", "--garment-uri", type=str, default=default_args["garment_uri"]
    )
    parser.add_argument(
        "-m", "--model-prompt", type=str, default=default_args["model_prompt"]
    )
    parser.add_argument("-t", "--category", type=str, default=default_args["category"])
    parser.add_argument(
        "-o", "--output-uri", type=str, default=default_args["output_uri"]
    )
    parser.add_argument(
        "-c", "--config-path", type=str, default=default_args["config_path"]
    )
    args = parser.parse_args()
    main_flow(**vars(args))
