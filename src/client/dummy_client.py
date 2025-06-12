import random
import time

from loguru import logger
from PIL import Image

fake_data = {
    "model_generation_job": "data/model.webp",
    "mask_job": "data/mask.webp",
    "tryon_job": "data/tryon.webp",
}
fake_time_waiting = {
    "model_generation_job": 125,
    "mask_job": 7,
    "tryon_job": 45,
}


class DummyJobClient:
    def __init__(self, server_url, job_type, max_timeout=600):
        assert job_type in fake_data, f"Invalid job type: {job_type}"
        self.job_type = job_type

    def run_job(self, input_data, retry=3):
        start = time.time()
        # Process the inputs
        attempt = 0
        while attempt < retry:
            try:
                # Send the request
                logger.info(f"Sending request... Attempt {attempt + 1}")
                time_to_wait = fake_time_waiting[self.job_type] * random.uniform(
                    0.5, 1.5
                )
                # Fail with 10% probability
                if random.random() < 0.1:
                    raise Exception("Failed to run the job")
                time.sleep(time_to_wait)
                # Process the response
                result_img = Image.open(fake_data[self.job_type])
                logger.info(f"Time taken to run the job: {time.time() - start}")
                return result_img
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                attempt += 1
                if attempt < retry:
                    logger.info("Retrying...")
                    time.sleep(2)  # Wait for 2 seconds before retrying
                else:
                    logger.error("All retry attempts failed.")
                    raise
