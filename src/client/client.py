import time

import requests
from loguru import logger

from .jobs import (
    FaceJob,
    HandsFixJob,
    MaskJob,
    ModelGenerationJob,
    RetouchJob,
    TryOnJob,
)

job_mapping = {
    "face_job": FaceJob,
    "handsfix_job": HandsFixJob,
    "mask_job": MaskJob,
    "model_generation_job": ModelGenerationJob,
    "retouch_job": RetouchJob,
    "tryon_job": TryOnJob,
}


class JobClient:
    def __init__(self, server_url, job_type, max_timeout=600):
        assert job_type in job_mapping, f"Invalid job type: {job_type}"
        self.job_class = job_mapping[job_type]
        self.job_type = job_type
        self.server_url = server_url
        self.max_timeout = max_timeout

    def run_job(self, input_data, retry=3):
        start = time.time()
        # Process the inputs
        files, form_data = self.job_class.process_inputs(input_data=input_data)

        attempt = 0
        while attempt < retry:
            try:
                # Send the request
                logger.info(f"Sending request... Attempt {attempt + 1}")
                response = requests.post(
                    url=f"{self.server_url}/run_job",
                    files=files,
                    data=form_data,
                    timeout=600,
                )

                # Check for response error
                self._check_response(response=response)

                # Process the response
                result_img = self.job_class.process_outputs(response=response)
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

    def _check_health(self):
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _check_response(self, response):
        # Check response status
        if response.status_code != 200:
            # Try to parse response as JSON
            try:
                error_data = response.json()
                logger.error(f"\nError: {error_data.get('error', 'Unknown error')}")

                # Print the stack trace if available
                if "stack_trace" in error_data:
                    logger.error("\n--- Server Stack Trace ---")
                    logger.error(error_data["stack_trace"])
            except:
                # If not JSON, print raw response
                logger.error(f"\nRaw server response: {response.text}")
