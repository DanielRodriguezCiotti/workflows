import base64
import json

from .helpers import ImageConverter


class FaceJob:
    def process_inputs(input_data):
        # Extract input data
        assert "model_img" in input_data, "Missing 'model_img' key in input data"
        assert "generation_type" in input_data, (
            "Missing 'generation_type' key in input data"
        )
        assert "inpaint_params" in input_data, (
            "Missing 'inpaint_params' key in input data"
        )
        assert "prompt" in input_data, "Missing 'prompt' key in input data"

        model_img = input_data.get("model_img")
        inpaint_params = input_data.get("inpaint_params")
        generation_type = input_data.get("generation_type")
        prompt = input_data.get("prompt")

        model_img = ImageConverter.to_bytes(model_img)
        model_file = ("model.png", model_img, "image/png")
        files = {"model_img_buffer": model_file}

        # Format the input data
        generation_data = {
            "inpaint_params": inpaint_params,
            "generation_type": generation_type,
            "prompt": prompt,
        }
        form_data = {"generation_data": json.dumps(generation_data)}

        return files, form_data

    def process_outputs(response):
        # Process the response
        result_data = response.json()
        result = result_data.get("result")
        decoded_result = base64.b64decode(result)
        result_img = ImageConverter.from_bytes(decoded_result)
        return result_img


class MaskJob:
    def process_inputs(input_data):
        # Extract input data
        assert "category" in input_data, "Missing 'category' key in input_data"
        assert "model_img" in input_data, "Missing 'model_img' key in input_data"

        model_img = input_data.get("model_img")
        category = input_data.get("category")

        model_img = ImageConverter.to_bytes(model_img)
        model_file = ("model.png", model_img, "image/png")
        files = {"model_img_buffer": model_file}

        # Format the input data
        generation_data = {
            "category": category,
        }
        form_data = {"generation_data": json.dumps(generation_data)}

        return files, form_data

    def process_outputs(response):
        # Process the response
        result_data = response.json()
        result = result_data.get("result")
        decoded_result = base64.b64decode(result)
        result_img = ImageConverter.from_bytes(decoded_result)
        return result_img


class TryOnJob:
    def process_inputs(input_data):
        # Extract input data
        assert "category" in input_data, "Missing 'category' key in input_data"
        assert "model_img" in input_data, "Missing 'model_img' key in input_data"
        assert "cloth_img" in input_data, "Missing 'cloth_img' key in input_data"
        assert "mask_img" in input_data, "Missing 'mask_img' key in input_data"

        model_img = input_data.get("model_img")
        cloth_img = input_data.get("cloth_img")
        mask_img = input_data.get("mask_img")
        category = input_data.get("category")

        model_img = ImageConverter.to_bytes(model_img)
        model_file = ("model.png", model_img, "image/png")
        cloth_img = ImageConverter.to_bytes(cloth_img)
        cloth_file = ("cloth.png", cloth_img, "image/png")
        if mask_img is not None:
            mask_img = ImageConverter.to_bytes(mask_img)
        mask_file = ("mask.png", mask_img, "image/png")
        files = {
            "model_img_buffer": model_file,
            "cloth_img_buffer": cloth_file,
            "mask_img_buffer": mask_file,
        }

        # Format the input data
        generation_data = {
            "category": category,
        }
        form_data = {"generation_data": json.dumps(generation_data)}

        return files, form_data

    def process_outputs(response):
        # Process the response
        result_data = response.json()
        result = result_data.get("result")
        decoded_result = base64.b64decode(result)
        result_img = ImageConverter.from_bytes(decoded_result)
        return result_img


class HandsFixJob:
    def process_inputs(input_data):
        # Extract input data
        assert "model_img" in input_data, "Missing 'model_img' key in input_data"

        model_img = input_data.get("model_img")
        model_img = ImageConverter.to_bytes(model_img)
        model_file = ("model.png", model_img, "image/png")
        files = {
            "model_img_buffer": model_file,
        }
        form_data = None

        return files, form_data

    def process_outputs(response):
        # Process the response
        result_data = response.json()
        result = result_data.get("result")
        decoded_result = base64.b64decode(result)
        result_img = ImageConverter.from_bytes(decoded_result)
        return result_img


class RetouchJob:
    def process_inputs(input_data):
        # Extract input data
        assert "model_img" in input_data, "Missing 'model_img' key in input_data"

        model_img = input_data.get("model_img")
        model_img = ImageConverter.to_bytes(model_img)
        model_file = ("model.png", model_img, "image/png")
        files = {
            "model_img_buffer": model_file,
        }
        generation_data = {
            "seed": input_data.get("seed", None),
        }
        form_data = {"generation_data": json.dumps(generation_data)}

        return files, form_data

    def process_outputs(response):
        # Process the response
        result_data = response.json()
        result = result_data.get("result")
        decoded_result = base64.b64decode(result)
        result_img = ImageConverter.from_bytes(decoded_result)
        return result_img


class ModelGenerationJob:
    def process_inputs(input_data):
        # Extract input data
        assert "prompt" in input_data, "Missing 'prompt' key in input data"

        prompt = input_data.get("prompt")
        seed = input_data.get("seed", None)
        files = None
        # Format the input data
        generation_data = {
            "prompt": prompt,
            "seed": seed,
        }
        form_data = {"generation_data": json.dumps(generation_data)}

        return files, form_data

    def process_outputs(response):
        # Process the response
        result_data = response.json()
        result = result_data.get("result")
        decoded_result = base64.b64decode(result)
        result_img = ImageConverter.from_bytes(decoded_result)
        return result_img
