# Individual inference methods for InternLM X-Composer 2.5 7B
from typing import Dict
import requests
import os
import shutil
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms.functional as F
from transformers import AutoModel, AutoTokenizer

from backends.multimodal_utils.base_utils import BaseVLM

IMG_CACHE = 'image_cache'

class InternVLM(BaseVLM):

    @staticmethod
    def custom_padding(img):
        padding = (0, 0, 0, 0)
        num_channels = len(img.getbands())
        if num_channels == 4:  # RGBA
            fill_color = (255, 255, 255, 255)
        elif num_channels == 3:  # RGB
            fill_color = (255, 255, 255)
        else:
            raise ValueError(f"Unsupported number of channels: {num_channels}")

        padded_img = F.pad(img, padding, fill=fill_color)

        if padded_img.mode == 'RGBA':
            padded_img = padded_img.convert('RGB')  # Convert RGBA to RGB

        return padded_img

    def prepare_inputs(self, messages: list[Dict], **kwargs):
        """
        Returns a separate history, the prompt and a list of images to be passed to the model

        :param messages: A list[Dict] type object passed to the backend containing 'role', 'content' and 'image'
        :return history, prompt, image: Inputs to the model
        """
        history = []
        image = []
        image_counter = 0
        prev_user_msg = ""

        for m in messages:
            if m['role'] == 'user':
                prev_user_msg = m['content']
                if 'image' in m:
                    if isinstance(m['image'], str):
                        # A single image is passed
                        image_counter += 1
                        prev_user_msg = f"Image{image_counter} <ImageHere>; " + prev_user_msg
                        image.append(m['image'])
                    elif isinstance(m['image'], list):
                        # A list of images is passed
                        for img in m['image']:
                            image_counter += 1
                            prev_user_msg = f"Image{image_counter} <ImageHere>; " + prev_user_msg
                            image.append(img)
                    else:
                        print("Please pass a valid type of image in the message - Either a str or List[str]")

            elif m['role'] == 'assistant':
                # Append User+Assistant Message in Sequence
                history.append((prev_user_msg, m['content']))

        return {
            "prompt": prev_user_msg,
            "image": image,
            "kwargs": {"history": history}
        }

    def get_tokens(self, prompt: str, processor: AutoTokenizer, **kwargs):
        """
        Get the tokens passed to the model for context check

        :param prompt: The current prompt passed to the model
        :param processor: The processor used for the model
        :param kwargs: The kwargs passed to the model [history,...]

        :return: The tokens passed to the model
        """

        history = kwargs["history"]

        collect_history = ""
        for h in history:
            collect_history += h[0] + h[1]
        prompt_text = prompt + collect_history
        prompt_tokens = processor.tokenize(prompt_text)

        return prompt_tokens

    def download_image(self, image_url: str, save_path: str) -> str:
        """
        Download an image from a URL and save it locally.

        :param image_url: URL of the image to be downloaded.
        :param save_path: Local path where the image will be saved.
        :return: Path to the saved image.
        """
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image.save(save_path)
            return save_path
        else:
            raise ValueError("Failed to download image")

    def preprocess_image(self, image_list):
        """
        InternVLM does not support RGBA images
        Only supports image, when a local path is given [Images needs to be downloaded temporarily for matchit]

        :param image_list: A list of images to be preprocessed.
        :return: The list containing paths to the preprocessed images
        """
        temp_dir = os.path.join(os.getcwd(), IMG_CACHE)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.mkdir(temp_dir)

        processed_image_paths = []
        for i, image in enumerate(image_list):
            save_path = os.path.join(temp_dir, f'{i}.jpg')

            if image.startswith('http'):
                response = requests.get(image)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image)

            padded_img = self.custom_padding(image)

            padded_img.save(save_path)
            processed_image_paths.append(save_path)

        return processed_image_paths


    def generate_output(self, prompt: str, image: list, model: AutoModel,
                        processor: AutoTokenizer, **kwargs) -> [Dict, str]:
        """
        Generate Outputs [response, response_text] for InternLM type Models
        Ref - https://huggingface.co/internlm/internlm-xcomposer2d5-7b

        :param prompt: The text prompt to be used for generating the response.
        :param image: A list of images to be included in the model's input.
        :param model: The model used for generating the output. This should be compatible with InternLM type models.
        :param processor: The processor/tokenizer used to preprocess the prompt.
        :param **kwargs: Additional keyword arguments that may be required by the model or tokenizer.

        :return response: The raw output from the model.
        :return response_text: The decoded text response generated by the model.
        """

        # TODO - Raise Warning When Using CPU and not CUDA
        # TODO - Add model.chat args in model registry, pass them as additional kwargs

        image = self.preprocess_image(image)

        history = kwargs["history"]

        # By default unset Gradient Calculation for inferencing
        torch.set_grad_enabled(False)

        model = model.cuda().eval()
        model.tokenizer = processor

        # Use CUDA to get the response
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            gen_text, _ = model.chat(processor, prompt, image,
                                       do_sample=False,
                                       num_beams=3,
                                       top_p=1,
                                       history=history,
                                       use_meta=True)

            # Unset top_p manually to avoid the following warning
            # UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.

            response_text = gen_text.strip()

            # Cast into Clemgame compatible form
            response = {"response": gen_text}

        # Delete the image cache
        shutil.rmtree(IMG_CACHE)

        return response, response_text