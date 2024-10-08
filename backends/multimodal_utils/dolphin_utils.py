import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import requests
from io import BytesIO
from typing import Dict, List, Any, Union, Tuple

from backends.multimodal_utils.base_utils import BaseMLLM

# disable some warnings according to the Base HF script - https://huggingface.co/cognitivecomputations/dolphin-vision-72b
transformers.logging.set_verbosity_error()
warnings.filterwarnings('ignore')


class DolphinMLLM(BaseMLLM):

    @staticmethod
    def prepare_inputs(messages: List[Dict[str, Any]], **kwargs) -> Dict:
        """
        Prepare the inputs for the model, including the prompt, and images.

        :param messages: A list of dictionaries, where each dictionary contains:
                         - 'role': The role of the message sender ('user', 'assistant' or 'system').
                         - 'content': The text content of the message.
                         - 'image': Optional; a single image URL (str) or a list of image URLs (List[str]).
        :param kwargs: Additional keyword arguments that may be used in the process.
        :return: A dictionary containing:
                 - 'prompt': The final prompt to be used by the model.
                 - 'images': A list of image URLs to be processed.
                 - 'processor_kwargs': A dictionary with 'history' (list of user-assistant message pairs). Passed to
                                       generate_outputs and get_tokens
        """
        input_messages = []
        images = []

        for message in messages:
            if message['role'] == 'user':
                str_builder = ""
                if 'image' in message:
                    if isinstance(message['image'], str):
                        # Single image
                        images.append(message['image'])
                        str_builder = "<image>"
                    elif isinstance(message['image'], list):
                        # List of images
                        for img in message['image']:
                            images.append(img)
                            str_builder += "<image>"
                    else:
                        raise ValueError("Invalid image type in message - should be str or List[str]")

                usr_msg = {'role': 'user', 'content': f'{str_builder}\n{message["content"]}'}
                input_messages.append(usr_msg)

            elif message['role'] == 'assistant':
                input_messages.append(message)

            elif message['role'] == 'system':
                input_messages.append(message)

        return {
            "prompt": input_messages,
            "images": images,
            "output_kwargs": {"device": kwargs.get('device')}
        }

    @staticmethod
    def get_tokens(prompt: list, handler: AutoTokenizer, **output_kwargs) -> List[str]:
        """
        Generate tokens for the given prompt and conversation history.

        :param prompt: The current prompt to be tokenized.
        :param handler: The tokenizer/processor used for tokenizing the prompt and history.
        :param kwargs: Additional keyword arguments, expecting 'history' which is a list of tuples (user message, assistant response).

        :return: A list of tokens generated from the combined prompt and conversation history.
        """

        tokens = handler.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True
        )

        return tokens

    @staticmethod
    def generate_outputs(prompt: list, images: List[str], model: AutoModelForCausalLM,
                         handler: AutoTokenizer, **output_kwargs) -> Tuple[Dict[str, Any], str]:
        """
        Generate model outputs given a prompt, images, and additional parameters.

        :param prompt: The list of prompt to be used for generating the response.
        :param images: A list of image URLs or paths to be included in the model's input.
        :param model: The model used for generating the output. This should be compatible with Dolphin 72B.
        :param handler: The tokenizer used to preprocess the prompt and handle the input images.
        :param kwargs: Additional keyword arguments for the model
        :return:
             - response (Dict[str, Any]): The raw output from the model, formatted as a dictionary.
             - response_text (str): The decoded text response generated by the model.
        """

        device = output_kwargs.get("device")
        torch.set_default_device(device)
        max_tokens = output_kwargs.get("max_tokens")
        max_tokens = 2048 # Override max tokens to the default set in Dolphin Model

        text = handler.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )

        text_chunks = [handler(chunk).input_ids for chunk in text.split('<image>')]
        input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[-1], dtype=torch.long).unsqueeze(0)

        # image, sample images can be found in images folder
        processed_images = []
        for img in images:
            if img.startswith("http"):
                response = requests.get(img)
                image = Image.open(BytesIO(response.content))
                processed_images.append(image)
            else:
                image = Image.open(img)
                processed_images.append(image)

        image_tensor = model.process_images(processed_images, model.config).to(dtype=model.dtype)

        # generate
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=max_tokens,
            use_cache=True)[0]

        response = handler.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True)

        response_text = response.strip()

        response = {"response": response}

        return response, response_text
