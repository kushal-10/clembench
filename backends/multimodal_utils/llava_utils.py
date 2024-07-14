"""
Additional Utility functions for LLaVA type models
"""
from typing import Dict
from PIL import Image
import requests
from jinja2 import Template


def load_image(image: str):
    """
    Load an image based on a given local path or URL

    :param image: Image path/url
    :return loaded_image: PIL Image
    """

    if image.startswith('http') or image.startswith('https'):
        image = Image.open(requests.get(image, stream=True).raw).convert('RGB')
    else:
        image = Image.open(image).convert('RGB')

    return image


def get_images(messages: list[Dict]) -> list:
    """
    Return loaded images from messages.

    :param messages: A list of messages passed to the model.
    :return images: A list of PIL Image objects.
    """
    # Collect image links/file locations mentioned in messages
    images = [
        img
        for message in messages
        if 'image' in message
        for img in (message['image'] if isinstance(message['image'], list) else [message['image']])
    ]

    # Return None if no image is passed
    if not images:
        return []

    # Load Images
    loaded_images = [load_image(img) for img in images]

    return loaded_images


def generate_llava_inputs(messages, template):
    # Collect Loaded Images for the processor
    images = get_images(messages)

    # Generate Input prompt string for the model
    template_str = template
    template = Template(template_str)
    prompt_text = template.render(messages=messages)

    return prompt_text, images


def get_llava_response(prompt_text, images, processor, model, max_tokens, device, split_prefix, cull):
    """
    Prompt Llava type models to generate the response
    """

    # If Image is not passed, use the Language part of the model via tokenizer, skip the image processing part
    if not images:  # If no images are present in the history + current utterance, use tokenizer to get inputs
        inputs = processor.tokenizer(prompt_text, return_tensors="pt").to(device)
    else:
        inputs = processor(prompt_text, images=images, return_tensors="pt").to(device)

    model_output = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_text = processor.batch_decode(model_output, skip_special_tokens=True)

    response = {"response": generated_text}

    response_text = generated_text[0].split(split_prefix)[-1]  # Get the last assistant response
    if cull:
        rt_split = response_text.split(cull)  # Cull from End of String token
        response_text = rt_split[0]
    response_text = response_text.strip()

    return response, response_text
