"""
Util functions for multimodal models.
"""

from typing import List, Dict, Tuple, Any
import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import requests
from io import BytesIO


##### INTERNVL2 + NVLM TYPE MODELS #####

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

 
def generate_history_internvl2(messages: List[str]) -> Tuple[List[Tuple], str]:
    """
    Separates the history and query from the list of messages in the current game instance.
    Compatible with InternVL2 and Nvidia NVLM models.

    Args:
        messages: A list containing user messages, system messages or assistant responses.
    
    Returns:
        A list of tuples containing the history and a user message string, passed to the model in the current game instance.

    Raises:
        ValueError: if msg['role'] is different than 'user', 'system', or 'assistant'.
    """

    history = []
    for msg in messages:
        if msg['role'] == 'system':
            continue # Skip the system message, Not passed to the model. Ref - https://huggingface.co/OpenGVLab/InternVL2-40B 
        elif msg['role'] == 'user':
            if 'image' in msg:
                user_message = f"</image>\n{msg['content']}" # Add <image> token if image is passed in this instance.
            else:
                user_message = msg['content']
        elif msg['role'] == 'assistant':
            history.append((user_message, msg['content']))
        else:
            raise ValueError(f"Invalid role: {msg['role']}. Expected 'user', 'system', or 'assistant'.")

    return history, user_message


def split_model(model_name):
    """
    Splits the model across available GPUs based on the model name.

    Args:
        model_name (str): The name of the model to be split. 
                          Expected values include 'InternVL2-1B', 'InternVL2-2B', 
                          'InternVL2-4B', 'InternVL2-8B', 'InternVL2-26B', 
                          'InternVL2-40B', 'InternVL2-Llama3-76B'.

    Returns:
        dict: A mapping of model layers to GPU indices.
    """
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def build_transform(input_size):
    """Builds a transformation pipeline for image preprocessing.

    Args:
        input_size (int): The size to which the image will be resized.

    Returns:
        torchvision.transforms.Compose: A composed transform for the image.
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Finds the closest aspect ratio from a set of target ratios.

    Args:
        aspect_ratio (float): The aspect ratio of the original image.
        target_ratios (list): A list of target aspect ratios.
        width (int): The width of the original image.
        height (int): The height of the original image.
        image_size (int): The size of the image for comparison.

    Returns:
        tuple: The best aspect ratio found.
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Processes the image to fit the closest aspect ratio and splits it into blocks.

    Args:
        image (PIL.Image): The image to be processed.
        min_num (int): Minimum number of blocks.
        max_num (int): Maximum number of blocks.
        image_size (int): The size of the image.
        use_thumbnail (bool): Whether to create a thumbnail.

    Returns:
        list: A list of processed image blocks.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    """Loads an image file and applies transformations.

    Args:
        image_file (str): The path to the image file.
        input_size (int): The size to which the image will be resized.
        max_num (int): Maximum number of blocks to create.

    Returns:
        torch.Tensor: A tensor containing the pixel values of the processed images.
    """
    if image_file.startswith("http"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_internvl2_image(messages: List[str], device: str):
    """
    Extracts the last user message containing image data and loads the corresponding images.

    Args:
        messages (List[str]): A list of message dictionaries containing user, system, and assistant messages.
        device (str): The device to which the image tensors will be moved (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: A tensor containing the pixel values of the processed images.

    Raises:
        ValueError: If no user message is found.
    """
    # Get last user message
    last_user_message = None
    for i in range(len(messages)):
        index = len(messages) - i - 1
        # Find last user message
        if messages[index]['role'] == 'user':
            last_user_message = messages[index]

    if last_user_message is None:
        raise ValueError("No user message found in the provided messages.")
    else:
        if len(last_user_message['image']) > 1:            
            pixel_values = load_image(last_user_message['image'][0], max_num=12).to(torch.bfloat16).to(device)
            for i in range(1, len(last_user_message['image'])):
                pixel_values1 = load_image(last_user_message['image'][i], max_num=12).to(torch.bfloat16).to(device)
                pixel_values = torch.cat((pixel_values, pixel_values1), dim=0)
        else:
            pixel_values = load_image(last_user_message['image'][0], max_num=12).to(torch.bfloat16).to(device)

    return pixel_values