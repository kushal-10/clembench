import requests
import torch
from PIL import Image
from io import BytesIO

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda:0"

# Note that passing the image urls (instead of the actual pil images) to the processor is also possible
# image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
# image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
# image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

# Add a model resize function - Reshape all images to the same dimension
image_links = ["https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
               "https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg",
               "https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg"]

def resize_images(image_list: list) -> list:
    """
    Get the min_width and min_height from the image list
    Resize every image to min_width, min_height.
    :param image_list: A list of image paths/urls
    :return: A list of resized images
    """
    images = []
    widths = []
    heights = []
    for image_link in image_list:
        if image_link.startswith("http"):
            image = Image.open(BytesIO(requests.get(image_link).content))
        else:
            image = Image.open(image_link)
        images.append(image)
        widths.append(image.width)
        heights.append(image.height)

    max_width = max(widths)
    max_height = max(heights)

    resized_images = []
    for image in images:
        image = image.resize((max_width, max_height))
        resized_images.append(image)

    return resized_images

images = resize_images(image_links)
image1 = images[0]
image2 = images[1]
image3 = images[2]

processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b-chatty")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
).to(DEVICE)

# Create inputs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What do we see in this image?"},
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty."},
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "And how about this image?"},
        ]
    },
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image1, image2], return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}


# Generate
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)
# ['User: What do we see in this image? \nAssistant: In this image, we can see the city of New York, and more specifically the Statue of Liberty. \nUser: And how about this image? \nAssistant: In this image we can see buildings, trees, lights, water and sky.']
