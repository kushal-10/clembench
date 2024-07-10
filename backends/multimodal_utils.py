## Individual inference methods for Multimodal Models
from typing import List, Dict, Tuple, Any
import torch
from transformers import AutoModel, AutoTokenizer


## InternLM X-Composer 2.5 7B
def get_intern_inputs(messages: list[Dict]):
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
                    print("Please pass a valid value of image in the message - Either a str or List[str]")

        elif m['role'] == 'assistant':
            # Append User+Assistant Message in Sequence
            history.append((prev_user_msg, m['content']))

    prompt = prev_user_msg

    return prompt, history, image


def generate_intern_response(messages: list[Dict], model, tokenizer):

    prompt, history, image = get_intern_inputs(messages)
    print(f"INPUT PRROOOOOOOOOOOMMMMMMMMPPPPPPPPPPPPPPTTTTTTTTTTTTTTT {prompt}")
    torch.set_grad_enabled(False)

    # init model and tokenizer
    # model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b', torch_dtype=torch.bfloat16,
    #                                   trust_remote_code=True).cuda().eval()
    # tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
    # model.tokenizer = tokenizer

    query = prompt
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        response, his = model.chat(tokenizer, query, image, do_sample=False, top_p=1, num_beams=3, history=history,
                                   use_meta=True)
        # Unset top_p manually to avoid the following warning
        #  UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
    print(response)
    print(f"RRRRRRREEEEEEEEEEEEEEEEEEEESSSSSSPPPPOOONNNSEEEEEEEEE: {response}")
    return response, prompt
