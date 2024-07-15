"""
Backend for open-weight multimodal models.

"""
from typing import List, Dict, Tuple, Any
import torch
from transformers import (AutoProcessor, AutoModelForVision2Seq, IdeficsForVisionText2Text,
                          AutoConfig, AutoModel, AutoTokenizer)

import backends
from backends.multimodal_utils.llava_utils import generate_llava_inputs, get_llava_response
from backends.multimodal_utils.intern_utils import get_intern_inputs, generate_intern_response

# CONSTANTS
MODEL_LOADER_MAP = {
    "Idefics": IdeficsForVisionText2Text,
    "Vision2Seq": AutoModelForVision2Seq,
    "Intern": AutoModel
}

FALLBACK_CONTEXT_SIZE = 256
logger = backends.get_logger(__name__)


# CONTEXT UTILS
def get_context_limit(model_spec: backends.ModelSpec) -> int:
    """
    Get the context limit of the model.

    :param model_spec: Contains definitions about the model to be used.
    :return: Context limit of the model.
    """
    hf_model_str = model_spec['huggingface_id']
    trust_remote_code = getattr(model_spec, 'trust_remote_code', False)

    model_config = AutoConfig.from_pretrained(hf_model_str, trust_remote_code=trust_remote_code)

    # Some models have 'max_position_embeddings', others have 'max_sequence_length'
    context = getattr(
        getattr(model_config, 'text_config', model_config),
        'max_position_embeddings',
        getattr(model_config, 'max_sequence_length', FALLBACK_CONTEXT_SIZE)
    )

    logger.info(f"Context limit for model - {hf_model_str} is {context}")
    return context


# CONTEXT UTILS
def check_context_limit(context_size: int, prompt_tokens: list, max_new_tokens: int = 100) -> Tuple[bool, int, int, int]:
    """
    External context limit check
    :param context_size: max_sequence_length/max_position_embeddings of the model
    :param prompt_tokens: List of prompt token IDs.
    :param max_new_tokens: How many tokens to generate ('at most', but no stop sequence is defined).
    :return: Tuple with
            Bool: True if context limit is not exceeded, False if too many tokens
            Number of tokens for the given messages and maximum new tokens
            Number of tokens of 'context space left'
            Total context token limit
    """
    prompt_size = len(prompt_tokens)
    tokens_used = prompt_size + max_new_tokens  # context includes tokens to be generated
    tokens_left = context_size - tokens_used
    fits = tokens_used <= context_size
    return fits, tokens_used, tokens_left, context_size


# MODEL UTILS
def load_processor(model_spec: backends.ModelSpec):
    """
    Load processor from AutoProcessor for a specific model (Example - LlavaProcessor).

    :param model_spec: A dictionary that defines the model to be used, loaded from Model Registry.
    :return processor: Processor for the specific model.
    """
    hf_model_str = model_spec['huggingface_id']  # Get the model name

    use_fast = not getattr(model_spec, 'not_fast', False)
    use_tokenizer = getattr(model_spec, 'tokenizer', False)
    trust_remote_code = getattr(model_spec, 'trust_remote_code', False)
    processor_class = AutoTokenizer if use_tokenizer else AutoProcessor

    processor = processor_class.from_pretrained(
        hf_model_str,
        use_fast=use_fast,
        device_map="auto",
        verbose=False,
        trust_remote_code=trust_remote_code
    )

    logger.info(f'Loading {processor_class} for model : {model_spec.model_name}')
    return processor


# MODEL UTILS
def load_model(model_spec: backends.ModelSpec):
    """
    Load a specific model.

    :param model_spec: A dictionary that defines the model to be used, loaded from Model Registry.
    :return model: The specific model.
    """
    logger.info(f'Start loading huggingface model weights: {model_spec.model_name}')
    hf_model_str = model_spec['huggingface_id']  # Get the model name

    model_type = MODEL_LOADER_MAP[model_spec['model_type']]  # Use the appropriate Auto class to load the model

    trust_remote_code = getattr(model_spec, 'trust_remote_code', False)
    use_bf16 = getattr(model_spec, 'use_bf16', False)
    print("Model Values")
    print(trust_remote_code, use_bf16)
    model = model_type.from_pretrained(
        hf_model_str,
        torch_dtype=torch.bfloat16 if use_bf16 else "auto",
        trust_remote_code=trust_remote_code,
        device_map="auto" if not trust_remote_code else None
    )

    # Set pad_token_id to eos_token_id if it's not already set
    generation_config = model.generation_config
    if getattr(generation_config, 'pad_token_id', None) is None:
        generation_config.pad_token_id = generation_config.eos_token_id

    logger.info(f"Finished loading huggingface model: {model_spec.model_name}")

    # Log the device map if it's available
    device_map = getattr(model, 'hf_device_map', None)
    if device_map:
        logger.info(f"Device Map: {device_map}")

    return model


# BACKEND UTILS
def check_multiple_image(messages: List[Dict]) -> bool:
    """
    Return True if a single message contains multiple images.

    :param messages: A list[Dict] type object passed to the backend containing 'role', 'content', and 'image'.
    :return: True if any message contains multiple images, otherwise False.
    """
    return any('image' in msg and isinstance(msg['image'], list) and len(msg['image']) > 1 for msg in messages)


class HuggingfaceMultimodal(backends.Backend):
    def __init__(self):
        super().__init__()

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        return HuggingfaceMultimodalModel(model_spec)


class HuggingfaceMultimodalModel(backends.Model):

    def __init__(self, model_spec: backends.ModelSpec):
        super().__init__(model_spec)

        # Load instance variable used for evey model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_spec['model_type']
        self.model_name = model_spec['model_name']
        self.processor = load_processor(model_spec)
        # self.multimodal_model = load_model(model_spec)
        self.multimodal_model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
        self.multimodal_model.tokenizer = self.processor  # Hardcode for internLM
        self.split_prefix = model_spec['output_split_prefix']
        self.context_size = get_context_limit(model_spec)

        # Type cast model_spec to a Dictionary, for cleaner loading of variables
        model_spec_dict = vars(model_spec)
        # Load model specific instance variables
        self.template = model_spec_dict.get('custom_chat_template', None)
        self.cull = model_spec_dict.get('eos_to_cull', None)
        self.supports_multiple_images = model_spec_dict.get('supports_multiple_images', False)

        # self.idefics = 'idefics' in model_spec['model_name']
        # self.intern = 'intern' in model_spec['model_name']

    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "user", "content": "Are there any clouds in the image? Answer with only "Yes" or "No"."},
                    {"role": "assistant", "content": "Yes"},
                    {"role": "user", "content": "This seems correct."},
                    {'role': 'user', 'content': 'Are there any chickens in the image? Answer with only "Yes" or "No".', 'image': 'games/cloudgame/resources/images/3.jpg'}
                ]
        :return: the continuation
        """

        # Check to see if game passes multiple images in a single turn
        # Proceed only if model supports multiple images, else return blanks for prompt, response and response_text
        has_multiple_images = check_multiple_image(messages=messages)
        if has_multiple_images and not self.supports_multiple_images:
            print(f"Multiple images not supported in a single turn for model {self.model_name}")
            return "", {"response": ""}, ""

        # Get input prompt by applying jinja template, if template is provided
        # prompt_text = ## Get Input String for counting tokens?
        if 'intern' in self.model_name:
            prompt, history, images = get_intern_inputs(messages)
            collect_history = ""
            for h in history:
                collect_history += h[0] + h[1]
            prompt_text = prompt + collect_history
            prompt_tokens = self.processor.tokenize(prompt_text)
        else:
            prompt_text, images = generate_llava_inputs(messages, self.template)
            prompt_tokens = self.processor.tokenizer.tokenize(prompt_text)

        # Check context limit
        context_check = check_context_limit(self.context_size, prompt_tokens, max_new_tokens=self.get_max_tokens())
        if not context_check[0]:  # if context is exceeded, context_check[0] is False
            logger.info(f"Context token limit for {self.model_spec.model_name} exceeded: "
                        f"{context_check[1]}/{context_check[3]}")
            # fail gracefully:
            raise backends.ContextExceededError(f"Context token limit for {self.model_spec.model_name} exceeded",
                                                tokens_used=context_check[1], tokens_left=context_check[2],
                                                context_size=context_check[3])

        prompt = {"inputs": prompt_text, "max_new_tokens": self.get_max_tokens(), "temperature": self.get_temperature()}

        # Based on this input_prompt, return response, response_text for each model
        # Store generated text
        if 'intern' in self.model_name:
            response, response_text = generate_intern_response(prompt=prompt, history=history, image=images,
                                                               model=self.multimodal_model, tokenizer=self.processor)
        else:
            response, response_text = get_llava_response(prompt_text, images, self.processor, self.multimodal_model,
                                                         self.get_max_tokens(), self.device, self.split_prefix, self.cull)

        print(f"################################################ TESTING RESPONSE #############################################################")
        print(f"INPUT: {prompt} \n RESPONSE: {response}\n RESPONSE TEXT: {response_text}")
        return prompt, response, response_text
