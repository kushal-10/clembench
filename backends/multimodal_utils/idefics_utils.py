"""
Additional Utility functions for Idefics type models
"""

def generate_idefics_input(messages: list[Dict]):
    """
    Return inputs specific to the format of Idefics

    param messages: A list[Dict] type object passed to the backend containing 'role', 'content' and 'image'
    """
    # Create a list containing the prompt text and images specific to Idefics input
    # Refer - https://huggingface.co/HuggingFaceM4/idefics-80b-instruct

    # Use idefics_input as is for input to the model
    # Use idefics_text, that contains everything from idefics_input, apart from image_urls/loaded_image, used for context check
    idefics_input = []
    idefics_text = ""
    for m in messages:
        if m['role'] == 'user':
            idefics_input.append('\nUser: ' + m['content'])
            idefics_text += 'User: ' + m['content']
            if 'image' in m.keys():
                if type(m['image']) == list:  # Check if multiple images are passed, append accordingly
                    for im in m['image']:
                        loaded_im = load_image(im)
                        idefics_input.append(loaded_im)
                else:
                    idefics_input.append(m['image'])
            idefics_input.append('<end_of_utterance>')
            idefics_text += '<end_of_utterance>'
        elif m['role'] == 'assistant':
            idefics_input.append('\nAssistant: ' + m['content'])
            idefics_input.append('<end_of_utterance>')
            idefics_text += '\nAssistant: ' + m['content']
            idefics_text += '<end_of_utterance>'
    idefics_input.append('\nAssistant:')
    idefics_input = [idefics_input]

    return idefics_input, idefics_text


def generate_idefics_output(messages: list[Dict],
                            model: IdeficsForVisionText2Text,
                            processor: AutoProcessor,
                            max_tokens: int,
                            device) -> list[str]:
    """
    Return generated text from Idefics model

    param messages: A list[Dict] type object passed to the backend containing 'role', 'content' and 'image'
    param model: Idefics model
    param processor: Idefics processor
    param device: Processing device - cuda/CPU
    """
    idefics_input, _ = generate_idefics_input(messages=messages)
    inputs = processor(idefics_input, add_end_of_utterance_token=False, return_tensors="pt").to(device)

    # Generation args for Idefics
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids,
                                   max_new_tokens=max_tokens)
    generated_text = processor.batch_decode(generated_ids)

    return generated_text

