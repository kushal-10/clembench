from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List

class BaseMLLM(ABC):
    """
    Abstract base class for Multimodal Large Language Models (MLLMs).

    This class defines the interface for preparing inputs and generating outputs
    for multimodal models. Any concrete implementation of this class must
    provide implementations for these abstract methods.

    Methods:
        prepare_inputs: Prepare and return the inputs required for the model.
        get_tokens: Return the token usage in the current turn for the model. Used to check context limit.
        generate_outputs: Generate and return outputs from the model given prepared inputs.
    """

    @abstractmethod
    def prepare_inputs(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Prepare and format the inputs required by the model.

        This method should be implemented to handle various formats of input data
        and return them in a format suitable for the model.

        :param args: Positional arguments required for preparing inputs.
        :param kwargs: Keyword arguments required for preparing inputs, which may include additional data.

        :return: A dictionary containing the prepared inputs for the model.
        """
        pass

    @abstractmethod
    def get_tokens(self, prompt: str, handler: Any, **kwargs: Any) -> List[str]:
        """
        Generate tokens for the given prompt and conversation history.

        :param prompt: The current prompt to be tokenized.
        :param handler: The processor/tokenizer used for tokenizing the prompt and history.
        :param kwargs: Additional keyword arguments, expecting 'history' which is a list of tuples (user message, assistant response).

        :return: A list of tokens generated from the combined prompt and conversation history.
        """
        pass

    @abstractmethod
    def generate_outputs(self, *args: Any, **kwargs: Any) -> Tuple[Dict[str, Any], str]:
        """
        Generate outputs from the model based on the prepared inputs.

        This method should be implemented to process the inputs with the model and
        produce outputs. The outputs should include both the raw response and a
        decoded text response.

        :param args: Positional arguments required for generating outputs.
        :param kwargs: Keyword arguments required for generating outputs, which may include model configuration or additional options.
        :return: A tuple containing:
            - response (Dict[str, Any]): The raw output from the model.
            - response_text (str): The decoded text response generated by the model.
        """
        pass