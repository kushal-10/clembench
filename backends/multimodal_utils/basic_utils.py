"""
Define some basic utility functions for the multimodal backend.
"""

from transformers import (AutoProcessor, AutoModelForVision2Seq, IdeficsForVisionText2Text,
                          AutoConfig, AutoModel, AutoTokenizer)


LOADER_MAP = {
    "Idefics": IdeficsForVisionText2Text,
    "Vision2Seq": AutoModelForVision2Seq,
    "Intern": AutoModel
}