import os
import platform
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.config import SLM_MODEL

_tokenizer = None
_model = None

def load_slm(force_cpu: bool = False):
    """
    Stable loader:
    - On macOS: default CPU (prevents random segfault/device_map issues)
    - Else: use CUDA if available
    """
    global _tokenizer, _model

    if _model is not None:
        return _tokenizer, _model

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    is_macos = platform.system().lower() == "darwin"
    use_cpu = force_cpu or is_macos or (not torch.cuda.is_available())

    _tokenizer = AutoTokenizer.from_pretrained(SLM_MODEL, use_fast=True)

    if use_cpu:
        _model = AutoModelForCausalLM.from_pretrained(
            SLM_MODEL,
            torch_dtype=torch.float32,
        ).to("cpu")
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            SLM_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    _model.eval()
    return _tokenizer, _model