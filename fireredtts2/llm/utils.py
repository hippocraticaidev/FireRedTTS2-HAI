import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import Union
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
from fireredtts2.llm.llm import Model, ModelArgs


@dataclass
class Segment:
    speaker: str
    text: str
    audio: torch.Tensor


class WarmupDecayLR(LambdaLR):
    """
    Learning rate scheduler with a linear warmup and specificable decay.
    """

    def __init__(
        self, optimizer, warmup_steps: int, total_steps: int, decay_type: str = "linear"
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_type = decay_type
        super().__init__(optimizer, self.lr_lambda, last_epoch=-1)

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / self.warmup_steps
        else:
            if self.decay_type == "linear":
                return (self.total_steps - step) / (
                    self.total_steps - self.warmup_steps
                )
            elif self.decay_type == "constant":
                return 1.0
            elif self.decay_type == "exponential":
                return 0.1 ** (
                    (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                )
            elif self.decay_type == "cosine":
                return 0.5 * (
                    1
                    + torch.cos(
                        torch.pi
                        * torch.tensor(
                            (step - self.warmup_steps)
                            / (self.total_steps - self.warmup_steps)
                        )
                    )
                )
            else:
                raise ValueError(f"Invalid decay type: {self.decay_type}")


additional_special_tokens = [
    "<|text_start|>",
    "<|text_end|>",
    "[S1]",
    "[S2]",
    "[S3]",
    "[S4]",
    "[S5]",
    "[S6]",
    "[S7]",
    "[S8]",
    "[S9]",
    "[S10]",
    "[S11]",
    "[S12]",
    "[S13]",
    "[S14]",
    "[S15]",
    "[S16]",
    "[S17]",
    "[S18]",
    "[S19]",
    "[S20]",
    "[S21]",
    "[S22]",
    "[S23]",
    "[S24]",
    "[S25]",
    "[S26]",
    "[S27]",
    "[S28]",
    "[S29]",
    "[S30]",
    "[S31]",
    "[S32]",
    "[S33]",
    "[S34]",
    "[S35]",
    "[S36]",
    "[S37]",
    "[S38]",
    "[S39]",
    "[S40]",
    "[S_PODCAST_1]",
    "[S_PODCAST_2]",
    "[S_PODCAST_3]",
    "[S_PODCAST_4]",
    "[S_PODCAST_5]",
    "[S_PODCAST_6]",
    "[S_PODCAST_7]",
    "[S_PODCAST_8]",
    "[S_PODCAST_9]",
    "[S_PODCAST_10]",
    "[S_DIALOG_1]",
    "[S_DIALOG_2]",
    "[S_DIALOG_3]",
    "[S_DIALOG_4]",
    "[S_DIALOG_5]",
    "[S_DIALOG_6]",
    "[S_DIALOG_7]",
    "[S_DIALOG_8]",
    "[S_DIALOG_9]",
    "[S_DIALOG_10]",
    "<|emotion_neutral|>",
    "<|emotion_happy|>",
    "<|emotion_sad|>",
    "<|emotion_concern|>",
    "<|emotion_confuse|>",
    "<|emotion_angry|>",
    "<|emotion_surprise|>",
    "<|emotion_disgust|>",
    "<|emotion_nervous|>",
    "<|emotion_apology|>",
    "<|emotion_understand|>",
    "<|emotion_fear|>",
    "<|emotion_comfort|>",
    "<|emotion_shy|>",
    "<|emotion_serious|>",
    "<|emotion_extra1|>",
    "<|emotion_extra2|>",
    "<|emotion_extra3|>",
    "<|emotion_extra4|>",
    "<|emotion_extra5|>",
    "<|emotion_extra6|>",
    "<|emotion_extra7|>",
    "<|emotion_extra8|>",
    "<|emotion_extra9|>",
    "<|emotion_extra10|>",
    "<|breath|>",
    "<|humph|>",
    "<|laugh_heng|>",
    "<|hissing|>",
    "<|sniff|>",
    "<|laugh_he|>",
    "<|sigh|>",
    "<|laugh|>",
    "<|laugh_ha|>",
    "<|quick_breath|>",
    "<|laugh_hei|>",
    "<|laugh_speak|>",
    "<|/laugh_speak|>",
    "<|cry|>",
    "<|choking|>",
    "<|cry_speak|>",
    "<|/cry_speak|>",
    "<|slurp|>",
    "<|clucking|>",
    "<|yawning|>",
    "<|cough|>",
    "<|smack|>",
    "<|hem|>",
    "<|stretch|>",
    "<|sneeze|>",
    "<|paralinguistic_extra1|>",
    "<|paralinguistic_extra2|>",
    "<|paralinguistic_extra3|>",
    "<|paralinguistic_extra4|>",
    "<|paralinguistic_extra5|>",
    "<|paralinguistic_extra6|>",
    "<|paralinguistic_extra7|>",
    "<|paralinguistic_extra8|>",
    "<|paralinguistic_extra10|>",
    "<|paralinguistic_extra11|>",
    "<|paralinguistic_extra12|>",
    "<|paralinguistic_extra13|>",
]

ARPABET_PHONE_SUPERSET = [
    # -------------------------
    # CMU consonants (24)
    # -------------------------
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L",
    "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V",
    "W", "Y", "Z", "ZH",

    # -------------------------
    # CMU bare vowels (15)
    # -------------------------
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY",
    "IH", "IY",
    "OW", "OY",
    "UH", "UW",

    # -------------------------
    # CMU stressed vowels (15 * 3 = 45)
    # -------------------------
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2",
    "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2",
    "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2",
    "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "OW0", "OW1", "OW2",
    "OY0", "OY1", "OY2", "UH0", "UH1", "UH2", "UW0", "UW1", "UW2",

    # -------------------------
    # Reduced vowels / variants (bare + stressed)
    # -------------------------
    "AX", "AX0", "AX1", "AX2",
    "AXR", "AXR0", "AXR1", "AXR2",
    "IX", "IX0", "IX1", "IX2",
    "UX", "UX0", "UX1", "UX2",

    # -------------------------
    # Syllabics + allophones (common in TIMIT/aligners)
    # -------------------------
    "EL", "EM", "EN",
    "DX", "NX", "Q",

    # -------------------------
    # TIMIT-specific / common alternates
    # -------------------------
    "HV",      # TIMIT "hv" (voiceless HH)
    "AX-H",    # TIMIT devoiced schwa
    "ENG",     # TIMIT "eng" (often maps to NG)

    # -------------------------
    # TIMIT closures
    # -------------------------
    "BCL", "DCL", "GCL", "KCL", "PCL", "TCL",

    # -------------------------
    # Silence / noise markers (varies by tool; include both cases)
    # -------------------------
    "H#", "SIL", "SP", "SPN", "PAU", "EPI", "BRTH",
    "NSN", "LAU", "VOCNOISE", "NOISE",
    "SPOKEN_NOISE", "MUSIC", "BREATH",
]
ARPABET_EXTENDED_SPECIAL_TOKENS = ["<|arpa_start|>"] + [f"<|arpa_{phone}|>" for phone in ARPABET_PHONE_SUPERSET] + ["<|arpa_end|>"]
additional_special_tokens += ARPABET_EXTENDED_SPECIAL_TOKENS


def load_custom_tokenizer(qwen2_tokenizer_path: str):
    tok = AutoTokenizer.from_pretrained(qwen2_tokenizer_path)
    special_tokens_dict = {
        "additional_special_tokens": additional_special_tokens,
    }
    tok.add_special_tokens(special_tokens_dict)
    return tok


def init_weights(model: nn.Module):
    """
    Initialize the weights of the model.
    - Xavier uniform initialization for linear layers
    - Normal initialization for embeddings
    - Xavier uniform initialization for parameters
    """

    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Parameter):
            nn.init.xavier_uniform_(m.data)

    model.apply(_init_weights)

    # Special handling for audio_head because it's nn.Parameter directly
    nn.init.xavier_uniform_(model.audio_head)

    return model


def load_model(
    configs,
    checkpoint_path: Union[str, Path] = None,
    device: Union[str, torch.device] = "cuda",
) -> Model:
    """Load model, add forward method, and move to device.

    Args:
        model_name_or_checkpoint_path: Name or path of pretrained model or checkpoint.
        device: Device to move the model to.
        decoder_loss_weight: Decoder loss weight.
    """

    model_arg = ModelArgs(
        backbone_flavor=configs["models"]["backbone_flavor"],
        decoder_flavor=configs["models"]["decoder_flavor"],
        text_vocab_size=configs["models"]["text_vocab_size"],
        audio_vocab_size=configs["models"]["audio_vocab_size"],
        audio_num_codebooks=configs["models"]["audio_num_codebooks"],
        decoder_loss_weight=configs["models"]["decoder_loss_weight"],
        use_text_loss=True,
    )
    model = Model(model_arg)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )["model"]
        model.load_state_dict(state_dict)
    else:
        model = init_weights(model)

    model = model.to(device=device)
    return model


def load_llm_model(
    configs,
    checkpoint_path: Union[str, Path] = None,
    device: Union[str, torch.device] = "cuda",
) -> Model:
    """Load model, add forward method, and move to device.

    Args:
        model_name_or_checkpoint_path: Name or path of pretrained model or checkpoint.
        device: Device to move the model to.
        decoder_loss_weight: Decoder loss weight.
    """

    model_arg = ModelArgs(
        backbone_flavor=configs["llm_models"]["backbone_flavor"],
        decoder_flavor=configs["llm_models"]["decoder_flavor"],
        text_vocab_size=configs["llm_models"]["text_vocab_size"],
        audio_vocab_size=configs["llm_models"]["audio_vocab_size"],
        audio_num_codebooks=configs["llm_models"]["audio_num_codebooks"],
        decoder_loss_weight=configs["llm_models"]["decoder_loss_weight"],
        use_text_loss=True,
    )
    model = Model(model_arg)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )["model"]
        model.load_state_dict(state_dict)
    else:
        model = init_weights(model)

    model = model.to(device=device)
    return model


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def get_grad_norm(model):
    total_norm = 0
    num = 0
    for name, p in model.named_parameters():
        try:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            num += 1
        except:
            print(name)
    total_norm = total_norm ** (1.0 / 2)
    total_norm = total_norm / num
    return total_norm


def read_jsonl(path):
    path = os.path.expanduser(path)
    with open(path, "r") as f:
        json_str = f.read()
    data_list = []
    for line in json_str.splitlines():
        data = json.loads(line)
        data_list.append(data)
    return data_list


def convert_snow_arpa_to_special_tokens(arpa_text: str) -> str:
    """
    Convert Snow ARPA text to special tokens.
    Snow ARPA is space separated list of arpabet phones. eg. "HH AH0 L OW1".
    Its converted to special tokens by wrapping each phone in <|arpa_> tags
    and wrapping the entire text in <|arpa_start|> and <|arpa_end|> tags.
    eg. "HH AH0 L OW1" -> "<|arpa_start|><|arpa_HH|><|arpa_AH0|><|arpa_L|><|arpa_OW1|><|arpa_end|>"
    Args:
        arpa_text: ARPA text to convert.
    Returns:
        snow arpa text represented as special tokens.
    """
    tokens = arpa_text.strip().split(" ")
    tokens = ["<|arpa_start|>"] + [f"<|arpa_{phone}|>" for phone in tokens] + ["<|arpa_end|>"]
    return "".join(tokens)
