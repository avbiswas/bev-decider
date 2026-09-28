"""Loading bev-decider from the Hugging Face Hub and answering typed questions."""

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoTokenizer

from .encode import Encoder
from .model import ChoiceHead, DeciderNetwork, load_backbone

DEFAULT_MODEL = "avbiswas/bev-decider-0.4B"


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Decider:
    def __init__(self, network, encoder, config, device, name):
        self.network, self.encoder, self.config, self.device, self.name = network, encoder, config, device, name

    @torch.no_grad()
    def probabilities(self, examples):
        batch = {k: v.to(self.device) for k, v in self.encoder.collate(examples).items()}
        # bf16 on GPU / Apple Silicon (as in training); fp32 on CPU
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type in ("cuda", "mps")):
            logits = self.network(**batch)
        return torch.softmax(logits.float(), dim=-1).cpu()

    def decide(self, state, questions, batch_size=16):
        """Answers Jev-format questions about one state.

        state: a string, or any JSON value (rendered as indented JSON).
        questions: {id: {"type": "choice" | "noul" | "score", "instructions": ..., "criteria": ...}}
        Returns {id: answer}, where answer is
          choice: {"type": "choice", "choice": <best key>, "probabilities": {key: p}}
          noul:   {"type": "noul", "noul": P(yes)}
          score:  {"type": "score", "score": <expected level>, "probabilities": {"0": p, ...}}
        """
        ids = list(questions)
        examples = [self.encoder.encode(state, questions[qid]) for qid in ids]
        answers = {}
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start + batch_size]
            probs = self.probabilities(chunk)
            for qid, ex, p in zip(ids[start:start + batch_size], chunk, probs):
                p = p[:len(ex["keys"])].tolist()
                kind = questions[qid]["type"]
                if kind == "noul":
                    answers[qid] = {"type": "noul", "noul": p[1]}
                elif kind == "choice":
                    best = max(range(len(p)), key=p.__getitem__)
                    answers[qid] = {"type": "choice", "choice": ex["keys"][best], "probabilities": dict(zip(ex["keys"], p))}
                else:
                    answers[qid] = {"type": "score", "score": sum(i * x for i, x in enumerate(p)),
                                    "probabilities": dict(zip(ex["keys"], p))}
        return answers


def load(model=DEFAULT_MODEL, revision=None, device=None, max_state_tokens=None):
    """Loads bev-decider from a Hub repo id or a local folder (config.json, adapter, head.safetensors)."""
    path = Path(model) if Path(model).is_dir() else Path(snapshot_download(model, revision=revision))
    config = json.loads((path / "config.json").read_text())
    device = torch.device(device) if device else default_device()

    backbone = load_backbone(config["base_model"], config["num_layers"], config.get("base_revision"))
    backbone = PeftModel.from_pretrained(backbone, path)
    head = ChoiceHead(hidden_dim=backbone.config.hidden_size, **config["head"])
    head.load_state_dict(load_file(path / "head.safetensors"))
    network = DeciderNetwork(backbone, head).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config.get("base_revision"))
    encoder = Encoder(tokenizer, max_state_tokens or config["max_state_tokens"], config["max_choice_tokens"])
    return Decider(network, encoder, config, device, name=str(model))
