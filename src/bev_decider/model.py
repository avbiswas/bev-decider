"""The bev-decider network: a truncated Qwen3 backbone (with a LoRA adapter) and a decision head over the options."""

import math

import torch
import torch.nn as nn
from transformers import Qwen3Model


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, new_dim):
        super().__init__()
        self.new_dim = new_dim
        self.q = nn.Linear(hidden_dim, new_dim)
        self.k = nn.Linear(hidden_dim, new_dim)
        self.v = nn.Linear(hidden_dim, new_dim)
        self.o = nn.Linear(new_dim, hidden_dim)

    def forward(self, x, padding_mask):
        # x: [B, T, H], padding_mask: [B, T] (True = real slot); padded option slots are never attended to
        scores = torch.einsum("bqd,bkd->bqk", self.q(x), self.k(x)) / math.sqrt(self.new_dim)
        scores = scores.masked_fill(~padding_mask[:, None, :], -float("inf"))
        return self.o(torch.einsum("bqk,bkd->bqd", torch.softmax(scores, dim=-1), self.v(x)))


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, new_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, new_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, new_dim), nn.GELU(), nn.Linear(new_dim, hidden_dim))

    def forward(self, x, padding_mask):
        x = x + self.attn(self.norm1(x), padding_mask)
        return x + self.mlp(self.norm2(x))


class ChoiceHead(nn.Module):
    """Self-attention over [task token, option embeddings..., answer embedding], then a bilinear score per option."""

    def __init__(self, hidden_dim, new_dim, num_layers, num_task_types):
        super().__init__()
        self.new_dim = new_dim
        self.task_embedding = nn.Embedding(num_task_types, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([AttentionBlock(hidden_dim, new_dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.answer_proj = nn.Linear(hidden_dim, new_dim)
        self.choice_proj = nn.Linear(hidden_dim, new_dim)

    def forward(self, choice_embeddings, answer_embedding, choice_mask, task_type):
        B = choice_embeddings.shape[0]
        task_token = self.task_embedding(task_type).unsqueeze(1)
        x = self.input_norm(torch.cat([task_token, choice_embeddings, answer_embedding.unsqueeze(1)], dim=1))

        always_valid = torch.ones(B, 1, dtype=torch.bool, device=choice_mask.device)
        padding_mask = torch.cat([always_valid, choice_mask, always_valid], dim=1)
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = self.final_norm(x)

        answer = self.answer_proj(x[:, -1])     # [B, D]
        choices = self.choice_proj(x[:, 1:-1])  # [B, C, D]
        logits = torch.einsum("bd,bcd->bc", answer, choices) / math.sqrt(self.new_dim)
        return logits.masked_fill(~choice_mask, -float("inf"))


class DeciderNetwork(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids, position_ids, attention_mask, choice_read_idx, choice_mask, answer_end_idx, task_type):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                               use_cache=False).last_hidden_state.float()
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        choice_embeddings = hidden[batch_idx[:, None], choice_read_idx]  # [B, C, H]
        answer_embedding = hidden[batch_idx, answer_end_idx]             # [B, H]
        # The head always runs in fp32, as in training
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.head(choice_embeddings, answer_embedding, choice_mask, task_type)


def load_backbone(model_name, num_layers, revision=None):
    # Only the first num_layers decoder layers are built and loaded; the rest are never executed
    return Qwen3Model.from_pretrained(model_name, revision=revision, dtype=torch.float32, num_hidden_layers=num_layers)
