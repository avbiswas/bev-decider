"""Turns a (state, question) pair into model inputs.

Every option gets the same starting position id and attends only to the prompt and itself, so the options are read in
parallel and their order cannot change the answer.
"""

import json

import torch

TASK_TYPES = {"choice": 0, "noul": 1, "score": 2}

# Short natural-language task hint, as used in training
TASK_PROMPTS = {
    "choice": "Choose the best option.",
    "score": "Rate it on the given scale.",
    "noul": "Answer yes or no.",
}
ANSWER_PROMPT = "The answer is:"


def as_text(x):
    """States, instructions and criteria may be strings or JSON values; JSON is rendered with indent=2."""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=2)


def question_options(question):
    """Question dict (Jev format) -> (option strings, keys the probabilities are reported under)."""
    kind, criteria = question["type"], question.get("criteria")
    if kind == "choice":
        if not criteria:
            raise ValueError("a choice question needs criteria: {option: description}")
        keys = list(criteria)
        options = [f"{key}: {as_text(criteria[key])}" if criteria[key] else key for key in keys]
    elif kind == "score":
        if not criteria:
            raise ValueError("a score question needs criteria: a list of level descriptions")
        keys = [str(i) for i in range(len(criteria))]
        options = [f"{i}: {as_text(level)}" for i, level in enumerate(criteria)]
    elif kind == "noul":
        keys = ["false", "true"]
        options = ["No", "Yes"]
        if criteria:  # optional {"true": ..., "false": ...} descriptions
            options = [f"No: {as_text(criteria['false'])}", f"Yes: {as_text(criteria['true'])}"]
    else:
        raise ValueError(f"unknown question type {kind!r}; expected choice, noul or score")
    return options, keys


class Encoder:
    def __init__(self, tokenizer, max_state_tokens, max_choice_tokens):
        self.tok = tokenizer
        self.max_state_tokens = max_state_tokens
        self.max_choice_tokens = max_choice_tokens
        self.option_begin = tokenizer.encode("<option>")
        self.option_end = tokenizer.encode("</option>")
        self.answer = tokenizer.encode(ANSWER_PROMPT)

    def truncate(self, text, max_tokens):
        tokens = self.tok.encode(text)
        return text if len(tokens) <= max_tokens else self.tok.decode(tokens[:max_tokens])

    def encode(self, state, question):
        options, keys = question_options(question)
        # The task hint and question come before the state, so every state token is read knowing what is asked
        prompt = (f"{TASK_PROMPTS[question['type']]}\nQuestion: {as_text(question['instructions'])}\n\n"
                  f"{self.truncate(as_text(state), self.max_state_tokens)}\n")
        prompt_tokens = self.tok.encode(prompt)
        option_tokens = [self.option_begin + self.tok.encode(self.truncate(o, self.max_choice_tokens)) + self.option_end
                         for o in options]

        input_ids = list(prompt_tokens)
        position_ids = list(range(len(prompt_tokens)))
        spans = []
        for tokens in option_tokens:  # every option starts at the same position
            spans.append((len(input_ids), len(input_ids) + len(tokens)))
            input_ids += tokens
            position_ids += range(len(prompt_tokens), len(prompt_tokens) + len(tokens))
        answer_start = len(input_ids)
        input_ids += self.answer
        position_ids += range(max(position_ids) + 1, max(position_ids) + 1 + len(self.answer))

        L, P = len(input_ids), len(prompt_tokens)
        mask = torch.zeros(L, L)
        mask[:P, :P] = torch.tril(torch.ones(P, P))
        for begin, end in spans:  # each option sees the prompt and its own earlier tokens only
            mask[begin:end, :P] = 1
            mask[begin:end, begin:end] = torch.tril(torch.ones(end - begin, end - begin))
        mask[answer_start:, :answer_start] = 1
        mask[answer_start:, answer_start:] = torch.tril(torch.ones(L - answer_start, L - answer_start))

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": mask,
            # Options are read at their last content token, before </option>
            "choice_read_idx": [end - 1 - len(self.option_end) for _, end in spans],
            "answer_end_idx": L - 1,
            "task_type": TASK_TYPES[question["type"]],
            "keys": keys,
        }

    def collate(self, batch):
        B = len(batch)
        L = max(len(ex["input_ids"]) for ex in batch)
        C = max(len(ex["choice_read_idx"]) for ex in batch)
        input_ids = torch.full((B, L), self.tok.pad_token_id, dtype=torch.long)
        position_ids = torch.zeros((B, L), dtype=torch.long)
        # Padded rows attend only to themselves, otherwise their softmax row is all -inf
        attention_mask = torch.eye(L).repeat(B, 1, 1)
        choice_read_idx = torch.zeros((B, C), dtype=torch.long)
        choice_mask = torch.zeros((B, C), dtype=torch.bool)
        answer_end_idx = torch.zeros(B, dtype=torch.long)
        for b, ex in enumerate(batch):
            n, c = len(ex["input_ids"]), len(ex["choice_read_idx"])
            input_ids[b, :n] = torch.tensor(ex["input_ids"])
            position_ids[b, :n] = torch.tensor(ex["position_ids"])
            attention_mask[b, :n, :n] = ex["attention_mask"]
            choice_read_idx[b, :c] = torch.tensor(ex["choice_read_idx"])
            choice_mask[b, :c] = True
            answer_end_idx[b] = ex["answer_end_idx"]
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": torch.where(attention_mask == 0, -float("inf"), 0.0).unsqueeze(1),
            "choice_read_idx": choice_read_idx,
            "choice_mask": choice_mask,
            "answer_end_idx": answer_end_idx,
            "task_type": torch.tensor([ex["task_type"] for ex in batch], dtype=torch.long),
        }
