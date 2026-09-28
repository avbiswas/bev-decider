# bev-decider

Run [**bev-decider-0.4B**](https://huggingface.co/avbiswas/bev-decider-0.4B), a 0.4B-parameter System One decision model. It reads a state (text or JSON) and typed questions about it, and returns calibrated probabilities in a single forward pass. It uses TypeSafe Jev's question and answer format, so a local server can stand in for the `/v1/systemone` API.

- **0.4B parameters.** It runs on a laptop CPU, Apple Silicon or any GPU.
- **Choice-order invariant.** Every option starts at the same position id and is read in parallel, so reordering the options cannot change the answer, and there is no bias toward the first or last option. Inside the backbone, the attention mask lets each option's tokens see only the question and their own earlier tokens, never another option. Each option's embedding then goes through a small, newly trained self-attention head, which is where the options are compared with each other. That head has no position information either, so the whole model is order invariant. This is exact in fp32: over 960 random shuffles of 3–12 options, no probability moved by more than 1e-5. On GPU or Apple Silicon the default bf16 inference adds rounding noise (about 0.001 typical), which can only flip near-ties.
- **Typed answers:** `choice` (a key and probabilities), `noul` (P(yes)) and `score` (an expected level and probabilities).

See the [model card](https://huggingface.co/avbiswas/bev-decider-0.4B) for benchmarks and known weaknesses.

## Install

```bash
pip install bev-decider            # library
pip install "bev-decider[serve]"   # + local /v1/systemone server
```

The model is a single ~1 GB file, downloaded from the Hugging Face Hub on first use.

## Python

```python
from bev_decider import load

decider = load()  # downloads avbiswas/bev-decider-0.4B

state = {
    "message": (
        "URGENT: you charged my card twice this month. "
        "Refund the duplicate within 24 hours or I'm disputing it with my bank."
    )
}

questions = {
    "intent": {
        "type": "choice",
        "instructions": "What does the customer want?",
        "criteria": {
            "refund": "money returned or a duplicate charge reversed",
            "technical_help": "a bug, outage or integration problem",
            "cancellation": "wants to cancel or downgrade",
        },
    },
    "urgent": {
        "type": "noul",
        "instructions": "Does the message communicate time pressure or a deadline?",
    },
    "anger": {
        "type": "score",
        "instructions": "How angry is the customer?",
        "criteria": ["calm", "mildly annoyed", "frustrated", "furious"],
    },
}

answers = decider.decide(state, questions)
```

Output:

```json
{
  "intent": {
    "type": "choice",
    "choice": "refund",
    "probabilities": {"refund": 1.0, "technical_help": 0.0, "cancellation": 0.0}
  },
  "urgent": {"type": "noul", "noul": 0.99},
  "anger": {
    "type": "score",
    "score": 1.90,
    "probabilities": {"0": 0.09, "1": 0.10, "2": 0.65, "3": 0.17}
  }
}
```

To choose a device, pass `load(device="cpu")`, `"mps"` or `"cuda"`.

Question types:

| type | criteria | answer |
|---|---|---|
| `choice` | `{key: description}` (a description may be empty) | `choice`: the most likely key, and `probabilities` per key |
| `noul` | optional `{"true": ..., "false": ...}` | `noul`: P(yes) |
| `score` | a list of level descriptions, lowest first | `score`: the expected level, and `probabilities` per level |

Questions about the same state are batched together. States longer than 2,048 tokens are truncated; change the limit with `load(max_state_tokens=...)`. The model was trained on states up to 1,024 tokens.

## Server

```bash
bev-decider serve --port 8008
curl -s localhost:8008/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Order 1182 arrived with a cracked screen.",
  "questions": {
    "damaged": {"type": "noul", "instructions": "Was the item damaged on arrival?"}
  }
}'
```

`POST /v1/systemone` accepts `{"state", "questions", "model"?}` and returns `{"model", "answers", "latency_ms"}`. `GET /v1/models` describes the loaded model. The server has no authentication and binds to `127.0.0.1` by default.

## Command line

```bash
echo '{"state": "...", "questions": {...}}' | bev-decider decide
```

## Tests

```bash
uv run pytest
```

The tests check that the package reproduces the released model's probabilities on 50 fixed questions on CPU, and that they stay within 0.02 of the training checkpoint. They also check that answers do not depend on option order and that the server works.

## License

The code is Apache-2.0. The model weights are CC-BY-NC-4.0 (see the model card).
