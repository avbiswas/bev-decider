# bev-decider

Run [**bev-decider-0.4B**](https://huggingface.co/avbiswas/bev-decider-0.4B), a 0.4B-parameter System One decision model. It reads a state (text or JSON) and typed questions about it, and returns calibrated probabilities in a single forward pass. It uses TypeSafe Jev's question and answer format, so a local server can stand in for the `/v1/systemone` API.

- **0.4B parameters.** It runs on a laptop CPU, Apple Silicon or any GPU.
- **Choice-order invariant.** Options are read in parallel from the same position, so reordering them cannot change the answer.
- **Typed answers:** `choice` (a key and probabilities), `noul` (P(yes)) and `score` (an expected level and probabilities).

See the [model card](https://huggingface.co/avbiswas/bev-decider-0.4B) for benchmarks and known weaknesses.

## Install

```bash
pip install bev-decider            # library
pip install "bev-decider[serve]"   # + local /v1/systemone server
```

The weights (about 35 MB) and the base `Qwen/Qwen3-0.6B` are downloaded from the Hugging Face Hub on first use.

## Python

```python
from bev_decider import load

decider = load()  # avbiswas/bev-decider-0.4B; pass device="cpu" | "mps" | "cuda" to override

answers = decider.decide(
    state={"message": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel."},
    questions={
        "intent": {"type": "choice", "instructions": "What does the customer want?",
                   "criteria": {"refund": "money returned or a duplicate charge reversed",
                                "technical_help": "a bug, outage or integration problem",
                                "cancellation": "wants to cancel or downgrade"}},
        "urgent": {"type": "noul", "instructions": "Does the message communicate time pressure or a deadline?"},
        "anger": {"type": "score", "instructions": "How angry is the customer?",
                  "criteria": ["calm", "mildly annoyed", "frustrated", "furious"]},
    },
)
# {"intent": {"type": "choice", "choice": "refund",
#             "probabilities": {"refund": 0.996, "technical_help": 0.000006, "cancellation": 0.0045}},
#  "urgent": {"type": "noul", "noul": 0.19},
#  "anger":  {"type": "score", "score": 0.88,
#             "probabilities": {"0": 0.45, "1": 0.25, "2": 0.27, "3": 0.03}}}
```

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
  "questions": {"damaged": {"type": "noul", "instructions": "Was the item damaged on arrival?"}}
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

The tests check that the package reproduces the probabilities of the released checkpoint on 50 fixed questions, in fp32 on CPU. They also check that answers do not depend on option order and that the server works.

## License

The code is Apache-2.0. The model weights are CC-BY-NC-4.0 (see the model card).
