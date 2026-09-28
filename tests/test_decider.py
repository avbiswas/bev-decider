"""The package must reproduce the training code's probabilities for the released weights.

parity_cases.json holds 50 questions (choice / noul / score, text and JSON states, 4 states over 6,000 characters)
with the probabilities the training code produced in fp32 on CPU. Tests download the model on first run.
"""

import json
from pathlib import Path

import pytest

from bev_decider import load

CASES = json.loads((Path(__file__).parent / "parity_cases.json").read_text())


@pytest.fixture(scope="module")
def decider():
    return load(revision=CASES["model_revision"], device="cpu")


def probs(answer):
    return list(answer["probabilities"].values()) if "probabilities" in answer else [1 - answer["noul"], answer["noul"]]


@pytest.mark.parametrize("i", range(len(CASES["cases"])))
def test_parity(decider, i):
    case = CASES["cases"][i]
    answer = decider.decide(case["state"], {"q": case["question"]})["q"]
    assert probs(answer) == pytest.approx(case["expected_probabilities"], abs=1e-4)


def test_batched_equals_single(decider):
    cases = CASES["cases"][:6]
    state = cases[0]["state"]
    questions = {str(i): c["question"] for i, c in enumerate(cases)}
    batched = decider.decide(state, questions, batch_size=6)
    for qid, question in questions.items():
        single = decider.decide(state, {qid: question})[qid]
        assert probs(batched[qid]) == pytest.approx(probs(single), abs=1e-4)


def test_choice_order_invariant(decider):
    case = next(c for c in CASES["cases"] if c["question"]["type"] == "choice" and len(c["question"]["criteria"]) >= 3)
    question = case["question"]
    reversed_question = {**question, "criteria": dict(reversed(list(question["criteria"].items())))}
    a = decider.decide(case["state"], {"q": question})["q"]
    b = decider.decide(case["state"], {"q": reversed_question})["q"]
    assert a["choice"] == b["choice"]
    for key, p in a["probabilities"].items():
        assert b["probabilities"][key] == pytest.approx(p, abs=1e-4)


def test_server(decider):
    from fastapi.testclient import TestClient

    from bev_decider.serve import create_app

    client = TestClient(create_app(decider))
    assert client.get("/v1/models").status_code == 200
    body = {"state": {"message": "Refund the duplicate charge today or we cancel."},
            "questions": {"urgent": {"type": "noul", "instructions": "Is there time pressure?"},
                          "intent": {"type": "choice", "instructions": "What does the customer want?",
                                     "criteria": {"refund": "money back", "technical_help": "a bug"}}}}
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200
    answers = r.json()["answers"]
    direct = decider.decide(body["state"], body["questions"])
    assert answers["intent"]["choice"] == direct["intent"]["choice"]
    assert answers["urgent"]["noul"] == pytest.approx(direct["urgent"]["noul"], abs=1e-6)
    assert client.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "bad"}}}).status_code == 400
