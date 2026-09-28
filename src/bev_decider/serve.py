"""A local server with TypeSafe's /v1/systemone request and response format.

    bev-decider serve --port 8008
    curl -s localhost:8008/v1/systemone -H 'content-type: application/json' \
      -d '{"state": "...", "questions": {"q": {"type": "noul", "instructions": "..."}}}'
"""

import threading
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from .decider import Decider

MODEL_NAME = "bev-decider-0.4b"


class Request(BaseModel):
    state: Any
    questions: dict[str, dict[str, Any]]
    model: str | None = None


def create_app(decider: Decider) -> FastAPI:
    app = FastAPI(title="bev-decider")
    lock = threading.Lock()  # one forward pass at a time on the device

    @app.get("/v1/models")
    def models():
        return {"models": [{"name": MODEL_NAME, "source": decider.name, "device": str(decider.device),
                            "base": decider.config["base_model"], "max_state_tokens": decider.encoder.max_state_tokens}]}

    @app.post("/v1/systemone")
    def systemone(req: Request, response: Response):
        if not req.questions:
            raise HTTPException(400, "questions is empty")
        start = time.perf_counter()
        try:
            with lock:
                answers = decider.decide(req.state, req.questions)
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(400, f"invalid question: {e}")
        response.headers["x-request-id"] = uuid.uuid4().hex
        return {"model": req.model or MODEL_NAME, "answers": answers,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1)}

    return app
