from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


def model_dump(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def split_pipe_separated_values(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [value.strip() for value in raw.split("|") if value.strip()]


__all__ = ["model_dump", "split_pipe_separated_values"]
