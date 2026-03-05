from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class TransactionIn(BaseModel):
    date: date
    description: str = Field(min_length=1)
    merchant: str | None = None
    amount: float
    type: Literal["debit", "credit"] | None = None


class AnalyzeRequest(BaseModel):
    transactions: list[TransactionIn]
    month: str | None = Field(
        default=None,
        description="Optional month filter in YYYY-MM. If omitted, latest month in the data is used.",
    )


class TipRequest(BaseModel):
    analysis: dict[str, Any] | None = Field(
        default=None,
        description="Optional analysis payload (currently unused by generic tip endpoint).",
    )


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    message: str = Field(min_length=1, max_length=2000)


class CoachRequest(BaseModel):
    message: str = Field(min_length=2, max_length=1200)
    analysis: dict[str, Any] | None = Field(
        default=None,
        description="Optional analysis payload for context-aware coaching.",
    )
    history: list[ChatTurn] | None = Field(
        default=None,
        description="Optional recent chat history for conversational context.",
    )
