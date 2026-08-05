"""Inference chaqiruvlarini event loop'dan tashqarida, cheklangan sonda bajaradi.

HTTP (FastAPI) va gRPC adapterlari bir xil chegaradan foydalanadi — aks holda
ikki protokol birgalikda GPU xotirasini to'ldirib yuborishi mumkin.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, TypeVar

from starlette.concurrency import run_in_threadpool

T = TypeVar("T")

MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_INFERENCE", "4"))
_slots = asyncio.Semaphore(MAX_CONCURRENT)


async def run_inference(fn: Callable[..., T], *args: Any) -> T:
    """Bloklovchi model chaqiruvini xavfsiz bajaradi."""
    async with _slots:
        return await run_in_threadpool(fn, *args)


async def run_io(fn: Callable[..., T], *args: Any) -> T:
    """Yengil bloklovchi ish (rasm dekodlash) — chegarasiz, lekin loop'dan tashqarida."""
    return await run_in_threadpool(fn, *args)
