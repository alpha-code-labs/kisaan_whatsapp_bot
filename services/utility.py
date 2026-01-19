import asyncio
from typing import Callable, Any

def set_timeout(
    delay_seconds: float,
    func: Callable,
    *args: Any,
    **kwargs: Any
):
    async def _runner():
        await asyncio.sleep(delay_seconds)

        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            await result

    asyncio.create_task(_runner())
