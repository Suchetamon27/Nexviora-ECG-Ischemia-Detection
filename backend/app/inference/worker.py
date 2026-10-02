import asyncio
from typing import Callable

from .ollama import OllamaECGClient
from .render import render_ecg_window


class InferenceWorker:
    def __init__(
        self,
        client: OllamaECGClient,
        window_queue: asyncio.Queue,
        result_callback: Callable,
        system_prompt: str,
        user_prompt: str,
        format_schema: dict | None = None,
    ) -> None:
        self.client = client
        self.window_queue = window_queue
        self.result_callback = result_callback
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.format_schema = format_schema

    async def run(self) -> None:
        while True:
            window = await self.window_queue.get()

            try:
                image = render_ecg_window(window)

                result = await self.client.infer_image(
                    image,
                    self.system_prompt,
                    self.user_prompt,
                    format=self.format_schema,
                )

                await self.result_callback(result)

            except Exception as exc:
                await self.result_callback(
                    {
                        "error": str(exc),
                    }
                )

            finally:
                self.window_queue.task_done()