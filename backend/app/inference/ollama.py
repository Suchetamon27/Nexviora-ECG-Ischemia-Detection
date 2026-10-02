import base64
from typing import Any

import httpx


class OllamaECGClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature

    async def infer_image(
        self,
        image_bytes: bytes,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        image_b64 = base64.b64encode(image_bytes).decode("ascii")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [image_b64],
                },
            ],
            "stream": False,
            "temperature": self.temperature,
        }

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=5.0,
                read=30.0,
                write=10.0,
                pool=5.0,
            )
        ) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )

        response.raise_for_status()

        return response.json()