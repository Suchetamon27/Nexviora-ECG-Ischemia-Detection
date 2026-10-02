import asyncio

class StreamManager:
    def __init__(self) -> None:
        self.clients: set = set()

    async def connect(self, websocket) -> None:
        await websocket.accept()
        self.clients.add(websocket)
        print(f"[StreamManager] Client connected. Total: {len(self.clients)}")

    def disconnect(self, websocket) -> None:
        self.clients.discard(websocket)
        print(f"[StreamManager] Client disconnected. Total: {len(self.clients)}")

    async def broadcast(self, message: dict) -> None:
        disconnected = []
        for websocket in list(self.clients):
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)

        for websocket in disconnected:
            self.disconnect(websocket)

# Shared singleton
stream_manager = StreamManager()
