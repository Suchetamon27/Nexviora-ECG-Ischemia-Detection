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
        clients_list = list(self.clients)
        if not clients_list:
            return

        for websocket in clients_list:
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"[StreamManager Broadcast Error] {type(e).__name__}: {e}")
                disconnected.append(websocket)

        for websocket in disconnected:
            self.disconnect(websocket)

# Shared singleton
stream_manager = StreamManager()
