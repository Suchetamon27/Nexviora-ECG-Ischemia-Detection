from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..streaming.manager import StreamManager


router = APIRouter()

manager = StreamManager()


@router.websocket("/ws/live")
async def live_stream(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)