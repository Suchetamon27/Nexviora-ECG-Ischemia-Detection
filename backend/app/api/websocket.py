from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..streaming.manager import stream_manager

router = APIRouter()

@router.websocket("/ws/live")
async def live_stream(websocket: WebSocket):
    await stream_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive & listen for client messages
            msg = await websocket.receive_text()
    except WebSocketDisconnect:
        stream_manager.disconnect(websocket)
    except Exception:
        stream_manager.disconnect(websocket)
