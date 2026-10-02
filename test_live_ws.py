import asyncio
import json
import websockets
import urllib.request

async def test_client():
    uri = "ws://localhost:8000/ws/live"
    print("Connecting to WebSocket...")
    async with websockets.connect(uri) as ws:
        print("Connected! Calling /api/stream/start...")
        req = urllib.request.Request("http://localhost:8000/api/stream/start", method="POST")
        with urllib.request.urlopen(req) as resp:
            print("Start stream response:", resp.read().decode())

        print("Listening for 10 incoming packets...")
        count = 0
        for _ in range(15):
            msg_str = await asyncio.wait_for(ws.recv(), timeout=5.0)
            msg = json.loads(msg_str)
            print(f"Packet received: type={msg.get('type')}, HR={msg.get('hr')}, samples_count={len(msg.get('filtered_samples', []))}")
            count += 1
            if count >= 8:
                break

        print("Calling /api/stream/stop...")
        req = urllib.request.Request("http://localhost:8000/api/stream/stop", method="POST")
        with urllib.request.urlopen(req) as resp:
            print("Stop response:", resp.read().decode())

asyncio.run(test_client())
