import httpx
import time

URL = "http://127.0.0.1:8000/tts"

async def send_request(text):
    async with httpx.AsyncClient(timeout=60.0) as client:
        start = time.time()

        response = await client.post(URL, json={"text": text})
        data = response.json()

        ttft = data["ttft"]
        generation_time = data["generation_time"]
        audio_duration = data["audio_duration"]

        rtf = generation_time / audio_duration

        return ttft, rtf