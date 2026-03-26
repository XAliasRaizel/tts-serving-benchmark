import asyncio
import time

class DynamicBatcher:
    def __init__(self, model, batch_size=8, timeout=0.02):
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue = []

    async def start(self):
        asyncio.create_task(self._batch_loop())

    async def _batch_loop(self):
        while True:
            if not self.queue:
                await asyncio.sleep(0.001)
                continue

            start = time.time()

            while len(self.queue) < self.batch_size and (time.time() - start) < self.timeout:
                await asyncio.sleep(0.001)

            batch = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]

            await self._process(batch)

    async def _process(self, batch):
        texts = [item["text"] for item in batch]
        outputs = self.model.generate(texts)

        for i, item in enumerate(batch):
            item["future"].set_result(outputs[i])

    async def enqueue(self, text):
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        start_time = time.time()

        self.queue.append({
            "text": text,
            "future": future,
            "start_time": start_time
        })

        result = await future

        first_chunk_time = time.time()

        return {
            "ttft": first_chunk_time - start_time,
            "audio_duration": result["audio_duration"],
            "end_time": time.time()
        }