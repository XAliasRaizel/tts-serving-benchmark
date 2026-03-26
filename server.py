from fastapi import FastAPI
from pydantic import BaseModel
from model import TTSModel
from batcher import DynamicBatcher

app = FastAPI()

model = TTSModel()
batcher = DynamicBatcher(model)

class TTSRequest(BaseModel):
    text: str

@app.on_event("startup")
async def startup():
    await batcher.start()

@app.post("/tts")
async def tts(req: TTSRequest):
    return await batcher.enqueue(req.text)