import time
import torch

class TTSModel:
    def __init__(self):
        print("GPU Available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))

    def generate(self, texts):
        # Simulate GPU compute
        time.sleep(0.05)

        outputs = []
        for text in texts:
            audio_duration = len(text) * 0.02
            outputs.append({
                "audio_duration": audio_duration
            })
        return outputs