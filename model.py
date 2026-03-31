import torch
import time
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

class TTSModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("GPU Available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))

        # Load models
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)

        # Speaker embedding (required)
        # embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.randn(1, 512).to(self.device)

    def generate(self, texts):
        outputs = []

        for text in texts:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)

            start = time.time()

            with torch.no_grad():
                speech = self.model.generate_speech(
                    inputs["input_ids"],
                    self.speaker_embeddings,
                    vocoder=self.vocoder
                )

            generation_time = time.time() - start

            audio = speech.cpu().numpy()
            sampling_rate = 16000
            audio_duration = len(audio) / sampling_rate

            outputs.append({
                "audio_duration": audio_duration,
                "generation_time": generation_time
            })

        return outputs