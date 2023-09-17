import os
import matplotlib
import matplotlib.pyplot as plt
import requests
import torch
import torchaudio
import pdb
from pvrecorder import PvRecorder

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
        emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.
        Returns:
        str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1) # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])
    
def speech2text(SPEECH_FILE):
    waveform, sample_rate = torchaudio.load(SPEECH_FILE)
    waveform = waveform.to(device)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    
    with torch.inference_mode():
        emission, _ = model(waveform)
    
    decoder = GreedyCTCDecoder(labels=label)
    transcript = decoder(emission[0])

    return transcript

SPEECH_URL = "https://keithito.com/LJ-Speech-Dataset/LJ025-0076.wav"
SPEECH_FILE = "_assets/speech.wav"
if not os.path.exists(SPEECH_FILE):
    os.makedirs("_assets", exist_ok=True)
    with open(SPEECH_FILE, "wb") as file:
        file.write(requests.get(SPEECH_URL).content)
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
label = list(bundle.get_labels())
label[1] = ' '
label = [letter.lower() for letter in label]

model = bundle.get_model().to(device)

SPEECH_FILE = './_assets/speech.wav'
script = speech2text(SPEECH_FILE)

recorder = PvRecorder(device_index=-1, frame_length=512)
try:
    recorder.start()

    while True:
        frame = recorder.read()
        # Do something ...
except KeyboardInterrupt:
    recorder.stop()
finally:
    recorder.delete()