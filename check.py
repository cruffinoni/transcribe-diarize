import torch
print("MPS built:", torch.backends.mps.is_built())
print("MPS avail:", torch.backends.mps.is_available())
from pyannote.audio import Pipeline
Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
print("OK: pyannote pipeline loaded")