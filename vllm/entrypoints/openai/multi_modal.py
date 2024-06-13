import aiohttp
import base64
import math
import torch
import torchaudio
import transformers
import io
from urllib.parse import urlparse

from vllm.sequence import MultiModalData
torch.set_printoptions(edgeitems=8, sci_mode=True)


SAMPLE_RATE = 16000


async def load_media(url: str) -> MultiModalData:
    parsed = urlparse(url)
    if parsed.scheme == "data":        
        tokens = parsed.path.split(';')
        if len(tokens) < 2:    
            raise ValueError("Invalid data URI format")
        mime_type = tokens[0]
        encoding, encoded = tokens[-1].split(",")
        if encoding != 'base64':            
            raise ValueError("Invalid data URI encoding")
        data = base64.b64decode(encoded)    
    else:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                mime_type = response.headers["Content-Type"]   
                data = await response.read()             
    return await _load_media_object(mime_type, data)
        
async def _load_media_object(mime_type: str, data: bytes) -> MultiModalData:
    if mime_type.startswith("image/"):
        raise ValueError("Not yet implemented")
    elif mime_type.startswith("audio/"):
        return _make_audio_object(data)
    else:
        raise ValueError("Not yet implemented")
import librosa
import numpy as np
audio_processor = None  
def _make_audio_object(audio: bytes) -> MultiModalData: 
    global audio_processor
    if not audio_processor:
        audio_processor = transformers.AutoProcessor.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )
    #audio_tensor, sr = torchaudio.load(io.BytesIO(audio))
    #transform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
    audio = librosa.load(io.BytesIO(audio), sr=SAMPLE_RATE)[0]
    audio = np.expand_dims(audio, 0)
    audio_tensor = torch.from_numpy(audio)
    print("audio tensor", audio_tensor)
    x = audio_processor(audio_tensor, sampling_rate=SAMPLE_RATE, padding="longest", return_tensors="pt").to(torch.bfloat16)
    audio_processed = x.get("input_features") or x.get("input_values")
    print("audio processed", audio_processed)
    audio_processed = audio_processed.squeeze(0)
    
    #audio_tensor = transform(audio_tensor).to(torch.float16) 
    return MultiModalData(type=MultiModalData.Type.AUDIO, data=audio_processed)
                                  
def process_prompt(prompt: str, data: MultiModalData) -> str:
    assert(data.type == MultiModalData.Type.AUDIO)
    audio_num_tokens = math.ceil(data.data.shape[1] / SAMPLE_RATE * 6.2375)
    return prompt.replace("<|audio|>", "<|audio|>" * audio_num_tokens)
    