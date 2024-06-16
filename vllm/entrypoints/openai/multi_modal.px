import aiohttp
import base64
import math
import torch
import torchaudio
import io
from urllib.parse import urlparse

from vllm.sequence import MultiModalData

SAMPLE_RATE = 16000


async def load_media(url: str) -> MultiModalData:
    parsed = urlparse(url)
    if parsed.scheme == "data":
        tokens = parsed.path.split(";")
        if len(tokens) < 2:
            raise ValueError("Invalid data URI format")
        mime_type = tokens[0]
        encoding, encoded = tokens[-1].split(",")
        if encoding != "base64":
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


def _make_audio_object(audio: bytes) -> MultiModalData:
    global audio_processor
    if not audio_processor:
        audio_processor = transformers.AutoProcessor.from_pretrained(
            "openai/whisper-small"
        )
    audio, _ = librosa.load(io.BytesIO(audio), sr=SAMPLE_RATE)
    audio_tensor = torch.from_numpy(audio)
    processed = audio_processor(
        audio_tensor,
        sampling_rate=SAMPLE_RATE,
        padding="longest",
        return_tensors="pt",
    )
    audio_features = processed["input_features"]

    return MultiModalData(type=MultiModalData.Type.AUDIO, data=audio_features)


def process_prompt(prompt: str, data: MultiModalData) -> str:
    assert data.type == MultiModalData.Type.AUDIO
    audio_num_tokens = math.ceil(data.data.shape[2] / 16)
    return prompt.replace(
        "<|audio|>", "<|reserved_special_token_0|>" * audio_num_tokens
    )
