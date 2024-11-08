from typing import Tuple, Union, cast

import numpy as np

from vllm.inputs.registry import InputContext
from vllm.multimodal.base import MultiModalInputs, MultiModalPlugin


class AudioPlugin(MultiModalPlugin):
    """Plugin for audio data."""

    def get_data_key(self) -> str:
        return "audio"

    def hash_content(self, data: object) -> int:
        (audio, sr) = cast(Tuple[np.ndarray, Union[float, int]], data)
        return hash((audio.data.tobytes(), sr))

    def _default_input_mapper(self, ctx: InputContext, data: object,
                              **mm_processor_kwargs) -> MultiModalInputs:
        raise NotImplementedError("There is no default audio input mapper")

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        raise NotImplementedError(
            "There is no default maximum multimodal tokens")
