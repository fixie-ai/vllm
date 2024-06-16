from typing import Dict, Tuple, Type, Union

import torch

from vllm.config import ModelConfig, AudioLanguageConfig
from vllm.logger import init_logger
from vllm.sequence import SequenceData
from vllm.transformers_utils.image_processor import cached_get_image_processor

from .base import MultiModalData, MultiModalPlugin

logger = init_logger(__name__)


def _get_dummy_seq_data(
    seq_len: int, alm_config: AudioLanguageConfig
) -> SequenceData:
    # NOTE: We assume that <image> token is repeated `image_feature_size` times
    # and then concatenated with the text prompt
    # TODO: Enable other ways of inserting the image into the prompt

    token_ids = [alm_config.audio_token_id] * alm_config.image_feature_size
    token_ids += [0] * (seq_len - alm_config.image_feature_size)

    return SequenceData(token_ids)


def _get_dummy_values(alm_config: AudioLanguageConfig) -> torch.Tensor:
    if alm_config.image_processor is None:
        values_dtype = torch.float16
    else:
        values_dtype = torch.uint8

    return torch.zeros(alm_config.image_input_shape, dtype=values_dtype)


def get_dummy_audio_data(
    seq_len: int,
    model_config: ModelConfig,
    alm_config: AudioLanguageConfig,
) -> Tuple[SequenceData, MultiModalData]:
    """Standard dummy data factory for image data (to be used in
    :meth:`vlm.multimodal.MultiModalRegistry.register_dummy_data`)."""
    seq_data = _get_dummy_seq_data(seq_len, alm_config)
    values = _get_dummy_values(alm_config)
    fake_mm_data = AudioSampleData(values)
    return seq_data, fake_mm_data


class AudioSampleData(MultiModalData):
    """
    The pixel data of an image. Can be one of:

    - :class:``PIL.Image``: An image object. Requires that a HuggingFace
      processor is available to the model.####??#?#?
    - :class:``torch.Tensor``: The raw pixel data which is passed to the model
      without additional pre-processing.
    """

    def __init__(self, data: torch.Tensor) -> None:
        if isinstance(image, Image.Image):
            # So that this class can be created inside the Image context manager
            image.load()

        self.image = image

    def __repr__(self) -> str:
        image = self.image
        if isinstance(image, Image.Image):
            return f"{type(self).__name__}(image={image})"

        return (
            f"{type(self).__name__}(image=torch.Tensor(shape="
            f"{image.shape}, dtype={image.dtype}))"
        )


class AudioSamplePlugin(MultiModalPlugin[AudioSampleData]):
    def get_data_type(self) -> Type[AudioSampleData]:
        return AudioSampleData

    def _get_hf_image_processor(
        self, model_config: ModelConfig, vlm_config: AudioLanguageConfig
    ):
        if vlm_config is None or vlm_config.image_processor is None:
            return None

        return cached_get_image_processor(
            vlm_config.image_processor,
            trust_remote_code=model_config.trust_remote_code,
            revision=vlm_config.image_processor_revision,
        )

    def _default_input_processor(
        self,
        data: AudioSampleData,
        model_config: ModelConfig,
        vlm_config: AudioLanguageConfig,
    ) -> Dict[str, torch.Tensor]:
        image = data.image

        if isinstance(image, Image.Image):
            image_processor = self._get_hf_image_processor(
                model_config, vlm_config
            )
            if image_processor is None:
                raise RuntimeError(
                    "No HuggingFace processor is available"
                    "to process the image object"
                )
            try:
                return (
                    image_processor.preprocess(image, return_tensors="pt")
                    .to(model_config.dtype)
                    .data
                )
            except Exception:
                logger.error("Failed to process image (%s)", image)
                raise
        elif isinstance(image, torch.Tensor):
            pixel_values = image.to(model_config.dtype)

            return {"pixel_values": pixel_values}

        raise TypeError(f"Invalid image type: {type(image)}")
