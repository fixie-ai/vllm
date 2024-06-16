# coding=utf-8
# NB chua: modified from https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llava/modeling_llava.py#L350
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Ultravox model."""

from typing import Iterable, List, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from transformers import CONFIG_MAPPING, PretrainedConfig, logging

from vllm.attention import AttentionMetadata
from vllm.config import AudioLanguageConfig, CacheConfig
from vllm.entrypoints.openai import multi_modal
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, )
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.whisper_streaming import WhisperEncoder

from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
import time

_KEYS_TO_MODIFY_MAPPING = {
    "language_model.lm_head": "lm_head",
    "language_model.model": "language_model",
}

logger = logging.get_logger(__name__)


def _merge_audio_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    audio_embeddings: torch.Tensor,
    audio_token_id: int,
):
    """
    In place merges in audio_embeddings with inputs_embeds.
    Occasionally the audio embeddings are longer than the supplied audio tokens
    in which case we just truncate the excess audio embeddings (in practice, just 1).
    """
    mask = input_ids == audio_token_id
    num_audio_tokens = torch.sum(mask)
    audio_embeddings = audio_embeddings[:, :num_audio_tokens, :]
    inputs_embeds[mask] = audio_embeddings.view(-1, audio_embeddings.shape[-1])


class UltravoxConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`UltravoxModel`]. It is used to instantiate an
    Ultravox model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        audio_config (`Wav2Vec2Config`,  *optional*):
            Custom audio config or dict
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `LlamaConfig` or `MistralConfig`.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        audio_token_index (`int`, *optional*, defaults to 32000):
            The audio token index to encode the audio prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the resulting model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~UltravoxModel`]

    Example:

    ```python
    >>> from transformers import UltravoxModel, Wav2Vec2Config, UltravoxConfig, LlamaConfig

    >>> # Initializing an audio encoder config
    >>> audio_config = Wav2Vec2Config()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a default configuration
    >>> configuration = UltravoxConfig(audio_config, text_config)

    >>> # Initializing a completely untrained model from the configuration
    >>> model = UltravoxModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # Initialize a model from pretrained checkpoints and random projector weights
    >>> config = UltravoxConfig(audio_model_id="facebook/wav2vec2-base-960h", text_model_id="meta-llama/Llama-2-7b-chat-hf")
    ```"""

    model_type = "ultravox"
    is_composition = False

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_model_id=None,
        text_model_id=None,
        ignore_index=-100,
        audio_token_index=128002,
        vocab_size=128256,
        hidden_size=4096,
        stack_factor=8,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.audio_token_index = audio_token_index
        self.vocab_size = vocab_size

        self.audio_model_id = audio_model_id
        self.text_model_id = text_model_id

        self.audio_config = audio_config
        self.text_config = text_config

        self.hidden_size = hidden_size
        self.stack_factor = stack_factor

        if isinstance(self.text_config, dict):
            text_config["model_type"] = (text_config["model_type"]
                                         if "model_type" in text_config else
                                         "llama")
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](
                **text_config)
            self.vocab_size = self.text_config.vocab_size
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["llama"]()

        if isinstance(self.audio_config, dict):
            audio_config["model_type"] = (audio_config["model_type"]
                                          if "model_type" in audio_config else
                                          "wav2vec2")
            self.audio_config = CONFIG_MAPPING[audio_config["model_type"]](
                **audio_config)
            self.vocab_size = self.audio_config.vocab_size
        elif audio_config is None:
            self.audio_config = CONFIG_MAPPING["wav2vec2"]()

        super().__init__(**kwargs)


class ProjectionLayer(nn.Module):

    def __init__(self, stack_factor: int = 8):
        super().__init__()
        # NB chua: stack_factor is the factor by which the audio embeddings are stacked
        # ideally this should be picked according to your hardware and should be a multiple of 8!
        # https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/
        self.stack_factor = stack_factor

    def _pad_and_stack(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        "Stack audio embeddings to downsample in time dimension, then pad to the nearest multiple of `stack_factor`"
        B, T, C = audio_embeds.shape
        audio_embeds = F.pad(
            audio_embeds, (0, 0, 0, self.stack_factor - T % self.stack_factor))
        B, T, C = audio_embeds.shape
        audio_embeds = audio_embeds.view(B, T // self.stack_factor,
                                         C * self.stack_factor)
        return audio_embeds


class RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        """
        From huggingface's LlamaRMSNorm
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
        """
        super().__init__()
        # the default initialization here is to 1
        # however, https://arxiv.org/abs/2206.10139 shows stronger improvements initializing to smaller weights
        # we arbitrarily pick 0.4 here, seemed like good results
        self.weight = nn.Parameter(torch.full((hidden_size, ), 0.4))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SwiGLU(nn.Module):

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class UltravoxProjector(ProjectionLayer):

    def __init__(self, config: UltravoxConfig):
        self.hidden_dim = config.hidden_size
        super().__init__(config.stack_factor)
        self.ln_pre = RMSNorm(config.audio_config.hidden_size *
                              self.stack_factor)
        self.linear_1 = nn.Linear(
            config.audio_config.hidden_size * self.stack_factor,
            self.hidden_dim,
            bias=False,
        )
        self.act = SwiGLU()
        self.linear_2 = nn.Linear(self.hidden_dim // 2,
                                  config.text_config.hidden_size,
                                  bias=False)
        self.ln_post = RMSNorm(config.text_config.hidden_size)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        audio_features = self._pad_and_stack(audio_features)
        audio_features = self.ln_pre(audio_features)
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.ln_post(hidden_states)
        return hidden_states


class UltravoxModel(nn.Module):

    def __init__(
        self,
        config: UltravoxConfig,
        audio_language_config: AudioLanguageConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional["QuantizationConfig"] = None,
    ):
        from vllm.model_executor.models.llama import LlamaModel

        super().__init__()
        self.config = config
        self.audio_language_config = audio_language_config
        assert self.audio_language_config

        # config.audio_model_id = config.audio_model_id or config.audio_config._name_or_path
        if config.audio_model_id is not None:
            self.audio_tower = WhisperEncoder.from_pretrained(
                config.audio_model_id)
        else:
            self.audio_tower = WhisperEncoder.from_config(config.audio_config)
        self.audio_tower = self.audio_tower.to(torch.bfloat16).to("cuda")
        torch.compile(self.audio_tower)
        self.audio_tower.eval()

        self.multi_modal_projector = (UltravoxProjector(config).to(
            torch.bfloat16).to("cuda"))
        self.quant_config = quant_config

        self.language_model = LlamaModel(config.text_config, cache_config,
                                         quant_config)
        self.unpadded_vocab_size = config.text_config.vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.text_config.hidden_size,
            org_num_embeddings=self.language_model.org_vocab_size,
        )
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.text_config.vocab_size,
                                                logit_scale)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        audio_input: Optional[torch.Tensor] = None,
    ) -> SamplerOutput:  # noqa: E501
        """Run forward pass for Llava 1.5.

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted audio embeddings.
        Concretely, consider a text prompt:
        "<|audio|>\nUSER: What's the content of the audio?\nASSISTANT:".
        Tokenizer outputs:
        [1, 32000, 29871, 13, 11889, 29901, 1724, 29915, 29879, 278,
        2793, 310, 278, 1967, 29973, 13, 22933, 9047, 13566, 29901].
        The to-be-inserted audio has a size that is essentially 6.25 tokens
        per second of audio.
        `input_ids` is thus [1, 32000, ..., 32000, 29871, 13, 11889, 29901,
        1724, 29915, 29879, 278, 2793, 310, 278, 1967, 29973, 13, 22933,
        9047, 13566, 29901].
        For a 3-second clip, there will be ~19 `32000` in the `input_ids`.
        (32000 is the token id for `<audio>` when using Llama 2 as the backbone.)

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            image_input: A batch of audio inputs, [1, 80, M].
        """
        t1 = time.perf_counter()
        t2 = None
        t3 = None
        if audio_input is not None:
            audio_input = audio_input.to(self.audio_tower.dtype)
            inputs_embeds = self.language_model.get_input_embeddings(input_ids)
            audio_features = self.audio_tower(audio_input).last_hidden_state
            t2 = time.perf_counter()
            audio_features = audio_features.to(self.audio_tower.dtype)

            audio_embeddings = self.multi_modal_projector(audio_features).to(
                inputs_embeds.dtype)
            audio_token_count = torch.sum(
                input_ids == self.audio_language_config.audio_token_id)
            if audio_embeddings.shape[1] != audio_token_count:
                logger.warning(
                    f"The number of audio tokens in the prompt ({audio_token_count}) "
                    f"does not match the number of audio embeddings "
                    f"({audio_embeddings.shape[1]}).")

            t3 = time.perf_counter()
            _merge_audio_embeddings(
                input_ids,
                inputs_embeds,
                audio_embeddings,
                self.audio_language_config.audio_token_id,
            )
            input_ids = None
        else:
            inputs_embeds = None

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )
        t4 = time.perf_counter()
        if t2 and t3:
            logger.info(f"encoder  : {((t2 - t1)*1000):0f} ms")
            logger.info(f"projector: {((t3 - t2)*1000):0f} ms")
        logger.info(f"llm:       {((t4 - (t3 or t1))*1000):0f} ms")
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader, )

        # only doing this for language model part for now.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            use_default_weight_loading = False
            if "audio" in name:
                if self.audio_tower is not None:
                    # We only do sharding for language model and
                    # not audio model for now.
                    use_default_weight_loading = True
            else:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    new_name = name.replace(weight_name, param_name)
                    param = params_dict[new_name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    use_default_weight_loading = True
            if use_default_weight_loading:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
