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
"""PyTorch Gazelle model."""

from email.mime import audio
from typing import Iterable, List, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from transformers import AutoModel, CONFIG_MAPPING, PretrainedConfig
from transformers import logging

from vllm.attention import AttentionMetadata
from vllm.config import AudioLanguageConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
import time

_KEYS_TO_MODIFY_MAPPING = {
    "language_model.lm_head": "lm_head",
    "language_model.model": "language_model",
}

logger = logging.get_logger(__name__)


def _merge_audio_embeddings(input_ids: torch.Tensor,
                            inputs_embeds: torch.Tensor,
                            audio_embeddings: torch.Tensor,
                            audio_token_id: int):
    """In place merges in vision_embeddings with inputs_embeds."""
    mask = (input_ids == audio_token_id)
    inputs_embeds[mask] = audio_embeddings.view(-1, audio_embeddings.shape[-1])


class GazelleConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GazelleForConditionalGeneration`]. It is used to instantiate an
    Gazelle model according to the specified arguments, defining the model architecture.

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
            `inputs_ids` passed when calling [`~GazelleForConditionalGeneration`]

    Example:

    ```python
    >>> from transformers import GazelleForConditionalGeneration, Wav2Vec2Config, GazelleConfig, LlamaConfig

    >>> # Initializing an audio encoder config
    >>> audio_config = Wav2Vec2Config()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a default configuration
    >>> configuration = GazelleConfig(audio_config, text_config)

    >>> # Initializing a completely untrained model from the configuration
    >>> model = GazelleForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # Initialize a model from pretrained checkpoints and random projector weights
    >>> config = GazelleConfig(audio_model_id="facebook/wav2vec2-base-960h", text_model_id="meta-llama/Llama-2-7b-chat-hf")
    ```"""

    model_type = "gazelle"
    is_composition = False

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_model_id=None,
        text_model_id=None,
        ignore_index=-100,
        audio_token_index=32000,
        vocab_size=32000,
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


class GazelleProjector(ProjectionLayer):

    def __init__(self, config: GazelleConfig):
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


class GazelleForConditionalGeneration(nn.Module):

    def __init__(self,
                 config: GazelleConfig,
                 audio_language_config: AudioLanguageConfig,
                 quant_config: Optional["QuantizationConfig"] = None):
        from vllm.model_executor.models.llama import LlamaModel
        super().__init__()
        self.config = config

        self.audio_language_config = audio_language_config
        assert self.audio_language_config

        if config.audio_model_id is not None:
            self.audio_tower = AutoModel.from_pretrained(config.audio_model_id)
        else:
            self.audio_tower = AutoModel.from_config(config.audio_config)
        self.audio_tower = self.audio_tower.to(torch.float16)
        torch.compile(self.audio_tower)
        self.audio_tower.eval()

        self.multi_modal_projector = GazelleProjector(config).to(torch.bfloat16)
        torch.compile(self.multi_modal_projector)
        self.quant_config = quant_config

        self.language_model = LlamaModel(config.text_config, quant_config)
        self.unpadded_vocab_size = config.text_config.vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.text_config.hidden_size,
            org_num_embeddings=self.language_model.org_vocab_size)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.text_config.vocab_size,
                                                logit_scale)
        self.sampler = Sampler()

    """
    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of
        )
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        audio_input: Optional[torch.Tensor] = None
    ) -> SamplerOutput:  # noqa: E501
        """Run forward pass for Llava 1.5.

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted image embeddings.
        Concretely, consider a text prompt:
        "<image>\nUSER: What's the content of the image?\nASSISTANT:".
        Tokenizer outputs:
        [1, 32000, 29871, 13, 11889, 29901, 1724, 29915, 29879, 278,
        2793, 310, 278, 1967, 29973, 13, 22933, 9047, 13566, 29901].
        The to-be-inserted image has a size of 576 (24 * 24) along the context
        length dimension.
        `input_ids` is thus [1, 32000, ..., 32000, 29871, 13, 11889, 29901,
        1724, 29915, 29879, 278, 2793, 310, 278, 1967, 29973, 13, 22933,
        9047, 13566, 29901].
        There will be 576 `32000` in the `input_ids`.
        (32000 is the token id for `<image>`.)

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        The model takes two types of image inputs: 
        PIXEL_VALUES and IMAGE_FEATURES.
        The following shows how each maps to huggingface implementation.
        PIXEL_VALUES: 
        - https://github.com/huggingface/transformers/blob/07bdbeb/src/transformers/models/llava/modeling_llava.py#L353
        IMAGE_FEATURES:
        - https://github.com/huggingface/transformers/blob/07bdbeb/src/transformers/models/llava/modeling_llava.py#L430
        before going through the multi modal projector.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            image_input: A batch of image inputs.
                For PIXEL_VALUES, expecting [1, 3, 336, 336].
                For IMAGE_FEATURES, expecting [1, 576, 1024].
        """       
        t1 = time.perf_counter()
        t2 = None
        t3 = None    
        if audio_input is not None:            
            audio_token_count = torch.sum(input_ids == self.audio_language_config.audio_token_id)
            logger.info(f"input_ids {input_ids}")        
            logger.info(f"audio_input {audio_input.shape if audio_input is not None else None} dtype {audio_input.dtype}")
            #if list(audio_input.shape[1:]) != [audio_token_count, 4096]:
            #    raise ValueError(
            #        f"The expected image tensor shape is batch dimension "
            #        f"plus "
            #        #f"{self.vision_language_config.image_input_shape[1:]}."
            #        f" You supplied {audio_input.shape}. "
            #        f"If you are using vLLM's entrypoint, make sure your "
            #        f"supplied image input is consistent with "
            #        f"image_input_shape in engine args.")
            
            inputs_embeds = self.language_model.get_input_embeddings(input_ids)
            if self.audio_tower is not None:
                #audio_input = audio_input.to(torch.bfloat16)
                audio_features = self.audio_tower(audio_input).last_hidden_state
                print("audio features", audio_features[:5])
                t2 = time.perf_counter()
                audio_features = audio_features.to(torch.bfloat16)              
                #if audio_features.shape[1] != audio_token_count:
                #    raise ValueError(
                #        f"The number of audio tokens in the prompt ({audio_token_count}) "
                #        f"does not match the number of audio tokens in the audio input "
                #        f"({audio_features.shape[1]})."
                #    )  
            else:
                audio_features = audio_input            
            audio_embeddings = self.multi_modal_projector(audio_features).to(inputs_embeds.dtype)                
            print("audio embeddings", audio_embeddings[:5])
            t3 = time.perf_counter()
            _merge_audio_embeddings(input_ids, inputs_embeds, audio_embeddings,
                                    self.audio_language_config.audio_token_id)  
            input_ids = None
        else:
            inputs_embeds = None

        hidden_states = self.language_model(input_ids=input_ids,
                                            positions=positions,
                                            kv_caches=kv_caches,
                                            attn_metadata=attn_metadata,
                                            inputs_embeds=inputs_embeds)
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
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
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
                    # not vision model for now.
                    use_default_weight_loading = True
            else:
                for (param_name, weight_name,
                     shard_id) in stacked_params_mapping:
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

"""

    def _merge_input_ids_with_audio_features(self, audio_features,
                                             inputs_embeds, input_ids,
                                             attention_mask, labels):
        num_audio_samples, num_audio_patches, embed_dim = audio_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(
            input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_audio_token_mask = input_ids == self.config.audio_token_index
        num_special_image_tokens = torch.sum(special_audio_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() *
                         (num_audio_patches - 1)) + sequence_length
        batch_indices, non_audio_indices = torch.where(
            input_ids != self.config.audio_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged audio-text sequence.
        # `special_audio_token_mask` identifies audio tokens. Each audio token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each audio token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = (torch.cumsum(
            (special_audio_token_mask * (num_audio_patches - 1) + 1), -1) - 1)
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:  # TODO FARZAD: I don't understand this
            new_token_positions += nb_image_pad[:,
                                                None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices,
                                                non_audio_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = torch.zeros(
            batch_size,
            max_embed_dim,
            dtype=attention_mask.dtype,
            device=inputs_embeds.device,
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim),
                self.config.ignore_index,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        # In case the audio model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_audio_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_audio_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<|audio|>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the audio features
        final_embedding[batch_indices,
                        text_to_overwrite] = inputs_embeds[batch_indices,
                                                           non_audio_indices]
        final_attention_mask[batch_indices,
                             text_to_overwrite] = attention_mask[
                                 batch_indices, non_audio_indices]
        if labels is not None:
            final_labels[batch_indices,
                         text_to_overwrite] = labels[batch_indices,
                                                     non_audio_indices]

        # 5. Fill the embeddings corresponding to the audio. Anything that is still zeros needs filling
        audio_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        audio_positions = audio_to_overwrite.to(torch.int16).cumsum(-1) - 1
        audio_left_pad_mask = audio_positions >= nb_image_pad[:, None].to(
            target_device)
        audio_to_overwrite &= audio_left_pad_mask

        if audio_to_overwrite.sum() != audio_features.shape[:-1].numel():
            # print()
            raise ValueError(
                f"The input provided to the model are wrong. The number of audio tokens is {torch.sum(special_audio_token_mask)} while"
                f" the number of audio given to the model is {num_audio_samples}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[audio_to_overwrite] = (
            audio_features.contiguous().reshape(-1, embed_dim).to(
                target_device, dtype=final_embedding.dtype))
        final_attention_mask |= audio_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(
            (final_attention_mask == 0), 1)

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

"""