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

from typing import Any, Iterable, List, Optional, Tuple, Union


import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

import transformers
from transformers import AutoModel, AutoModelForCausalLM, CONFIG_MAPPING, PretrainedConfig, ProcessorMixin, TensorType, BatchFeature
from transformers import logging
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)

from vllm.attention import AttentionMetadata
from vllm.config import AudioLanguageConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead


from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput

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
    inputs_embeds[mask] = audio_embeddings.view(-1,
                                                 audio_embeddings.shape[-1])




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
            text_config["model_type"] = (
                text_config["model_type"] if "model_type" in text_config else "llama"
            )
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
            self.vocab_size = self.text_config.vocab_size
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["llama"]()

        if isinstance(self.audio_config, dict):
            audio_config["model_type"] = (
                audio_config["model_type"]
                if "model_type" in audio_config
                else "wav2vec2"
            )
            self.audio_config = CONFIG_MAPPING[audio_config["model_type"]](
                **audio_config
            )
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
            audio_embeds, (0, 0, 0, self.stack_factor - T % self.stack_factor)
        )
        B, T, C = audio_embeds.shape
        audio_embeds = audio_embeds.view(
            B, T // self.stack_factor, C * self.stack_factor
        )
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
        self.weight = nn.Parameter(torch.full((hidden_size,), 0.4))
        # self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class GazelleProjector(ProjectionLayer):
    def __init__(self, config: GazelleConfig):
        self.hidden_dim = config.hidden_size
        super().__init__(config.stack_factor)
        self.ln_pre = RMSNorm(config.audio_config.hidden_size * self.stack_factor)
        self.linear_1 = nn.Linear(
            config.audio_config.hidden_size * self.stack_factor,
            self.hidden_dim,
            bias=False,
        )
        self.act = SwiGLU()
        self.linear_2 = nn.Linear(
            self.hidden_dim // 2, config.text_config.hidden_size, bias=False
        )
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
    def __init__(self, config: GazelleConfig, audio_language_config: AudioLanguageConfig, quant_config: Optional["QuantizationConfig"] = None):
        from vllm.model_executor.models.llama import LlamaModel
        super().__init__()
        self.config = config
        self.audio_language_config = audio_language_config
        if config.audio_model_id is not None:
            self.audio_tower = AutoModel.from_pretrained(config.audio_model_id)
        else:
            self.audio_tower = AutoModel.from_config(config.audio_config)

        self.multi_modal_projector = GazelleProjector(config)
        self.vocab_size = config.vocab_size
        self.quant_config = quant_config
        
        if config.text_model_id is not None:
            self.language_model = AutoModelForCausalLM.from_pretrained(
                config.text_model_id, attn_implementation=config._attn_implementation
            )
        else:
            self.language_model = LlamaModel(config.text_config, quant_config)
        self.unpadded_vocab_size = config.text_config.vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.text_config.hidden_size,
            org_num_embeddings=self.language_model.org_vocab_size)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()
        self.pad_token_id = (
            config.pad_token_id if config.pad_token_id is not None else -1
        )

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

    def _merge_input_ids_with_audio_features(
        self, audio_features, inputs_embeds, input_ids, attention_mask, labels
    ):
        num_audio_samples, num_audio_patches, embed_dim = audio_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(
            input_ids[:, -1] == torch.tensor(self.pad_token_id)
        )
        # 1. Create a mask to know where special image tokens are
        special_audio_token_mask = input_ids == self.config.audio_token_index
        num_special_image_tokens = torch.sum(special_audio_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (
            num_special_image_tokens.max() * (num_audio_patches - 1)
        ) + sequence_length
        batch_indices, non_audio_indices = torch.where(
            input_ids != self.config.audio_token_index
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged audio-text sequence.
        # `special_audio_token_mask` identifies audio tokens. Each audio token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each audio token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = (
            torch.cumsum((special_audio_token_mask * (num_audio_patches - 1) + 1), -1)
            - 1
        )
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:  # TODO FARZAD: I don't understand this
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]

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
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_audio_indices
        ]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
            batch_indices, non_audio_indices
        ]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[
                batch_indices, non_audio_indices
            ]

        # 5. Fill the embeddings corresponding to the audio. Anything that is still zeros needs filling
        audio_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        audio_positions = audio_to_overwrite.to(torch.int16).cumsum(-1) - 1
        audio_left_pad_mask = audio_positions >= nb_image_pad[:, None].to(target_device)
        audio_to_overwrite &= audio_left_pad_mask

        if audio_to_overwrite.sum() != audio_features.shape[:-1].numel():
            # print()
            raise ValueError(
                f"The input provided to the model are wrong. The number of audio tokens is {torch.sum(special_audio_token_mask)} while"
                f" the number of audio given to the model is {num_audio_samples}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[audio_to_overwrite] = (
            audio_features.contiguous()
            .reshape(-1, embed_dim)
            .to(target_device, dtype=final_embedding.dtype)
        )
        final_attention_mask |= audio_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(
            (final_attention_mask == 0), 1
        )

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

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
        #logger.info(f"input_ids {input_ids}")
        #logger.info(f"positions {positions}")
        #logger.info(f"kv_caches {[kv_cache.shape for kv_cache in kv_caches if kv_cache is not None]}")
        #logger.info(f"attn_metadata {attn_metadata}")
        #logger.info(f"audio_input {audio_input.shape if audio_input is not None else None}")
        if audio_input is not None:
            #if list(audio_input.shape[1:]) != [1, 3, 336, 336]:
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
                # TODO(xwjiang): Maybe port minimal CLIPVisionModel over.
                audio_features = self.audio_tower(audio_input).last_hidden_state
                #audio_outputs = audio_outputs.to(inputs_embeds.dtype)
                                                  
                #audio_features = audio_outputs.hidden_states[-1] ####                    
                # Copied from https://github.com/huggingface/transformers/blob/39c3c0a72af6fbda5614dde02ff236069bb79827/src/transformers/models/llava/modeling_llava.py#L421  # noqa               
            else:
                audio_features = audio_input
            logger.info(f"audio_features: {audio_features.shape}")
            audio_embeddings = self.multi_modal_projector(audio_features)
            logger.info(f"audio_embs: {audio_embeddings.shape}")
            logger.info(f"input_embs: {inputs_embeds.shape}")
            _merge_audio_embeddings(
                input_ids, inputs_embeds, audio_embeddings,
                self.audio_language_config.audio_token_id)
            logger.info(f"input_ids: {input_ids}")
            logger.info(f"input_embs: {inputs_embeds.shape}")
            input_ids = None
        else:
            inputs_embeds = None
        hidden_states = self.language_model(input_ids=input_ids,
                                            positions=positions,
                                            kv_caches=kv_caches,
                                            attn_metadata=attn_metadata,
                                            inputs_embeds=inputs_embeds)

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
    

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        audio_values=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.audio_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[
                    :, -(cache_length + input_ids.shape[1]) :
                ]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "audio_values": audio_values,
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
    


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
        #for key in params_dict.keys():
        #    logger.info(f"param: {key}")        
        for name, loaded_weight in weights:
            #logger.info(f"weight: {name}")
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



# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""
Processor class for Gazelle.
"""


class GazelleProcessor(ProcessorMixin):
    r"""
    Constructs a Gazelle processor which wraps a Gazelle image processor and a Gazelle tokenizer into a single processor.

    [`GazelleProcessor`] offers all the functionalities of [`Wav2Vec2Processor`] and [`LlamaTokenizerFast`]. See the
    [`~GazelleProcessor.__call__`] and [`~GazelleProcessor.decode`] for more information.

    Args:
        audio_processor ([`Wav2Vec2Processor`, `SeamlessM4TFeatureExtractor`], *optional*):
            The audio processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["audio_processor", "tokenizer"]
    audio_processor_class = (
        "Wav2Vec2Processor",
        "SeamlessM4TFeatureExtractor",
    )
    tokenizer_class = (
        "LlamaTokenizer",
        "LlamaTokenizerFast",
    )

    def __init__(self, audio_processor=None, tokenizer=None):
        super().__init__(audio_processor, tokenizer)

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        audio=None,
        text_padding: Union[bool, str, PaddingStrategy] = False,
        text_truncation: Union[bool, str, TruncationStrategy] = None,
        text_max_length=None,
        audio_padding: Union[bool, str, PaddingStrategy] = False,
        audio_truncation: Union[bool, str, TruncationStrategy] = None,
        audio_max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        sampling_rate: int = 16000,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audio (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                 The audio or batch of audios to be prepared. Each audio can be NumPy array or PyTorch tensor. In case of a
                NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels, and T the
                sample length of the audio.
            audio_padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            audio_max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            audio_truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
            sampling_rate (`int`, *optional*, defaults to 16000):
                Sampling rate of the input audio. We expect 16kHz audio. Don't change this value unless you know what
                you are doing.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **audio_values** -- Processed audio values to be fed to a model. Returned when `audios` is not `None`.
        """
        if audio is not None and len(audio) > 0:
            x = self.audio_processor(
                audio,
                return_tensors=return_tensors,
                sampling_rate=sampling_rate,
                padding=audio_padding,
                truncation=audio_truncation,
                max_length=audio_max_length,
            )
            audio_values = x.input_values  # features
        else:
            audio_values = None
        if text is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=text_padding,
                truncation=text_truncation,
                max_length=text_max_length,
            )
            return BatchFeature(data={**text_inputs, "audio_values": audio_values})
        else:
            return BatchFeature(data={"audio_values": audio_values})

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        audio_processor_input_names = self.audio_processor_class.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + audio_processor_input_names))
    

