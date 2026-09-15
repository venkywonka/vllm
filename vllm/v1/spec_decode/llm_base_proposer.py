# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

from vllm import envs
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    replace,
)
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.deepseek_eagle3 import Eagle3DeepseekV2ForCausalLM
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import current_stream
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import (
    empty_exponential_noise_like,
    sample_with_exponential_noise,
)
from vllm.v1.sample.sampler import _SAMPLING_EPS
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    compute_new_slot_mapping,
    copy_and_expand_eagle_inputs_kernel,
    eagle_prepare_inputs_padded_kernel,
    eagle_prepare_next_token_padded_kernel,
    eagle_step_update_slot_mapping_and_metadata,
    extend_all_queries_by_N,
    next_power_of_2,
)
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)

_DRAFT_ATTN_METADATA_LOG_INTERVAL = 16_384
_MISSING_DRAFT_ATTN_BUILDER_CONFIG = object()
_NO_DRAFT_CHAIN_REPLAY = object()
# A chain graph can retain additional shared-pool memory after the Stage 1
# PIECEWISE graphs have been profiled. Keep it fail-closed until that memory can
# be bounded without changing the KV cache capacity selected for the lane.
_DRAFT_CHAIN_CUDAGRAPH_ENABLED = False


@dataclass
class _DraftChainGraphEntry:
    graph: Any
    output: torch.Tensor
    num_speculative_tokens: int
    first_num_actual_tokens: int
    first_input_batch_size: int
    remaining_input_batch_size: int
    token_indices_to_sample: torch.Tensor
    num_rejected_tokens: torch.Tensor
    seq_lens: torch.Tensor
    block_table_tensor: torch.Tensor
    references: tuple[Any, ...]


class SpecDecodeBaseProposer:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        pass_hidden_states_to_model: bool,
        runner=None,
    ):
        self.vllm_config = vllm_config
        assert vllm_config.speculative_config is not None
        self.speculative_config = vllm_config.speculative_config
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method
        self.pass_hidden_states_to_model = pass_hidden_states_to_model
        self._share_mtp_indices = False

        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens

        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()
        self.inputs_embeds_size = self.draft_model_config.get_inputs_embeds_size()

        # DeepSeek V4 MTP consumes the target's pre-hc_head residual stream,
        # shape (T, hc_mult * hidden_size). Expand the hidden_states buffer
        # so target_hidden_states fits; detect DeepseekV4 via draft hf_config.
        draft_hf_config = self.draft_model_config.hf_config
        if hasattr(draft_hf_config, "compress_ratios") and hasattr(
            draft_hf_config, "hc_mult"
        ):
            self.hidden_size = self.hidden_size * draft_hf_config.hc_mult

        # Unifying eagle, draft model, and parallel drafting support.
        # DFlash always uses parallel drafting (all tokens in one pass),
        # but has an additional slot for the next_token_id (does not shift like EAGLE)
        self.parallel_drafting: bool = self.speculative_config.parallel_drafting
        self.extra_slots_per_request = (
            1 if not self.parallel_drafting else self.num_speculative_tokens
        )
        self.net_num_new_slots_per_request = self.extra_slots_per_request - (
            1 if (self.pass_hidden_states_to_model and self.method != "dflash") else 0
        )
        self.needs_extra_input_slots = self.net_num_new_slots_per_request > 0

        # When True, all draft steps reuse the same position as the
        # first step instead of advancing by one each iteration.
        # Used by draft models with Q-only attention that share KV
        # with the target and always predict from the same position.
        self.constant_draft_positions: bool = False

        self.parallel_drafting_token_id: int = 0
        self.parallel_drafting_hidden_state_tensor: torch.Tensor | None = None
        if self.parallel_drafting:
            self._init_parallel_drafting_params()
        self.use_local_argmax_reduction: bool = (
            self.speculative_config.use_local_argmax_reduction
        )
        self.use_fp64_gumbel = vllm_config.model_config.use_fp64_gumbel

        self.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_arange_np = np.arange(self.max_num_tokens, dtype=np.int32)

        # Can be specialized by methods like DFlash to reduce the limit
        self.max_query_tokens = self.max_num_tokens
        self.max_positions = self.max_num_tokens

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )

        self.draft_attn_groups: list[AttentionGroup] = []
        self.kv_cache_gid: int = -1
        self.eagle3_use_aux_hidden_state: bool = (
            self._get_eagle3_use_aux_hidden_state_from_config()
        )

        self.compilation_config = self.vllm_config.compilation_config

        # Cudagraph dispatcher for PIECEWISE-only dispatching in eagle.
        # Keys are initialized later via initialize_cudagraph_keys() called from
        # gpu_model_runner._check_and_update_cudagraph_mode after
        # adjust_cudagraph_sizes_for_spec_decode is called.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        # persistent buffers for cuda graph
        self.input_ids = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=device
        )
        # Use draft model's M-RoPE setting, not target model's
        # Draft models may be text-only even if target is multimodal
        self.uses_mrope = self.draft_model_config.uses_mrope
        self.uses_xdrope_dim = self.vllm_config.model_config.uses_xdrope_dim
        self.draft_uses_xdrope_dim = self.draft_model_config.uses_xdrope_dim
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = torch.zeros(
                (3, self.max_positions + 1), dtype=torch.int64, device=device
            )
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions = torch.zeros(
                (self.uses_xdrope_dim, self.max_positions + 1),
                dtype=torch.int64,
                device=device,
            )
        else:
            # RoPE need (max_num_tokens,)
            self.positions = torch.zeros(
                self.max_positions,
                dtype=torch.int64,
                device=device,
            )
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device
        )

        # Will be set when we initialize the attention backend
        self.block_size: int = -1

        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        max_num_slots_for_arange = max(self.max_batch_size + 1, self.max_num_tokens)
        self.arange = torch.arange(
            max_num_slots_for_arange, device=device, dtype=torch.int32
        )

        if self.needs_extra_input_slots:
            self._raise_if_padded_drafter_batch_disabled()
            self._warn_if_multimodal()
            self._raise_if_mrope()

        self.is_rejected_token_mask: torch.Tensor | None = None
        self.is_masked_token_mask: torch.Tensor | None = None
        if self.needs_extra_input_slots:
            # For draft models and parallel drafting, we need to keep track of
            # which tokens are rejected to update the slot mapping with padding slots.
            self.is_rejected_token_mask = torch.zeros(
                (self.max_num_tokens,), dtype=torch.bool, device=device
            )
            # For parallel drafting, we also need to keep track of which tokens
            # are parallel-padding tokens used to sample at later positions.
            # We populate this tensor even when using draft models for simplicity.
            self.is_masked_token_mask = torch.zeros(
                (self.max_num_tokens,), dtype=torch.bool, device=device
            )

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.inputs_embeds_size),
            dtype=self.dtype,
            device=device,
        )

        self.backup_next_token_ids = CpuGpuBuffer(
            self.max_batch_size,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )
        self._enable_probabilistic_draft_probs = (
            self.speculative_config.rejection_sample_method == "standard"
            and self.speculative_config.draft_sample_method == "probabilistic"
        )
        self._last_draft_probs: torch.Tensor | None = None

        self._slot_mapping_buffer = torch.zeros(
            self.max_positions,
            dtype=torch.int64,
            device=device,
        )

        # The FlashAttention drafting builder mostly reconstructs Python
        # containers around persistent tensors. Cache that topology only for
        # the narrowly verified EAGLE3 path; all other paths rebuild normally.
        self._draft_attn_metadata_cache: dict[
            tuple[Any, ...],
            tuple[list[object], dict[str, object]],
        ] = {}
        self._disabled_draft_attn_metadata_cache_keys: set[tuple[Any, ...]] = set()
        self._draft_attn_metadata_cache_group_key: tuple[Any, ...] | None = None
        self._draft_query_start_loc_cpu_cache: dict[int, torch.Tensor] = {}
        self.metadata_template_builds = 0
        self.metadata_template_hits = 0
        self.metadata_template_fallbacks = 0

        # Stage 2 captures the complete serial drafter chain. Entries bind the
        # runner's persistent sequence lengths and block table; every other
        # dynamic input is copied into proposer-owned storage before replay.
        self._draft_chain_graphs: dict[tuple[Any, ...], _DraftChainGraphEntry] = {}
        self._draft_chain_graph_capability_key: tuple[Any, ...] | None = None
        self._draft_chain_graph_pool = (
            current_platform.get_global_graph_pool()
            if _DRAFT_CHAIN_CUDAGRAPH_ENABLED
            else None
        )
        self._draft_chain_graph_is_profiling = False
        self.draft_chain_graph_captures = 0
        self.draft_chain_graph_replays = 0
        self.draft_chain_graph_fallbacks = 0
        self._draft_chain_forward_events = 0
        self._draft_chain_graph_stats_by_k: dict[int, dict[str, int]] = {}

        # Determine allowed attention backends once during initialization.
        self.allowed_attn_types: tuple | None = None
        if current_platform.is_rocm():
            from vllm.models.deepseek_v4.amd.rocm import (
                DeepseekV4ROCMAiterMLASparseMetadata,
                DeepseekV4ROCMAiterSparseSWAMetadata,
            )

            # MiniMax-M3 sparse (lightning-indexer) attention. The multi-step
            # drafting machinery is shared code at num_speculative_tokens>1.
            # this just opts the metadata into the ROCm allowlist.
            from vllm.models.minimax_m3.common.sparse_attention import (
                MiniMaxM3SparseMetadata,
            )
            from vllm.v1.attention.backends.mla.indexer import (
                DeepseekV32IndexerMetadata,
            )
            from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
                ROCMAiterMLASparseMetadata,
            )
            from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata

            rocm_types = [
                TritonAttentionMetadata,
                RocmAttentionMetadata,
                ROCMAiterMLASparseMetadata,
                DeepseekV4ROCMAiterMLASparseMetadata,
                DeepseekV4ROCMAiterSparseSWAMetadata,
                DeepseekV32IndexerMetadata,
                MiniMaxM3SparseMetadata,
            ]
            # ROCM_AITER_FA is an optional backend
            # We check is_enabled() here to avoid importing the backend module during
            # auto-discovery when VLLM_ROCM_USE_AITER=0, which would trigger aiter
            # import and JIT compilation warnings. Explicit backend selection via
            # attention_config still works because the backend module is loaded
            # directly when selected, not through this auto-discovery path.
            # Check if backend module exists to allow explicit selection
            if find_spec(
                AttentionBackendEnum.ROCM_AITER_FA.get_path(include_classname=False)
            ):
                from vllm.v1.attention.backends.rocm_aiter_fa import (
                    AiterFlashAttentionMetadata,
                )

                rocm_types.append(AiterFlashAttentionMetadata)

            # TRITON_MLA backend support for MLA models (e.g., DeepSeek)
            from vllm.model_executor.layers.attention.mla_attention import (
                MLACommonMetadata,
            )

            rocm_types.append(MLACommonMetadata)

            # FlexAttention backend support
            from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata

            rocm_types.append(FlexAttentionMetadata)

            self.allowed_attn_types = tuple(rocm_types)

    def _raise_if_padded_drafter_batch_disabled(self):
        if self.speculative_config.disable_padded_drafter_batch:
            raise NotImplementedError(
                "Speculative Decoding with draft models or parallel drafting only "
                "supports padded drafter batch. Please unset "
                "disable_padded_drafter_batch in the speculative_config."
            )

    def _warn_if_multimodal(self):
        if self.supports_mm_inputs:
            logger.warning(
                "Speculative Decoding with draft models or parallel drafting "
                "does not fully support multimodal models yet. "
                "Proceeding with text-only speculative decoding."
            )

    def _raise_if_mrope(self):
        if self.draft_model_config.uses_mrope:
            raise NotImplementedError(
                "Speculative Decoding with draft models or parallel drafting "
                "does not support M-RoPE yet"
            )

    def _init_parallel_drafting_params(self):
        # For parallel drafting, we need the token ID to use for masked slots
        # And for EAGLE + parallel drafting, we need the hidden state tensor to use
        # for those masked slots.

        model_hf_config = self.draft_model_config.hf_config
        # DFlash stores mask_token_id in dflash_config
        dflash_config = getattr(model_hf_config, "dflash_config", None)
        if dflash_config and "mask_token_id" in dflash_config:
            self.parallel_drafting_token_id = dflash_config["mask_token_id"]
        elif hasattr(model_hf_config, "pard_token"):
            self.parallel_drafting_token_id = model_hf_config.pard_token
        elif hasattr(model_hf_config, "ptd_token_id"):
            self.parallel_drafting_token_id = model_hf_config.ptd_token_id
        else:
            raise ValueError(
                "For parallel drafting, the draft model config must have "
                "`pard_token`, `ptd_token_id`, or "
                "`dflash_config.mask_token_id` specified in its config.json."
            )

        if self.pass_hidden_states_to_model:
            self.parallel_drafting_hidden_state_tensor = torch.empty(
                self.hidden_size, dtype=self.dtype, device=self.device
            )

    def _get_positions(self, num_tokens: int):
        if self.uses_mrope:
            return self.mrope_positions[:, :num_tokens]
        if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            return self.xdrope_positions[:, :num_tokens]
        return self.positions[:num_tokens]

    def _set_positions(self, num_tokens: int, positions: torch.Tensor):
        if self.uses_mrope:
            self.mrope_positions[:, :num_tokens] = positions
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions[:, :num_tokens] = positions
        else:
            # Convert M-RoPE positions if target model uses M-RoPE
            # but draft doesn't, For text inputs, all M-RoPE
            # dimensions are identical
            if self.vllm_config.model_config.uses_mrope:
                positions = positions[0]
            self.positions[:num_tokens] = positions

    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return slot_mapping dict for EAGLE layers.

        If slot_mapping is provided, copies it into the buffer first.
        """
        if slot_mapping is not None:
            num_actual = slot_mapping.shape[0]
            self._slot_mapping_buffer[:num_actual].copy_(slot_mapping)
            if num_tokens > num_actual:
                self._slot_mapping_buffer[num_actual:num_tokens].fill_(PADDING_SLOT_ID)

        view = self._slot_mapping_buffer[:num_tokens]
        return {name: view for name in self._draft_attn_layer_names}

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Initialize cudagraph dispatcher keys for the drafter.

        Only supports PIECEWISE cudagraphs (via mixed_mode).
        This should be called after adjust_cudagraph_sizes_for_spec_decode.
        """
        if (
            not self.speculative_config.enforce_eager
            and cudagraph_mode.mixed_mode()
            in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        ):
            eagle_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            eagle_cudagraph_mode = CUDAGraphMode.NONE

        self.cudagraph_dispatcher.initialize_cudagraph_keys(eagle_cudagraph_mode)

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy-sample draft tokens from hidden states."""
        if self.use_local_argmax_reduction:
            return self.model.get_top_tokens(hidden_states)
        return self.model.compute_logits(hidden_states).argmax(dim=-1)

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self._enable_probabilistic_draft_probs:
            return logits.argmax(dim=-1), None
        if sampling_metadata.all_greedy:
            return logits.argmax(dim=-1), None
        return compute_probs_and_sample_next_token(
            logits, sampling_metadata, self.use_fp64_gumbel
        )

    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self._enable_probabilistic_draft_probs or sampling_metadata.all_greedy:
            return self._greedy_sample(hidden_states), None
        logits = self.model.compute_logits(hidden_states)
        return self._sample_from_logits(logits, sampling_metadata)

    def take_last_draft_probs(self) -> torch.Tensor | None:
        return self._last_draft_probs

    def propose(
        self,
        num_speculative_tokens,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        self.num_speculative_tokens = num_speculative_tokens
        self._last_draft_probs = None
        batch_size = common_attn_metadata.batch_size()

        if self.method in ("eagle3", "dflash"):
            model = self.model
            if isinstance(model, BreakableCUDAGraphWrapper):
                model = model.unwrap()
            assert isinstance(
                model,
                (
                    Eagle3LlamaForCausalLM,
                    Eagle3DeepseekV2ForCausalLM,
                    DFlashQwen3ForCausalLM,
                ),
            )
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )
            assert target_hidden_states.shape[-1] == self.hidden_size

        num_tokens, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=token_indices_to_sample,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        )

        if self.method == "eagle3" and _DRAFT_CHAIN_CUDAGRAPH_ENABLED:
            draft_chain_output = self._try_replay_draft_chain_graph(
                num_speculative_tokens=num_speculative_tokens,
                num_tokens=num_tokens,
                token_indices_to_sample=token_indices_to_sample,
                common_attn_metadata=common_attn_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
            if draft_chain_output is not _NO_DRAFT_CHAIN_REPLAY:
                return cast(torch.Tensor, draft_chain_output)

        if self._can_cache_draft_attn_metadata(common_attn_metadata, draft_index=0):
            cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
                self._determine_batch_execution_and_padding(num_tokens)
            )
            if cudagraph_runtime_mode is CUDAGraphMode.PIECEWISE:
                per_group_attn_metadata, per_layer_attn_metadata = (
                    self._get_draft_attn_metadata(
                        common_attn_metadata,
                        draft_index=0,
                        phase="first",
                        input_batch_size=num_input_tokens,
                        cudagraph_runtime_mode=cudagraph_runtime_mode,
                        cache_is_supported=True,
                    )
                )
            else:
                self._record_metadata_template_event("fallback")
                per_group_attn_metadata, per_layer_attn_metadata = (
                    self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
                )
        else:
            if self.method == "eagle3":
                self._record_metadata_template_event("fallback")
            per_group_attn_metadata, per_layer_attn_metadata = (
                self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
            )
            cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
                self._determine_batch_execution_and_padding(num_tokens)
            )

        model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
            num_tokens, num_input_tokens, mm_embed_inputs
        )
        # Step 0 of index_share_for_mtp_iteration: let the MTP layer
        # compute its own indices (skip_topk=False) so subsequent steps
        # can reuse them.
        if self._share_mtp_indices and hasattr(self.model.model, "set_skip_topk"):
            self.model.model.set_skip_topk(False)

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                slot_mapping_size, common_attn_metadata.slot_mapping
            ),
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

        # After step 0: switch to reuse mode so steps 1+ skip the indexer
        # and read the indices that step 0 just wrote into the shared buffer.
        if self._share_mtp_indices and hasattr(self.model.model, "set_skip_topk"):
            self.model.model.set_skip_topk(True)

        sample_hidden_states = last_hidden_states[token_indices_to_sample]

        # No draft tokens requested (e.g. Dynamic SD decided K=0).
        # The prefill forward pass above already ran to keep the drafter
        # KV cache in sync, so just return an empty tensor.
        if self.num_speculative_tokens == 0:
            return torch.empty(
                batch_size,
                0,
                device=sample_hidden_states.device,
                dtype=torch.int64,
            )

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            draft_token_ids, draft_probs = self._sample_draft_tokens(
                sample_hidden_states, sampling_metadata
            )
            if draft_probs is not None:
                self._last_draft_probs = draft_probs.view(
                    -1, self.num_speculative_tokens, draft_probs.shape[-1]
                ).contiguous()
            return draft_token_ids.view(-1, self.num_speculative_tokens)

        if self.uses_mrope:
            positions = self.mrope_positions[:, token_indices_to_sample]
        else:
            positions = self.positions[token_indices_to_sample]
        hidden_states = hidden_states[token_indices_to_sample]

        if self.constant_draft_positions:
            # Write the sampling positions into the front of the
            # positions buffer so that subsequent loop iterations
            # (which read via _get_positions) use the correct values.
            self.positions[:batch_size] = positions

        draft_token_ids, draft_probs = self._sample_draft_tokens(
            sample_hidden_states, sampling_metadata
        )
        draft_probs_list = None if draft_probs is None else [draft_probs]

        if self.allowed_attn_types is not None:
            for group_md in per_group_attn_metadata:
                if not isinstance(group_md, self.allowed_attn_types):
                    raise ValueError(
                        f"Unsupported attention metadata type for speculative "
                        "decoding with num_speculative_tokens > 1: "
                        f"{type(group_md)}. Supported types are: "
                        f"{self.allowed_attn_types}"
                    )

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
            self._determine_batch_execution_and_padding(batch_size)
        )

        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
        cache_remaining_metadata = (
            cudagraph_runtime_mode is CUDAGraphMode.PIECEWISE
            and self._can_cache_draft_attn_metadata(common_attn_metadata, draft_index=1)
        )
        if cache_remaining_metadata:
            common_attn_metadata.query_start_loc_cpu = (
                self._get_draft_query_start_loc_cpu(batch_size)
            )
        else:
            common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
                self.token_arange_np[: batch_size + 1]
            ).clone()

        # In padded drafter batch, we need to adjust the sequence lengths
        # to remove the "padding" (i.e. rejected tokens).
        # Only apply this adjustment when we have rejected tokens
        # (i.e., not the first proposal).
        if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
            # Invalidate the CPU-side shadows to avoid H<>D sync.
            common_attn_metadata._seq_lens_cpu = None
            common_attn_metadata._num_computed_tokens_cpu = None

        block_size = self.block_size
        assert block_size > 0, "block_size has not been initialized."
        for token_index in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()

            if not self.constant_draft_positions:
                positions = self._update_positions_dependent_metadata(
                    positions,
                    common_attn_metadata,
                    batch_size,
                    input_batch_size,
                    block_size,
                )

            # Rebuild attention metadata. When draft positions are constant
            # (e.g. Gemma4 MTP), common_attn_metadata is invariant across
            # loop iterations so we build once and reuse.
            if not self.constant_draft_positions or token_index == 0:
                if cache_remaining_metadata:
                    _, per_layer_attn_metadata = self._get_draft_attn_metadata(
                        common_attn_metadata,
                        draft_index=token_index + 1,
                        phase="remaining",
                        input_batch_size=input_batch_size,
                        cudagraph_runtime_mode=cudagraph_runtime_mode,
                        cache_is_supported=True,
                    )
                else:
                    if self.method == "eagle3":
                        self._record_metadata_template_event("fallback")
                    _, per_layer_attn_metadata = (
                        self.build_per_group_and_layer_attn_metadata(
                            common_attn_metadata, draft_index=token_index + 1
                        )
                    )

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self.hidden_states[:batch_size] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

                input_ids = None
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None

            # Run the model.
            model_kwargs = {
                "input_ids": input_ids,
                "positions": self._get_positions(input_batch_size),
                "inputs_embeds": inputs_embeds,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]

            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(input_batch_size),
            ):
                ret_hidden_states = self.model(**model_kwargs)
                if not self.model_returns_tuple():
                    last_hidden_states = ret_hidden_states
                    hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states

            hidden_states = hidden_states[:batch_size]
            draft_token_ids, draft_probs = self._sample_draft_tokens(
                last_hidden_states[:batch_size], sampling_metadata
            )
            if draft_probs is not None:
                assert draft_probs_list is not None
                draft_probs_list.append(draft_probs)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if draft_probs_list is not None:
            self._last_draft_probs = torch.stack(draft_probs_list, dim=1).contiguous()
        return draft_token_ids

    def _update_positions_dependent_metadata(
        self,
        positions: torch.Tensor,
        common_attn_metadata,
        batch_size: int,
        input_batch_size: int,
        block_size: int,
    ) -> torch.Tensor:
        """Update positions, slot mappings, and sequence metadata for the
        next draft step. Returns the updated positions tensor."""
        positions_1d = positions[0] if self.uses_mrope else positions
        if self.uses_mrope:
            out_pos = self.mrope_positions[0, :batch_size]
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            out_pos = self.xdrope_positions[0, :batch_size]
        else:
            out_pos = self.positions[:batch_size]
        eagle_step_update_slot_mapping_and_metadata(
            positions_1d=positions_1d,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            seq_lens=common_attn_metadata.seq_lens,
            block_size=block_size,
            max_model_len=self.max_model_len,
            out_clamped_positions=out_pos,
            out_slot_mapping=self._slot_mapping_buffer[:input_batch_size],
            input_batch_size=input_batch_size,
        )
        common_attn_metadata.slot_mapping = self._slot_mapping_buffer[:batch_size]
        if self.uses_mrope:
            self.mrope_positions[1:, :batch_size] = self.mrope_positions[0, :batch_size]
            positions = self.mrope_positions[:, :batch_size]
        elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
            self.xdrope_positions[1:, :batch_size] = self.xdrope_positions[
                0, :batch_size
            ]
            positions = self.xdrope_positions[0, :batch_size]
        else:
            positions = self.positions[:batch_size]
        common_attn_metadata.max_seq_len = min(
            common_attn_metadata.max_seq_len + 1,
            self.max_model_len,
        )

        if common_attn_metadata._seq_lens_cpu is not None:
            common_attn_metadata._seq_lens_cpu += 1
        if common_attn_metadata._num_computed_tokens_cpu is not None:
            common_attn_metadata._num_computed_tokens_cpu += 1
        if common_attn_metadata.seq_lens_cpu_upper_bound is not None:
            common_attn_metadata.seq_lens_cpu_upper_bound += 1

        return positions

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        if not self.needs_extra_input_slots:
            # Default EAGLE pathway: no reshaping of input tensors needed.
            # Simply rotate the input ids and leave the positions unchanged,
            # Inserting the next token ids at the last slot in each request.
            if token_indices_to_sample is None:
                token_indices_to_sample = cad.query_start_loc[1:] - 1

            num_tokens = target_token_ids.shape[0]
            # Shift the input ids by one token.
            # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
            self.input_ids[: num_tokens - 1] = target_token_ids[1:]
            # Replace the last token with the next token.
            # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
            self.input_ids[token_indices_to_sample] = next_token_ids

            # copy inputs to buffer for cudagraph
            if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim == 0:
                target_positions = target_positions[0]
            self._set_positions(num_tokens, target_positions)

            self.hidden_states[:num_tokens] = target_hidden_states

            return num_tokens, token_indices_to_sample, cad
        else:
            assert self.is_rejected_token_mask is not None
            assert self.is_masked_token_mask is not None
            # 1.
            # Call a custom triton kernel to copy input_ids and positions
            # into the correct slots in the preallocated buffers self.input_ids,
            # self.positions.
            batch_size = cad.batch_size()
            # Since we might have to copy a lot of data for prefills, we select the
            # block size based on the max query length and limit to max 256 slots/block.
            max_num_tokens_per_request = (
                cad.max_query_len + self.net_num_new_slots_per_request
            )
            BLOCK_SIZE_TOKENS = min(256, next_power_of_2(max_num_tokens_per_request))
            num_blocks = (
                max_num_tokens_per_request + BLOCK_SIZE_TOKENS - 1
            ) // BLOCK_SIZE_TOKENS
            total_num_input_tokens = target_token_ids.shape[0]
            total_num_output_tokens = total_num_input_tokens + (
                self.net_num_new_slots_per_request * batch_size
            )

            token_indices_to_sample = torch.empty(
                batch_size * self.extra_slots_per_request,
                dtype=torch.int32,
                device=self.device,
            )

            # Destination indices to write target_hidden_states into drafting buffer.
            out_hidden_state_mapping = torch.empty(
                total_num_input_tokens, dtype=torch.int32, device=self.device
            )

            # Kernel grid: one program per request (row)
            grid = (batch_size, num_blocks)
            query_start_loc = cad.query_start_loc
            query_end_loc = cad.query_start_loc[1:] - 1
            if num_rejected_tokens_gpu is not None:
                query_end_loc = query_end_loc - num_rejected_tokens_gpu

            copy_and_expand_eagle_inputs_kernel[grid](
                # (Padded) Inputs from the target model
                target_token_ids_ptr=target_token_ids,
                target_positions_ptr=target_positions,
                next_token_ids_ptr=next_token_ids,  # sampled tokens, one per request
                # Outputs to the drafting buffers
                out_input_ids_ptr=self.input_ids,
                out_positions_ptr=self.positions,  # Doesn't support mrope for now
                out_is_rejected_token_mask_ptr=self.is_rejected_token_mask,
                out_is_masked_token_mask_ptr=self.is_masked_token_mask,
                out_new_token_indices_ptr=token_indices_to_sample,
                out_hidden_state_mapping_ptr=out_hidden_state_mapping,
                # Input metadata
                query_start_loc_ptr=query_start_loc,
                query_end_loc_ptr=query_end_loc,
                padding_token_id=0,
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                # Sizing info
                # Note that we can deduce batch_size for free from the grid size
                total_input_tokens=total_num_input_tokens,
                num_padding_slots_per_request=self.extra_slots_per_request,
                shift_input_ids=self.pass_hidden_states_to_model,
                BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
            )
            if self.pass_hidden_states_to_model:
                assert self.parallel_drafting_hidden_state_tensor is not None
                self.hidden_states[out_hidden_state_mapping] = target_hidden_states
                # Use torch.where to avoid DtoH sync from boolean indexing
                mask = self.is_masked_token_mask[:total_num_output_tokens]
                torch.where(
                    mask.unsqueeze(1),
                    self.parallel_drafting_hidden_state_tensor,
                    self.hidden_states[:total_num_output_tokens],
                    out=self.hidden_states[:total_num_output_tokens],
                )

            # 2.
            # Recompute the slot mapping based on the new positions and
            # rejection mask.
            assert self.block_size > 0, "block_size has not been initialized."
            new_slot_mapping = compute_new_slot_mapping(
                cad=cad,
                new_positions=self.positions[:total_num_output_tokens],
                is_rejected_token_mask=self.is_rejected_token_mask[
                    :total_num_output_tokens
                ],
                block_size=self.block_size,
                num_new_tokens=self.net_num_new_slots_per_request,
                max_model_len=self.max_model_len,
            )

            # 3. Update the common attention metadata with the new (meta)data
            new_cad = extend_all_queries_by_N(
                cad,
                N=self.net_num_new_slots_per_request,
                arange=self.arange,
                new_slot_mapping=new_slot_mapping,
            )

            return total_num_output_tokens, token_indices_to_sample, new_cad

    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        model_kwargs = {
            "input_ids": input_ids,
            "positions": self._get_positions(num_input_tokens),
            "inputs_embeds": inputs_embeds,
        }
        if self.pass_hidden_states_to_model:
            model_kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]

        return model_kwargs, num_input_tokens

    def build_per_group_and_layer_attn_metadata(
        self, common_attn_metadata: CommonAttentionMetadata, draft_index: int = 0
    ) -> tuple[list[object], dict[str, object]]:
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=common_attn_metadata, draft_index=draft_index
            )
            per_group_attn_metadata.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_group_attn_metadata, per_layer_attn_metadata

    def _get_draft_query_start_loc_cpu(self, batch_size: int) -> torch.Tensor:
        query_start_loc_cpu = self._draft_query_start_loc_cpu_cache.get(batch_size)
        if query_start_loc_cpu is None:
            query_start_loc_cpu = torch.from_numpy(
                self.token_arange_np[: batch_size + 1]
            ).clone()
            self._draft_query_start_loc_cpu_cache[batch_size] = query_start_loc_cpu
        return query_start_loc_cpu

    def _initialize_draft_attn_metadata_cache_capability(self) -> None:
        """Prove the static part of the narrowly scoped cache contract."""
        self._draft_attn_metadata_cache_group_key = None
        self._draft_attn_metadata_cache.clear()
        self._disabled_draft_attn_metadata_cache_keys.clear()
        if (
            self.method != "eagle3"
            or self.parallel_drafting
            or self.constant_draft_positions
            or self.needs_extra_input_slots
            or self.supports_mm_inputs
            or self.uses_mrope
            or (self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0)
            or self.vllm_config.parallel_config.data_parallel_size != 1
            or self.speculative_config.disable_padded_drafter_batch
            or len(self.draft_attn_groups) != 1
        ):
            return

        # Keep FlashAttention an optional dependency of this generic proposer.
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionBackend,
            FlashAttentionMetadataBuilder,
        )

        model = self.model
        if isinstance(model, BreakableCUDAGraphWrapper):
            model = model.unwrap()
        if not isinstance(model, Eagle3LlamaForCausalLM):
            return

        attn_group = self.draft_attn_groups[0]
        builder = attn_group.get_metadata_builder()
        flash_attention_supported = (
            attn_group.backend is FlashAttentionBackend
            and type(builder) is FlashAttentionMetadataBuilder
            and builder.dcp_world_size == 1
        )
        triton_attention_supported = (
            attn_group.backend is TritonAttentionBackend
            and type(builder) is TritonAttentionMetadataBuilder
            and builder.vllm_config.parallel_config.decode_context_parallel_size == 1
        )
        if (
            not (flash_attention_supported or triton_attention_supported)
            or not attn_group.layer_names
            or tuple(attn_group.layer_names) != tuple(builder.layer_names)
            or set(attn_group.layer_names) != self._draft_attn_layer_names
        ):
            return

        # Keep the group, backend, and builder identities in the per-forward key.
        # Builder capture configuration is read into each stability key below.
        self._draft_attn_metadata_cache_group_key = (
            id(attn_group),
            attn_group.backend,
            id(builder),
        )

    def _initialize_draft_chain_graph_capability(self) -> None:
        """Prove the static contract for complete drafter-chain capture.

        Unlike the Stage 1 metadata cache, chain capture intentionally admits
        only the exact Triton pair. Triton accepts ``max_seqlen_k`` but does not
        use it in launch selection, kernel arguments, or masking; FlashAttention
        does use the host scalar and therefore keeps the Stage 1 path.
        """
        self._draft_chain_graph_capability_key = None
        self.clear_draft_chain_graphs(reset_stats=False)
        if not _DRAFT_CHAIN_CUDAGRAPH_ENABLED:
            return
        parallel_config = self.vllm_config.parallel_config
        if (
            self._draft_attn_metadata_cache_group_key is None
            or self.method != "eagle3"
            or self.parallel_drafting
            or self.constant_draft_positions
            or self.needs_extra_input_slots
            or self.supports_mm_inputs
            or self.uses_mrope
            or (self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0)
            or self.speculative_config.disable_padded_drafter_batch
            or self._enable_probabilistic_draft_probs
            or self._share_mtp_indices
            or self.max_batch_size != 1
            or parallel_config.data_parallel_size != 1
            or parallel_config.pipeline_parallel_size != 1
            or self.vllm_config.lora_config is not None
            or len(self.draft_attn_groups) != 1
        ):
            return

        model = self.model
        if isinstance(model, BreakableCUDAGraphWrapper):
            model = model.unwrap()
        if not isinstance(model, Eagle3LlamaForCausalLM):
            return

        attn_group = self.draft_attn_groups[0]
        builder = attn_group.get_metadata_builder()
        if (
            attn_group.backend is not TritonAttentionBackend
            or type(builder) is not TritonAttentionMetadataBuilder
            or builder.vllm_config.parallel_config.decode_context_parallel_size != 1
            or not attn_group.layer_names
            or tuple(attn_group.layer_names) != tuple(builder.layer_names)
            or set(attn_group.layer_names) != self._draft_attn_layer_names
        ):
            return

        self._draft_chain_graph_capability_key = (
            id(attn_group),
            attn_group.backend,
            id(builder),
            self.use_local_argmax_reduction,
            self.pass_hidden_states_to_model,
        )

    def _can_cache_draft_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> bool:
        if (
            self._draft_attn_metadata_cache_group_key is None
            or draft_index < 0
            or draft_index >= max(self.num_speculative_tokens, 1)
        ):
            return False

        if (
            common_attn_metadata.causal is not True
            or common_attn_metadata.mm_req_doc_ranges is not None
            or common_attn_metadata.logits_indices_padded is not None
            or common_attn_metadata.num_logits_indices is not None
            or common_attn_metadata.encoder_seq_lens is not None
            or common_attn_metadata.encoder_seq_lens_cpu is not None
            or common_attn_metadata.dcp_local_seq_lens is not None
            or common_attn_metadata.dcp_local_seq_lens_cpu is not None
        ):
            return False

        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        if not all(
            isinstance(tensor, torch.Tensor)
            for tensor in (query_start_loc, seq_lens, block_table, slot_mapping)
        ):
            return False

        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        return (
            num_reqs > 0
            and num_actual_tokens > 0
            and common_attn_metadata.max_query_len > 0
            and common_attn_metadata.max_seq_len > 0
            and query_start_loc.ndim == 1
            and query_start_loc.shape[0] == num_reqs + 1
            and seq_lens.ndim == 1
            and seq_lens.shape[0] == num_reqs
            and block_table.ndim == 2
            and block_table.shape[0] >= num_reqs
            and slot_mapping.ndim == 1
            and slot_mapping.shape[0] >= num_actual_tokens
        )

    def _draft_attn_metadata_cache_key_for(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
        phase: str,
        input_batch_size: int,
        cudagraph_runtime_mode: CUDAGraphMode,
    ) -> tuple[Any, ...]:
        builder = self.draft_attn_groups[0].get_metadata_builder()
        return (
            self._draft_attn_metadata_cache_group_key,
            getattr(
                builder,
                "use_full_cuda_graph",
                _MISSING_DRAFT_ATTN_BUILDER_CONFIG,
            ),
            getattr(
                builder,
                "max_cudagraph_size",
                _MISSING_DRAFT_ATTN_BUILDER_CONFIG,
            ),
            getattr(
                builder,
                "max_num_splits",
                _MISSING_DRAFT_ATTN_BUILDER_CONFIG,
            ),
            envs.VLLM_BATCH_INVARIANT,
            self.num_speculative_tokens,
            phase,
            draft_index,
            cudagraph_runtime_mode,
            input_batch_size,
            common_attn_metadata.num_reqs,
            common_attn_metadata.batch_size(),
            common_attn_metadata.num_actual_tokens,
            common_attn_metadata.max_query_len,
            common_attn_metadata.causal,
            common_attn_metadata.query_start_loc.shape,
            common_attn_metadata.seq_lens.shape,
            common_attn_metadata.block_table_tensor.shape,
            common_attn_metadata.slot_mapping.shape,
        )

    @staticmethod
    def _draft_attn_tensor_binding_matches(cached: object, current: object) -> bool:
        if not isinstance(cached, torch.Tensor) or not isinstance(
            current, torch.Tensor
        ):
            return False
        return (
            cached.data_ptr() == current.data_ptr()
            and cached.storage_offset() == current.storage_offset()
            and cached.shape == current.shape
            and cached.stride() == current.stride()
            and cached.dtype == current.dtype
            and cached.layout == current.layout
            and cached.device == current.device
        )

    @classmethod
    def _draft_attn_metadata_bindings_match(
        cls,
        cached: tuple[list[object], dict[str, object]],
        common_attn_metadata: CommonAttentionMetadata,
    ) -> bool:
        attn_metadata = cast(Any, cached[0][0])
        return (
            cls._draft_attn_tensor_binding_matches(
                attn_metadata.query_start_loc,
                common_attn_metadata.query_start_loc,
            )
            and cls._draft_attn_tensor_binding_matches(
                attn_metadata.seq_lens,
                common_attn_metadata.seq_lens,
            )
            and cls._draft_attn_tensor_binding_matches(
                attn_metadata.block_table,
                common_attn_metadata.block_table_tensor,
            )
            and cls._draft_attn_tensor_binding_matches(
                attn_metadata.slot_mapping,
                common_attn_metadata.slot_mapping,
            )
        )

    @classmethod
    def _draft_attn_metadata_matches_common(
        cls,
        cached: tuple[list[object], dict[str, object]],
        common_attn_metadata: CommonAttentionMetadata,
        attn_group: AttentionGroup,
    ) -> bool:
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionBackend,
            FlashAttentionMetadata,
            FlashAttentionMetadataBuilder,
        )

        per_group_attn_metadata, per_layer_attn_metadata = cached
        if len(per_group_attn_metadata) != 1 or set(per_layer_attn_metadata) != set(
            attn_group.layer_names
        ):
            return False

        attn_metadata = cast(Any, per_group_attn_metadata[0])
        if any(
            per_layer_attn_metadata[layer_name] is not attn_metadata
            for layer_name in attn_group.layer_names
        ):
            return False

        builder = attn_group.get_metadata_builder()
        flash_metadata_supported = (
            attn_group.backend is FlashAttentionBackend
            and type(builder) is FlashAttentionMetadataBuilder
            and type(attn_metadata) is FlashAttentionMetadata
        )
        triton_metadata_supported = (
            attn_group.backend is TritonAttentionBackend
            and type(builder) is TritonAttentionMetadataBuilder
            and type(attn_metadata) is TritonAttentionMetadata
        )
        if not (flash_metadata_supported or triton_metadata_supported):
            return False

        common_metadata_matches = (
            attn_metadata.num_actual_tokens == common_attn_metadata.num_actual_tokens
            and attn_metadata.max_query_len == common_attn_metadata.max_query_len
            and attn_metadata.max_seq_len == common_attn_metadata.max_seq_len
            and cls._draft_attn_metadata_bindings_match(cached, common_attn_metadata)
            and attn_metadata.use_cascade is False
            and attn_metadata.common_prefix_len == 0
            and attn_metadata.cu_prefix_query_lens is None
            and attn_metadata.prefix_kv_lens is None
            and attn_metadata.suffix_kv_lens is None
            and attn_metadata.scheduler_metadata is None
            and attn_metadata.prefix_scheduler_metadata is None
            and attn_metadata.causal is common_attn_metadata.causal
            and attn_metadata.mm_prefix_range_tensor is None
        )
        if not common_metadata_matches:
            return False

        if flash_metadata_supported:
            flash_metadata = cast(FlashAttentionMetadata, attn_metadata)
            return (
                flash_metadata.max_dcp_context_kv_len == 0
                and flash_metadata.dcp_context_kv_lens is None
            )
        triton_metadata = cast(TritonAttentionMetadata, attn_metadata)
        triton_builder = cast(TritonAttentionMetadataBuilder, builder)
        return (
            triton_metadata.seq_threshold_3D == triton_builder.seq_threshold_3D
            and triton_metadata.num_par_softmax_segments
            == triton_builder.num_par_softmax_segments
            and cls._draft_attn_tensor_binding_matches(
                triton_metadata.softmax_segm_output,
                triton_builder.softmax_segm_output,
            )
            and cls._draft_attn_tensor_binding_matches(
                triton_metadata.softmax_segm_max,
                triton_builder.softmax_segm_max,
            )
            and cls._draft_attn_tensor_binding_matches(
                triton_metadata.softmax_segm_expsum,
                triton_builder.softmax_segm_expsum,
            )
            and triton_metadata.mm_prefix_range is None
        )

    def _record_metadata_template_event(self, event: str) -> None:
        if event == "build":
            self.metadata_template_builds += 1
            event_total = self.metadata_template_builds
        elif event == "hit":
            self.metadata_template_hits += 1
            event_total = self.metadata_template_hits
        else:
            assert event == "fallback"
            self.metadata_template_fallbacks += 1
            event_total = self.metadata_template_fallbacks

        event_count = (
            self.metadata_template_builds
            + self.metadata_template_hits
            + self.metadata_template_fallbacks
        )
        snapshot = None
        if event_total == 1:
            snapshot = f"first-{event}"
        elif event_count % _DRAFT_ATTN_METADATA_LOG_INTERVAL == 0:
            snapshot = "periodic"
        if snapshot is not None:
            self.log_draft_attn_metadata_cache_stats(snapshot)

    def log_draft_attn_metadata_cache_stats(self, snapshot: str) -> None:
        logger.info(
            "EAGLE3 draft attention metadata cache: "
            "worker snapshot=%s, builds=%d, hits=%d, fallbacks=%d, entries=%d",
            snapshot,
            self.metadata_template_builds,
            self.metadata_template_hits,
            self.metadata_template_fallbacks,
            len(self._draft_attn_metadata_cache),
        )

    def get_draft_attn_metadata_cache_stats(self) -> dict[str, Any]:
        return {
            "metadata_template_builds": self.metadata_template_builds,
            "metadata_template_hits": self.metadata_template_hits,
            "metadata_template_fallbacks": self.metadata_template_fallbacks,
            "metadata_template_entries": len(self._draft_attn_metadata_cache),
        }

    def _record_draft_chain_graph_event(
        self,
        event: str,
        num_drafter_forwards: int = 0,
        num_speculative_tokens: int | None = None,
    ) -> None:
        k = (
            self.num_speculative_tokens
            if num_speculative_tokens is None
            else num_speculative_tokens
        )
        stats = self._draft_chain_graph_stats_by_k.setdefault(
            k,
            {
                "captures": 0,
                "replays": 0,
                "fallbacks": 0,
                "forward_events": 0,
            },
        )
        if event == "capture":
            self.draft_chain_graph_captures += 1
            stats["captures"] += 1
            event_total = stats["captures"]
        elif event == "replay":
            self.draft_chain_graph_replays += 1
            stats["replays"] += 1
            event_total = stats["replays"]
        else:
            assert event == "fallback"
            self.draft_chain_graph_fallbacks += 1
            stats["fallbacks"] += 1
            event_total = stats["fallbacks"]

        prior_forward_events = stats["forward_events"]
        stats["forward_events"] += num_drafter_forwards
        self._draft_chain_forward_events += num_drafter_forwards
        snapshot = None
        if event_total == 1:
            snapshot = f"first-{event}"
        elif (
            prior_forward_events // _DRAFT_ATTN_METADATA_LOG_INTERVAL
            < stats["forward_events"] // _DRAFT_ATTN_METADATA_LOG_INTERVAL
        ):
            snapshot = "periodic"
        if snapshot is not None:
            self.log_draft_chain_graph_stats(snapshot, k)

    def log_draft_chain_graph_stats(
        self, snapshot: str, num_speculative_tokens: int | None = None
    ) -> None:
        if num_speculative_tokens is None:
            ks = sorted(self._draft_chain_graph_stats_by_k)
            if not ks:
                ks = [self.num_speculative_tokens]
        else:
            ks = [num_speculative_tokens]

        for k in ks:
            stats = self._draft_chain_graph_stats_by_k.get(k, {})
            entries = sum(
                getattr(entry, "num_speculative_tokens", None) == k
                for entry in self._draft_chain_graphs.values()
            )
            logger.info(
                "EAGLE3 drafter chain CUDA graph: "
                "worker snapshot=%s, k=%d, enabled=%s, captures=%d, "
                "replays=%d, fallbacks=%d, entries=%d",
                snapshot,
                k,
                _DRAFT_CHAIN_CUDAGRAPH_ENABLED,
                stats.get("captures", 0),
                stats.get("replays", 0),
                stats.get("fallbacks", 0),
                entries,
            )

    def get_draft_chain_graph_stats(
        self, num_speculative_tokens: int | None = None
    ) -> dict[str, int]:
        if num_speculative_tokens is not None:
            stats = self._draft_chain_graph_stats_by_k.get(num_speculative_tokens, {})
            entries = sum(
                getattr(entry, "num_speculative_tokens", None) == num_speculative_tokens
                for entry in self._draft_chain_graphs.values()
            )
            return {
                "draft_chain_graph_captures": stats.get("captures", 0),
                "draft_chain_graph_replays": stats.get("replays", 0),
                "draft_chain_graph_fallbacks": stats.get("fallbacks", 0),
                "draft_chain_graph_entries": entries,
                "draft_chain_forward_events": stats.get("forward_events", 0),
            }
        return {
            "draft_chain_graph_captures": self.draft_chain_graph_captures,
            "draft_chain_graph_replays": self.draft_chain_graph_replays,
            "draft_chain_graph_fallbacks": self.draft_chain_graph_fallbacks,
            "draft_chain_graph_entries": len(self._draft_chain_graphs),
            "draft_chain_forward_events": self._draft_chain_forward_events,
        }

    def clear_draft_chain_graphs(self, reset_stats: bool = False) -> None:
        """Drop graphs before their bound KV cache or pool is released."""
        if hasattr(self, "_draft_chain_graphs"):
            self._draft_chain_graphs.clear()
        if reset_stats:
            self.draft_chain_graph_captures = 0
            self.draft_chain_graph_replays = 0
            self.draft_chain_graph_fallbacks = 0
            self._draft_chain_forward_events = 0
            self._draft_chain_graph_stats_by_k.clear()

    def set_draft_chain_graph_pool(
        self, graph_pool: Any, *, is_profiling: bool = False
    ) -> None:
        assert not self._draft_chain_graphs, (
            "Draft chain graphs must be cleared before changing graph pools."
        )
        self._draft_chain_graph_pool = graph_pool
        self._draft_chain_graph_is_profiling = is_profiling

    @staticmethod
    def _draft_chain_tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
        return (
            tensor.shape,
            tensor.stride(),
            tensor.dtype,
            tensor.layout,
            tensor.device,
        )

    def _draft_chain_graph_key_for(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        num_speculative_tokens: int,
        first_num_actual_tokens: int,
        first_input_batch_size: int,
        remaining_input_batch_size: int,
    ) -> tuple[Any, ...]:
        return (
            self._draft_chain_graph_capability_key,
            num_speculative_tokens,
            first_num_actual_tokens,
            first_input_batch_size,
            remaining_input_batch_size,
            self.block_size,
            self.max_model_len,
            self._draft_chain_tensor_signature(common_attn_metadata.seq_lens),
            self._draft_chain_tensor_signature(common_attn_metadata.block_table_tensor),
        )

    def _can_use_draft_chain_graph(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        num_speculative_tokens: int,
        first_num_actual_tokens: int,
        token_indices_to_sample: torch.Tensor | None,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[tuple[Any, ...], int, int] | None:
        max_capture_size = self.compilation_config.max_cudagraph_capture_size
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        if (
            self._draft_chain_graph_capability_key is None
            or self.speculative_config.enforce_eager
            or not self.compilation_config.cudagraph_mode.has_full_cudagraphs()
            or num_speculative_tokens < 2
            or first_num_actual_tokens != num_speculative_tokens + 1
            or max_capture_size is None
            or first_num_actual_tokens > max_capture_size
            or token_indices_to_sample is None
            or token_indices_to_sample.numel() != 1
            or mm_embed_inputs is not None
            or self._share_mtp_indices
            or self._enable_probabilistic_draft_probs
            or common_attn_metadata.num_reqs != 1
            or common_attn_metadata.batch_size() != 1
            or common_attn_metadata.num_actual_tokens != first_num_actual_tokens
            or common_attn_metadata.max_query_len != first_num_actual_tokens
            or query_start_loc_cpu.device.type != "cpu"
            or query_start_loc_cpu.shape != (2,)
            or query_start_loc_cpu.tolist() != [0, first_num_actual_tokens]
            or not self._can_cache_draft_attn_metadata(
                common_attn_metadata, draft_index=0
            )
        ):
            return None

        first_mode, first_input_batch_size, first_tokens_across_dp = (
            self._determine_batch_execution_and_padding(first_num_actual_tokens)
        )
        remaining_mode, remaining_input_batch_size, remaining_tokens_across_dp = (
            self._determine_batch_execution_and_padding(1)
        )
        if (
            first_mode is not CUDAGraphMode.PIECEWISE
            or remaining_mode is not CUDAGraphMode.PIECEWISE
            or first_tokens_across_dp is not None
            or remaining_tokens_across_dp is not None
            or first_input_batch_size > max_capture_size
            or remaining_input_batch_size > max_capture_size
        ):
            return None

        key = self._draft_chain_graph_key_for(
            common_attn_metadata,
            num_speculative_tokens,
            first_num_actual_tokens,
            first_input_batch_size,
            remaining_input_batch_size,
        )
        return key, first_input_batch_size, remaining_input_batch_size

    def _make_draft_chain_common_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        *,
        num_actual_tokens: int,
        query_start_loc: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
    ) -> CommonAttentionMetadata:
        # ``max_seq_len`` is deliberately capture-static. For the exact Triton
        # backend admitted above it is unused after being passed to
        # unified_attention; device seq_lens controls masking and loop bounds.
        return common_attn_metadata.replace(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            num_reqs=1,
            num_actual_tokens=num_actual_tokens,
            max_query_len=num_actual_tokens,
            max_seq_len=self.max_model_len,
            slot_mapping=slot_mapping,
            positions=positions,
            seq_lens_cpu_upper_bound=None,
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            _num_computed_tokens_cache=None,
        )

    def _update_positions_dependent_metadata_for_graph(
        self,
        positions: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        input_batch_size: int,
    ) -> torch.Tensor:
        """Run only the device portion of the existing position update."""
        out_positions = self.positions[:1]
        eagle_step_update_slot_mapping_and_metadata(
            positions_1d=positions,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            seq_lens=common_attn_metadata.seq_lens,
            block_size=self.block_size,
            max_model_len=self.max_model_len,
            out_clamped_positions=out_positions,
            out_slot_mapping=self._slot_mapping_buffer[:input_batch_size],
            input_batch_size=input_batch_size,
        )
        common_attn_metadata.slot_mapping = self._slot_mapping_buffer[:1]
        return out_positions

    def _run_draft_chain_graph_body(
        self,
        *,
        num_speculative_tokens: int,
        first_input_batch_size: int,
        remaining_input_batch_size: int,
        token_indices_to_sample: torch.Tensor,
        num_rejected_tokens: torch.Tensor,
        first_attn_metadata: dict[str, object],
        remaining_attn_metadata: list[dict[str, object]],
        first_slot_mapping: dict[str, torch.Tensor],
        remaining_slot_mapping: dict[str, torch.Tensor],
        remaining_common_attn_metadata: CommonAttentionMetadata,
    ) -> torch.Tensor:
        first_model_kwargs, _ = self.build_model_inputs_first_pass(
            num_speculative_tokens + 1,
            first_input_batch_size,
            mm_embed_inputs=None,
        )
        with set_forward_context(
            first_attn_metadata,
            self.vllm_config,
            num_tokens=first_input_batch_size,
            num_tokens_across_dp=None,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=first_slot_mapping,
        ):
            model_output = self.model(**first_model_kwargs)
            if self.model_returns_tuple():
                last_hidden_states, hidden_states = model_output
            else:
                last_hidden_states = hidden_states = model_output

        sample_hidden_states = last_hidden_states[token_indices_to_sample]
        positions = self.positions[token_indices_to_sample]
        hidden_states = hidden_states[token_indices_to_sample]
        draft_token_ids = self._greedy_sample(sample_hidden_states)
        draft_token_ids_list = [draft_token_ids]

        # Always subtract a persistent buffer. It contains zero for the first
        # proposal and the real rejection count thereafter.
        remaining_common_attn_metadata.seq_lens.sub_(num_rejected_tokens)
        for token_index in range(num_speculative_tokens - 1):
            positions = self._update_positions_dependent_metadata_for_graph(
                positions,
                remaining_common_attn_metadata,
                remaining_input_batch_size,
            )
            self.input_ids[:1].copy_(draft_token_ids.int())
            self.hidden_states[:1].copy_(hidden_states)
            model_kwargs = {
                "input_ids": self.input_ids[:remaining_input_batch_size],
                "positions": self.positions[:remaining_input_batch_size],
                "inputs_embeds": None,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = self.hidden_states[
                    :remaining_input_batch_size
                ]

            with set_forward_context(
                remaining_attn_metadata[token_index],
                self.vllm_config,
                num_tokens=remaining_input_batch_size,
                num_tokens_across_dp=None,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=remaining_slot_mapping,
            ):
                model_output = self.model(**model_kwargs)
                if self.model_returns_tuple():
                    last_hidden_states, hidden_states = model_output
                else:
                    last_hidden_states = hidden_states = model_output

            hidden_states = hidden_states[:1]
            draft_token_ids = self._greedy_sample(last_hidden_states[:1])
            draft_token_ids_list.append(draft_token_ids)

        return torch.stack(draft_token_ids_list, dim=1)

    def _capture_draft_chain_graph(
        self,
        num_tokens: int,
        common_attn_metadata: CommonAttentionMetadata | None,
    ) -> bool:
        """Capture one complete chain during the runner's capture window."""
        if (
            not _DRAFT_CHAIN_CUDAGRAPH_ENABLED
            or common_attn_metadata is None
            or torch.cuda.is_current_stream_capturing()
        ):
            return False
        num_speculative_tokens = self.num_speculative_tokens
        token_indices_to_sample = torch.full(
            (1,), num_tokens - 1, dtype=torch.int64, device=self.device
        )
        graph_spec = self._can_use_draft_chain_graph(
            common_attn_metadata,
            num_speculative_tokens,
            num_tokens,
            token_indices_to_sample,
            mm_embed_inputs=None,
        )
        if graph_spec is None:
            return False
        key, first_input_batch_size, remaining_input_batch_size = graph_spec
        if key in self._draft_chain_graphs:
            return True

        self._get_slot_mapping(
            first_input_batch_size, common_attn_metadata.slot_mapping
        )
        first_query_start_loc = torch.tensor(
            [0, num_tokens], dtype=torch.int32, device=self.device
        )
        first_query_start_loc_cpu = torch.tensor([0, num_tokens], dtype=torch.int32)
        remaining_query_start_loc = torch.tensor(
            [0, 1], dtype=torch.int32, device=self.device
        )
        remaining_query_start_loc_cpu = torch.tensor([0, 1], dtype=torch.int32)
        first_common = self._make_draft_chain_common_metadata(
            common_attn_metadata,
            num_actual_tokens=num_tokens,
            query_start_loc=first_query_start_loc,
            query_start_loc_cpu=first_query_start_loc_cpu,
            slot_mapping=self._slot_mapping_buffer[:num_tokens],
            positions=self.positions[:first_input_batch_size],
        )
        remaining_common = self._make_draft_chain_common_metadata(
            common_attn_metadata,
            num_actual_tokens=1,
            query_start_loc=remaining_query_start_loc,
            query_start_loc_cpu=remaining_query_start_loc_cpu,
            slot_mapping=self._slot_mapping_buffer[:1],
            positions=self.positions[:remaining_input_batch_size],
        )

        first_groups, first_layers = self.build_per_group_and_layer_attn_metadata(
            first_common, draft_index=0
        )
        remaining_groups: list[list[object]] = []
        remaining_layers: list[dict[str, object]] = []
        for draft_index in range(1, num_speculative_tokens):
            groups, layers = self.build_per_group_and_layer_attn_metadata(
                remaining_common, draft_index=draft_index
            )
            remaining_groups.append(groups)
            remaining_layers.append(layers)

        attn_group = self.draft_attn_groups[0]
        if not self._draft_attn_metadata_matches_common(
            (first_groups, first_layers), first_common, attn_group
        ) or any(
            not self._draft_attn_metadata_matches_common(
                (groups, layers), remaining_common, attn_group
            )
            for groups, layers in zip(remaining_groups, remaining_layers)
        ):
            return False

        first_slot_mapping = self._get_slot_mapping(first_input_batch_size)
        remaining_slot_mapping = {
            name: self._slot_mapping_buffer[:remaining_input_batch_size]
            for name in self._draft_attn_layer_names
        }
        num_rejected_tokens = torch.zeros(
            (1,), dtype=common_attn_metadata.seq_lens.dtype, device=self.device
        )

        def run_chain() -> torch.Tensor:
            return self._run_draft_chain_graph_body(
                num_speculative_tokens=num_speculative_tokens,
                first_input_batch_size=first_input_batch_size,
                remaining_input_batch_size=remaining_input_batch_size,
                token_indices_to_sample=token_indices_to_sample,
                num_rejected_tokens=num_rejected_tokens,
                first_attn_metadata=first_layers,
                remaining_attn_metadata=remaining_layers,
                first_slot_mapping=first_slot_mapping,
                remaining_slot_mapping=remaining_slot_mapping,
                remaining_common_attn_metadata=remaining_common,
            )

        # The eager call initializes compiled pieces and communicators before
        # capture. Restore every mutable input it touches before recording.
        saved_seq_lens = common_attn_metadata.seq_lens.clone()
        saved_input_ids = self.input_ids[:first_input_batch_size].clone()
        saved_positions = self.positions[:first_input_batch_size].clone()
        saved_hidden_states = self.hidden_states[:first_input_batch_size].clone()
        saved_slot_mapping = self._slot_mapping_buffer[:first_input_batch_size].clone()

        def restore_inputs() -> None:
            common_attn_metadata.seq_lens.copy_(saved_seq_lens)
            self.input_ids[:first_input_batch_size].copy_(saved_input_ids)
            self.positions[:first_input_batch_size].copy_(saved_positions)
            self.hidden_states[:first_input_batch_size].copy_(saved_hidden_states)
            self._slot_mapping_buffer[:first_input_batch_size].copy_(saved_slot_mapping)

        run_chain()
        restore_inputs()
        torch.accelerator.synchronize()

        graph = torch.cuda.CUDAGraph()
        graph_pool = self._draft_chain_graph_pool
        if graph_pool is not None:
            set_graph_pool_id(graph_pool)
        else:
            set_graph_pool_id(current_platform.graph_pool_handle())
        with torch.cuda.graph(graph, pool=graph_pool, stream=current_stream()):
            output = run_chain()
        restore_inputs()

        references = (
            first_common,
            remaining_common,
            first_groups,
            first_layers,
            remaining_groups,
            remaining_layers,
            first_slot_mapping,
            remaining_slot_mapping,
        )
        self._draft_chain_graphs[key] = _DraftChainGraphEntry(
            graph=graph,
            output=output,
            num_speculative_tokens=num_speculative_tokens,
            first_num_actual_tokens=num_tokens,
            first_input_batch_size=first_input_batch_size,
            remaining_input_batch_size=remaining_input_batch_size,
            token_indices_to_sample=token_indices_to_sample,
            num_rejected_tokens=num_rejected_tokens,
            seq_lens=common_attn_metadata.seq_lens,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            references=references,
        )
        if not self._draft_chain_graph_is_profiling:
            self._record_draft_chain_graph_event(
                "capture", num_speculative_tokens=num_speculative_tokens
            )
        return True

    def _reconcile_common_metadata_after_draft_chain(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        num_speculative_tokens: int,
        had_rejected_tokens: bool,
    ) -> None:
        num_updates = num_speculative_tokens - 1
        common_attn_metadata.num_actual_tokens = 1
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[:2]
        common_attn_metadata.query_start_loc_cpu = self._get_draft_query_start_loc_cpu(
            1
        )
        common_attn_metadata.slot_mapping = self._slot_mapping_buffer[:1]
        common_attn_metadata.max_seq_len = min(
            common_attn_metadata.max_seq_len + num_updates,
            self.max_model_len,
        )
        if had_rejected_tokens:
            common_attn_metadata._seq_lens_cpu = None
            common_attn_metadata._num_computed_tokens_cpu = None
        else:
            if common_attn_metadata._seq_lens_cpu is not None:
                common_attn_metadata._seq_lens_cpu += num_updates
            if common_attn_metadata._num_computed_tokens_cpu is not None:
                common_attn_metadata._num_computed_tokens_cpu += num_updates
        if common_attn_metadata.seq_lens_cpu_upper_bound is not None:
            common_attn_metadata.seq_lens_cpu_upper_bound += num_updates

    def _try_replay_draft_chain_graph(
        self,
        *,
        num_speculative_tokens: int,
        num_tokens: int,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> torch.Tensor | object:
        if not _DRAFT_CHAIN_CUDAGRAPH_ENABLED:
            return _NO_DRAFT_CHAIN_REPLAY
        graph_spec = (
            None
            if num_rejected_tokens_gpu is None
            else self._can_use_draft_chain_graph(
                common_attn_metadata,
                num_speculative_tokens,
                num_tokens,
                token_indices_to_sample,
                mm_embed_inputs,
            )
        )
        entry = (
            None if graph_spec is None else self._draft_chain_graphs.get(graph_spec[0])
        )
        if (
            entry is None
            or not self._draft_attn_tensor_binding_matches(
                entry.seq_lens, common_attn_metadata.seq_lens
            )
            or not self._draft_attn_tensor_binding_matches(
                entry.block_table_tensor,
                common_attn_metadata.block_table_tensor,
            )
            or common_attn_metadata.slot_mapping.shape[0] < num_tokens
            or (
                num_rejected_tokens_gpu is not None
                and num_rejected_tokens_gpu.numel() != 1
            )
        ):
            self._record_draft_chain_graph_event(
                "fallback",
                num_drafter_forwards=max(num_speculative_tokens, 1),
                num_speculative_tokens=num_speculative_tokens,
            )
            return _NO_DRAFT_CHAIN_REPLAY

        assert token_indices_to_sample is not None
        entry.token_indices_to_sample.copy_(token_indices_to_sample)
        if num_rejected_tokens_gpu is None:
            entry.num_rejected_tokens.zero_()
        else:
            entry.num_rejected_tokens.copy_(num_rejected_tokens_gpu)
        self._get_slot_mapping(
            entry.first_input_batch_size, common_attn_metadata.slot_mapping
        )
        entry.graph.replay()
        self._reconcile_common_metadata_after_draft_chain(
            common_attn_metadata,
            num_speculative_tokens,
            had_rejected_tokens=num_rejected_tokens_gpu is not None,
        )
        self._record_draft_chain_graph_event(
            "replay",
            num_drafter_forwards=num_speculative_tokens,
            num_speculative_tokens=num_speculative_tokens,
        )
        return entry.output

    def _get_draft_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
        phase: str,
        input_batch_size: int,
        cudagraph_runtime_mode: CUDAGraphMode,
        cache_is_supported: bool = False,
    ) -> tuple[list[object], dict[str, object]]:
        if (
            phase not in ("first", "remaining")
            or (phase == "first") != (draft_index == 0)
            or input_batch_size < common_attn_metadata.num_actual_tokens
            or cudagraph_runtime_mode is not CUDAGraphMode.PIECEWISE
            or (
                not cache_is_supported
                and not self._can_cache_draft_attn_metadata(
                    common_attn_metadata, draft_index
                )
            )
        ):
            if self.method == "eagle3":
                self._record_metadata_template_event("fallback")
            return self.build_per_group_and_layer_attn_metadata(
                common_attn_metadata, draft_index
            )

        stability_key = self._draft_attn_metadata_cache_key_for(
            common_attn_metadata,
            draft_index,
            phase,
            input_batch_size,
            cudagraph_runtime_mode,
        )
        if stability_key in self._disabled_draft_attn_metadata_cache_keys:
            self._record_metadata_template_event("fallback")
            return self.build_per_group_and_layer_attn_metadata(
                common_attn_metadata, draft_index
            )

        cache_entry = self._draft_attn_metadata_cache.get(stability_key)
        if cache_entry is not None and not self._draft_attn_metadata_bindings_match(
            cache_entry, common_attn_metadata
        ):
            self._draft_attn_metadata_cache.pop(stability_key, None)
            self._disabled_draft_attn_metadata_cache_keys.add(stability_key)
            self._record_metadata_template_event("fallback")
            return self.build_per_group_and_layer_attn_metadata(
                common_attn_metadata, draft_index
            )

        if cache_entry is not None:
            cached = cache_entry
            cast(Any, cached[0][0]).max_seq_len = common_attn_metadata.max_seq_len
            self._record_metadata_template_event("hit")
            return cached

        built = self.build_per_group_and_layer_attn_metadata(
            common_attn_metadata, draft_index
        )
        attn_group = self.draft_attn_groups[0]
        if len(
            self._draft_attn_metadata_cache
        ) >= 256 or not self._draft_attn_metadata_matches_common(
            built, common_attn_metadata, attn_group
        ):
            self._disabled_draft_attn_metadata_cache_keys.add(stability_key)
            self._record_metadata_template_event("fallback")
            return built

        self._draft_attn_metadata_cache[stability_key] = built
        self._record_metadata_template_event("build")
        return built

    def model_returns_tuple(self) -> bool:
        if self.method == "mtp":
            # DeepSeek-family MTP (deepseek_mtp.py) recycles the post-final-
            # norm hidden, so its forward returns (logit_hidden,
            # recycle_hidden). Other MTP families return a single tensor.
            return "DeepSeekMTPModel" in (
                self.draft_model_config.hf_config.architectures or []
            )
        return self.method not in ("mtp", "draft_model", "dflash")

    def prepare_next_token_ids_cpu(
        self,
        sampled_token_ids: list[list[int]],
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        num_scheduled_tokens: dict[str, int],
    ) -> torch.Tensor:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids for each request based on the sampled
        token ids from the CPU. If a request has no sampled token ids (e.g.,
        during the initial decoding steps), it falls back to using the request
        state to get the next token id.
        """
        req_ids = gpu_input_batch.req_ids
        next_token_ids: list[int] = []
        for i, token_ids in enumerate(sampled_token_ids):
            if token_ids:
                # Common case.
                next_token_id = token_ids[-1]
            else:
                # Partial prefill (rare case).
                # Get the next token id from the request state.
                req_id = req_ids[i]
                req_state = requests[req_id]
                seq_len = req_state.num_computed_tokens + num_scheduled_tokens[req_id]
                next_token_id = req_state.get_token_id(seq_len)
            next_token_ids.append(next_token_id)
        next_token_ids = torch.tensor(
            next_token_ids, dtype=torch.int32, device=self.input_ids.device
        )
        return next_token_ids

    def prepare_next_token_ids_padded(
        self,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids and the number of valid sampled tokens
        for each request, considering the "discarded" requests whose next token
        is not sampled and comes from `request.get_token_id()` instead. This is denoted
        the "backup" token id. It also counts rejected tokens via `sampled_token_ids`.
        """
        # Precompute backup token IDs for discarded requests.
        num_reqs = gpu_input_batch.num_reqs
        for i in range(num_reqs):
            self.backup_next_token_ids.np[i] = requests[
                gpu_input_batch.req_ids[i]
            ].get_token_id(gpu_input_batch.num_tokens_no_spec[i] - 1)
        self.backup_next_token_ids.copy_to_gpu(num_reqs)
        backup_tokens_gpu = self.backup_next_token_ids.gpu

        batch_size, num_tokens = sampled_token_ids.shape
        device = sampled_token_ids.device

        assert discard_request_mask.dtype == torch.bool
        assert backup_tokens_gpu.dtype == torch.int32

        next_token_ids = torch.empty(batch_size, dtype=torch.int32, device=device)
        valid_sampled_tokens_count = next_token_ids.new_empty(batch_size)

        # Kernel grid: one program per request (row)
        grid = (batch_size,)

        # Find the next power of 2 for block sizes
        BLOCK_SIZE_TOKENS = next_power_of_2(num_tokens)
        eagle_prepare_next_token_padded_kernel[grid](
            sampled_token_ids,
            discard_request_mask,
            backup_tokens_gpu,
            next_token_ids,
            valid_sampled_tokens_count,
            gpu_input_batch.vocab_size,
            num_tokens,
            batch_size,
            sampled_token_ids.stride(0),
            BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
        )

        return next_token_ids, valid_sampled_tokens_count

    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        No blocking CPU operations should be introduced in this function.
        """
        num_reqs = common_attn_metadata.num_reqs
        device = valid_sampled_tokens_count.device

        token_indices_to_sample = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )
        num_rejected_tokens_gpu = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )

        grid = (num_reqs,)
        eagle_prepare_inputs_padded_kernel[grid](
            spec_decode_metadata.cu_num_draft_tokens,
            valid_sampled_tokens_count,
            common_attn_metadata.query_start_loc,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
            num_reqs,
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            seq_lens_cpu_upper_bound=common_attn_metadata.seq_lens_cpu_upper_bound,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=common_attn_metadata.max_seq_len,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return (
            spec_common_attn_metadata,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
        )

    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: list[list[int]],
        num_draft_tokens: list[int],
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        # E.g.
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1, q1 + q2, q1 + q2 + q3]
        #  common_attn_metadata.seq_lens{_cpu}: [s1, s2, s3]
        #  num_rejected_tokens: [n1, n2, n3]
        # This function computes the intermediate values:
        #  num_tokens_per_req: [q1 - n1, q2 - n2, q3 - n3]
        # And returns:
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        #  common_attn_metadata.seq_lens{_cpu}:
        #       [s1 - n1 + 1, s2 - n2 + 1, s3 - n3 + 1]
        #  token_indices: [0, 1, ..., q1 - n1 - 1,
        #                 q1, q1 + 1, ..., q1 + q2 - n2 - 1,
        #                 q1 + q2, q1 + q2 + 1, ..., q1 + q2 + q3 - n3 - 1]

        num_rejected_tokens = [
            n + 1 - len(sampled_token_ids[i]) if n > 0 else 0
            for i, n in enumerate(num_draft_tokens)
        ]
        num_rejected_tokens = torch.tensor(num_rejected_tokens, dtype=torch.int32)

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        # upper_bound - rejected = actual post-rejection seq_lens (no D2H sync).
        assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
        new_seq_lens_cpu = (
            common_attn_metadata.seq_lens_cpu_upper_bound - num_rejected_tokens
        )

        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # [q1 - n1, q2 - n2, q3 - n3] ->
        # [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
        )
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])

        total_num_tokens = new_query_start_loc_np[-1]
        # Example assuming num_tokens_per_req_np = [2, 4, 3]
        # this implies that `new_query_start_locs` is:
        # [0, 2, 6, 9] ->
        # [0, 0, 2, 2, 2, 2, 6, 6, 6]
        #  _r1_  ____r2____  ___r3__
        new_query_start_locs_expanded = np.repeat(
            new_query_start_loc_np[:-1], new_num_tokens_per_req_np
        )
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->
        # [0, 1, 0, 1, 2, 3, 0, 1, 2]
        #  _r1_  ____r2____  ___r3__
        token_offsets = (
            self.token_arange_np[:total_num_tokens] - new_query_start_locs_expanded
        )

        # Expand starting positions to match token pattern
        # [0, q1, q1 + q2] ->
        # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]
        #  _r1_  _____r2_______  ___________r3____________
        old_query_start_locs_expanded = np.repeat(
            query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np
        )
        # Final token indices are:
        # [0, 1,                                // req 1
        #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,       // req 2
        #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2] // req 3
        token_indices_np = token_offsets + old_query_start_locs_expanded
        token_indices = torch.from_numpy(token_indices_np).to(device, non_blocking=True)

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc_cpu.to(device, non_blocking=True),
            seq_lens=new_seq_lens_cpu.to(device, non_blocking=True),
            query_start_loc_cpu=new_query_start_loc_cpu,
            _seq_lens_cpu=new_seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            seq_lens_cpu_upper_bound=new_seq_lens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=new_seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[token_indices],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return spec_common_attn_metadata, token_indices

    def get_model_name(self, model: nn.Module) -> str:
        if hasattr(model, "module"):  # multi-GPU
            model = model.module
        return model.__class__.__name__

    def _create_draft_vllm_config(self) -> VllmConfig:
        """Return a VllmConfig with kernel-level overrides for the proposer.
        Subclasses may override to apply additional config changes.
        """
        spec_cfg = self.speculative_config
        base = self.vllm_config

        if spec_cfg.moe_backend is not None:
            base = replace(
                base,
                kernel_config=replace(
                    base.kernel_config,
                    moe_backend=spec_cfg.moe_backend,
                ),
            )

        # Note (matt): Never inherit the attention backend from base, because there are
        # many opportunities for incompatibility, so we always independently autoselect
        # unless explicitly specified in the speculative config.
        base = replace(
            base,
            attention_config=replace(
                base.attention_config,
                backend=spec_cfg.attention_backend,
            ),
        )

        return base

    def _get_model(self) -> nn.Module:
        """
        Default method to call get_model(). Can be overridden by subclasses which
        need to customize model loading.
        """
        from vllm.compilation.backends import set_model_tag

        draft_vllm_config = self._create_draft_vllm_config()
        with set_model_tag("eagle_head"):
            model = get_model(
                vllm_config=draft_vllm_config,
                model_config=self.speculative_config.draft_model_config,
                load_config=self.speculative_config.draft_load_config,
            )
        return model

    def load_model(self, target_model: nn.Module) -> None:
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )

        self.model = self._get_model()

        # Find draft layers (attention layers added by draft model)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        # Filter to only layers that have KV cache specs.
        self._draft_attn_layer_names = {
            name
            for name in (set(all_attn_layers.keys()) - target_attn_layer_names)
            if all_attn_layers[name].get_kv_cache_spec(self.vllm_config) is not None
        }

        if self.supports_mm_inputs:
            # Even if the target model is multimodal, we can also use
            # text-only draft models
            try:
                dummy_input_ids = torch.tensor([[1]], device=self.input_ids.device)
                self.model.embed_input_ids(dummy_input_ids, multimodal_embeddings=None)
            except (NotImplementedError, AttributeError, TypeError):
                logger.warning(
                    "Draft model does not support multimodal inputs, "
                    "falling back to text-only mode"
                )
                self.supports_mm_inputs = False

        if supports_multimodal(target_model):
            # handle multimodality
            assert hasattr(target_model, "config")
            if self.get_model_name(target_model) in [
                "Cohere2VisionForConditionalGeneration",
                "Exaone4_5_ForConditionalGeneration",
                "GlmOcrForConditionalGeneration",
                "HunYuanVLForConditionalGeneration",
                "InternS2PreviewForConditionalGeneration",
                "MiMoV2OmniForCausalLM",
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
                "Qwen3VLMoeForConditionalGeneration",
                "Gemma4ForConditionalGeneration",
                "Gemma4UnifiedForConditionalGeneration",
                "Step3p7ForConditionalGeneration",
            ]:
                self.model.config.image_token_index = target_model.config.image_token_id
            elif self.get_model_name(target_model) == "PixtralForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.vision_config.image_token_id
                )
            elif self.get_model_name(target_model) == "KimiK25ForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.media_placeholder_token_id
                )
            else:
                self.model.config.image_token_index = (
                    target_model.config.image_token_index
                )
            target_language_model = cast(
                SupportsMultiModal, target_model
            ).get_language_model()
        else:
            target_language_model = target_model

        self._maybe_share_embeddings(target_language_model)
        self._maybe_share_lm_head(target_language_model)

        if (
            self.parallel_drafting
            and self.pass_hidden_states_to_model
            and self.parallel_drafting_hidden_state_tensor is not None
        ):
            flat_mask = self.model.mask_hidden.view(-1)
            if self.eagle3_use_aux_hidden_state:
                # EAGLE3: mask_hidden stores all aux hidden states,
                # project through combine_hidden_states
                self.parallel_drafting_hidden_state_tensor.copy_(
                    self.model.combine_hidden_states(flat_mask)
                )
            else:
                self.parallel_drafting_hidden_state_tensor.copy_(flat_mask)

    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        """
        Some draft models may not have their own embedding layers, and some may
        have a duplicate copy of the target model's embedding layers. In these cases,
        we share the target model's embedding layers with the draft model to save
        memory.
        """
        if get_pp_group().world_size == 1:
            inner_model = getattr(target_language_model, "model", None)
            if inner_model is None:
                raise AttributeError("Target model does not have 'model' attribute")
            if hasattr(inner_model, "embed_tokens"):
                target_embed_tokens = inner_model.embed_tokens
            elif hasattr(inner_model, "embedding"):
                target_embed_tokens = inner_model.embedding
            else:
                raise AttributeError(
                    "Target model does not have 'embed_tokens' or 'embedding' attribute"
                )

            share_embeddings = False
            if hasattr(self.model, "has_own_embed_tokens"):
                # EAGLE model
                if not self.model.has_own_embed_tokens:
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model without its own embed_tokens in the"
                        " checkpoint. Sharing target model embedding weights with the"
                        " draft model."
                    )
                elif (
                    isinstance(target_embed_tokens.weight, torch.Tensor)
                    and isinstance(self.model.model.embed_tokens.weight, torch.Tensor)
                    # TODO: Offload to CPU for comparison to avoid extra GPU memory
                    # usage in CI testing environments with limited GPU memory
                    and torch.equal(
                        target_embed_tokens.weight.cpu(),
                        self.model.model.embed_tokens.weight.cpu(),
                    )
                ):
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model with embed_tokens identical to the target"
                        " model. Sharing target model embedding weights with the draft"
                        " model."
                    )
                else:
                    logger.info(
                        "Detected EAGLE model with distinct embed_tokens weights. "
                        "Keeping separate embedding weights from the target model."
                    )
            else:
                # MTP model
                share_embeddings = True
                logger.info(
                    "Detected MTP model. "
                    "Sharing target model embedding weights with the draft model."
                )

            if share_embeddings:
                if hasattr(self.model.model, "embed_tokens"):
                    del self.model.model.embed_tokens
                self.model.model.embed_tokens = target_embed_tokens
        else:
            logger.info(
                "The draft model's vocab embedding will be loaded separately"
                " from the target model."
            )

    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        """
        Some draft models may not have their own LM head, and some may have a
        duplicate copy of the target model's LM head. In these cases, we share
        the target model's LM head with the draft model to save memory.
        """
        share_lm_head = False
        if hasattr(self.model, "has_own_lm_head"):
            # EAGLE model
            if not self.model.has_own_lm_head:
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model without its own lm_head in the checkpoint. "
                    "Sharing target model lm_head weights with the draft model."
                )
            elif (
                hasattr(target_language_model, "lm_head")
                and hasattr(target_language_model.lm_head, "weight")
                and hasattr(self.model.lm_head, "weight")
                and isinstance(target_language_model.lm_head.weight, torch.Tensor)
                and isinstance(self.model.lm_head.weight, torch.Tensor)
                # TODO: Offload to CPU for comparison to avoid extra GPU memory
                # usage in CI testing environments with limited GPU memory
                and torch.equal(
                    target_language_model.lm_head.weight.cpu(),
                    self.model.lm_head.weight.cpu(),
                )
            ):
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model with lm_head identical to the target model. "
                    "Sharing target model lm_head weights with the draft model."
                )
            else:
                logger.info(
                    "Detected EAGLE model with distinct lm_head weights. "
                    "Keeping separate lm_head weights from the target model."
                )
        else:
            # MTP model
            share_lm_head = True
            logger.info(
                "Detected MTP model. "
                "Sharing target model lm_head weights with the draft model."
            )

        if share_lm_head and hasattr(target_language_model, "lm_head"):
            if hasattr(self.model, "lm_head"):
                del self.model.lm_head
            self.model.lm_head = target_language_model.lm_head

            # MTP models call compute_logits via shared_head.head (a
            # ParallelLMHead inside each MTP layer), not self.model.lm_head.
            # If the checkpoint omits a copy of the lm_head weights at the
            # MTP layer path, shared_head.head stays uninitialised and
            # produces NaN logits. Always share it explicitly.
            inner = getattr(self.model, "model", None)
            layers = getattr(inner, "layers", None) if inner else None
            if layers is not None:
                items = layers.values() if isinstance(layers, nn.ModuleDict) else layers
                for layer in items:
                    sh = getattr(layer, "shared_head", None)
                    if sh is not None and hasattr(sh, "head"):
                        del sh.head
                        sh.head = target_language_model.lm_head
                        logger.info(
                            "Shared target model lm_head with MTP shared_head.head."
                        )

        if hasattr(target_language_model.model, "topk_indices_buffer"):
            target_buffer = target_language_model.model.topk_indices_buffer
            if hasattr(self.model.model, "topk_indices_buffer"):
                del self.model.model.topk_indices_buffer
            self.model.model.topk_indices_buffer = target_buffer
            # Also share at per-module level so that the indexer and
            # sparse-attention backends in each MTP layer read from
            # the target model's buffer.
            for _, module in self.model.model.named_modules():
                if hasattr(module, "topk_indices_buffer"):
                    module.topk_indices_buffer = target_buffer
            logger.info(
                "Detected MTP model with topk_indices_buffer. "
                "Sharing target model topk_indices_buffer with the draft model."
            )

        # Detect index_share_for_mtp_iteration: when True, the proposer
        # toggles skip_topk so step 0 computes MTP's own indices and
        # steps 1+ reuse them.
        spec_config = self.vllm_config.speculative_config
        draft_hf_config = (
            spec_config.draft_model_config.hf_config
            if spec_config is not None
            else None
        )
        self._share_mtp_indices = getattr(
            draft_hf_config, "index_share_for_mtp_iteration", False
        )

        if self.use_local_argmax_reduction:
            if not hasattr(self.model, "get_top_tokens"):
                raise ValueError(
                    "use_local_argmax_reduction is enabled but draft model "
                    f"{self.model.__class__.__name__} does not implement "
                    "get_top_tokens()."
                )
            logger.info(
                "Using local argmax reduction for draft token generation "
                "(communication: O(2*tp_size) vs O(vocab_size))."
            )

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
        common_attn_metadata: CommonAttentionMetadata | None = None,
    ) -> None:
        if (
            is_graph_capturing
            and self.method == "eagle3"
            and _DRAFT_CHAIN_CUDAGRAPH_ENABLED
            and self._capture_draft_chain_graph(num_tokens, common_attn_metadata)
        ):
            return

        # FIXME: when using tree-based specdec, adjust number of forward-passes
        # according to the depth of the tree.
        only_one_forward_pass = is_graph_capturing or self.parallel_drafting
        for fwd_idx in range(
            1 if only_one_forward_pass else self.num_speculative_tokens
        ):
            if fwd_idx <= 1:
                cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
                    self._determine_batch_execution_and_padding(
                        num_tokens, use_cudagraphs=use_cudagraphs
                    )
                )

            # Make sure to use EAGLE's own buffer during cudagraph capture.
            if (
                self._draft_attn_layer_names
                and slot_mappings is not None
                and next(iter(self._draft_attn_layer_names)) in slot_mappings
            ):
                slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
            else:
                slot_mapping_dict = slot_mappings or {}

            with set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=slot_mapping_dict,
            ):
                if self.supports_mm_inputs:
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:num_input_tokens]
                else:
                    input_ids = self.input_ids[:num_input_tokens]
                    inputs_embeds = None

                kwargs = dict(
                    input_ids=input_ids,
                    positions=self._get_positions(num_input_tokens),
                    inputs_embeds=inputs_embeds,
                )
                if self.pass_hidden_states_to_model:
                    kwargs["hidden_states"] = self.hidden_states[:num_input_tokens]
                self.model(**kwargs)

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        """
        Some eagle3 heads (e.g., nvidia/gpt-oss-120b-Eagle3-v2) do not use auxiliary
        hidden states and directly uses the last layer output just like eagle1.
        They might indicate this by setting "use_aux_hidden_state" to False
        inside the "eagle_config" dict of their hf_config.
        """
        if self.method != "eagle3":
            return False
        # Assume that eagle3 heads use aux hidden states by default
        use_aux_hidden_state = True
        eagle_config = getattr(self.draft_model_config.hf_config, "eagle_config", None)
        if eagle_config is not None:
            use_aux_hidden_state = eagle_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Validate that all drafting layers belong to the same KVCacheGroup.
        Need this assumption to ensure all drafting layers can use the
        same AttentionMetadata.
        May extend to multiple AttentionMetadata in the future.
        """
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id
        assert (
            len(
                set(
                    [
                        kv_cache_groups[layer_name]
                        for layer_name in self._draft_attn_layer_names
                    ]
                )
            )
            == 1
        ), "All drafting layers should belong to the same kv cache group"

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        """
        Initialize AttentionGroups for draft layers using kv_cache_config.
        Called from the model runner's initialize_metadata_builders.
        """
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        # Find which kv_cache_group the draft layers belong to
        self.validate_same_kv_cache_group(kv_cache_config)
        kv_cache_spec = None
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            if self._draft_attn_layer_names & set(group.layer_names):
                self.kv_cache_gid = gid
                kv_cache_spec = group.kv_cache_spec
                break

        attention_groups: dict[tuple[str, str], AttentionGroup] = {}
        if kv_cache_spec is not None:
            for layer_name in self._draft_attn_layer_names:
                attn_backend = all_attn_layers[layer_name].get_attn_backend()
                backend_key = attn_backend.full_cls_name()
                if backend_key not in attention_groups:
                    layer_kv_cache_spec = kv_cache_spec
                    if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                        layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[
                            layer_name
                        ]

                    kernel_block_size = (
                        kernel_block_sizes[self.kv_cache_gid]
                        if kernel_block_sizes is not None
                        and self.kv_cache_gid < len(kernel_block_sizes)
                        else None
                    )
                    attn_group = AttentionGroup(
                        backend=attn_backend,
                        layer_names=[layer_name],
                        kv_cache_spec=layer_kv_cache_spec,
                        kv_cache_group_id=self.kv_cache_gid,
                    )
                    attn_group.create_metadata_builders(
                        self.vllm_config,
                        self.device,
                        kernel_block_size=kernel_block_size,
                    )
                    attention_groups[backend_key] = attn_group
                else:
                    attention_groups[backend_key].layer_names.append(layer_name)

        self.draft_attn_groups = list(attention_groups.values())
        self.block_size = (
            self.draft_attn_groups[0].get_metadata_builder().kv_cache_spec.block_size
        )
        self._initialize_draft_attn_metadata_cache_capability()
        self._initialize_draft_chain_graph_capability()
        logger.debug("Using block size %d for drafting layers", self.block_size)

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
    ) -> tuple[CUDAGraphMode, int, torch.Tensor | None]:
        cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            valid_modes=({CUDAGraphMode.NONE} if not use_cudagraphs else None),
        )
        num_tokens_padded = batch_desc.num_tokens

        # Extra coordination when running data-parallel since we need to
        # coordinate across ranks
        # TODO(Flechman): support DBO ubatching
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.vllm_config.parallel_config,
                    allow_microbatching=False,
                    num_tokens_padded=num_tokens_padded,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )
            assert not should_ubatch, "DBO ubatching not implemented for EAGLE"

            # Extract DP-synced values
            if num_tokens_across_dp is not None:
                dp_rank = self.dp_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding so we have the correct
                # batch_descriptor
                cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
                    num_tokens_padded,
                    valid_modes={CUDAGraphMode(synced_cudagraph_mode)},
                )
                # Assert to make sure the agreed upon token count is correct
                # otherwise num_tokens_across_dp will no-longer be valid
                assert batch_desc.num_tokens == num_tokens_padded
                num_tokens_across_dp[dp_rank] = num_tokens_padded

        return cudagraph_mode, num_tokens_padded, num_tokens_across_dp


# NOTE(woosuk): Currently, the below code is not used and we always use argmax
# to sample the draft tokens. We will use this after we find a way to manage
# the draft prob tensor.
# Refer to https://github.com/vllm-project/vllm/pull/16899 for the details.
# FIXME(woosuk): The logic here is duplicated with the main sampling code.
# We should refactor this to reuse the same sampling implementation.
def compute_probs_and_sample_next_token(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    use_fp64_gumbel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling_metadata.all_greedy:
        # For greedy requests, draft_probs is not used in rejection sampling.
        # Therefore, we can just return the logits.
        probs = logits
        next_token_ids = logits.argmax(dim=-1)
        return next_token_ids, probs

    assert sampling_metadata.temperature is not None

    # Use epsilon comparison to detect greedy sampling (temperature ~ 0.0)
    # consistent with sampler.py's _SAMPLING_EPS threshold
    temperature = sampling_metadata.temperature
    # Avoid division by zero if there are greedy requests.
    if not sampling_metadata.all_random:
        is_greedy = temperature < _SAMPLING_EPS
        temperature = torch.where(is_greedy, 1.0, temperature)
    logits.div_(temperature.view(-1, 1))
    probs = logits.softmax(dim=-1, dtype=torch.float32)

    # NOTE(woosuk): Currently, we ignore most of the sampling parameters in
    # generating the draft tokens. We only use the temperature. While this
    # could degrade the acceptance rate, it does not affect the distribution
    # of the generated tokens after rejection sampling.

    # TODO(woosuk): Consider seeds.
    q = empty_exponential_noise_like(probs, use_fp64_gumbel)
    q.exponential_()
    # NOTE(woosuk): We shouldn't use `probs.div_(q)` because the draft_probs
    # will be used later for rejection sampling.
    next_token_ids = sample_with_exponential_noise(probs.clone(), q)
    if not sampling_metadata.all_random:
        greedy_token_ids = probs.argmax(dim=-1)
        next_token_ids = torch.where(is_greedy, greedy_token_ids, next_token_ids)
    return next_token_ids, probs
