# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest import mock

import numpy as np
import pytest
import torch

from vllm import envs
from vllm.config import CUDAGraphMode
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.worker.utils import AttentionGroup


def _make_common_metadata() -> CommonAttentionMetadata:
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=8,
        block_table_tensor=torch.tensor([[0, 1]], dtype=torch.int32),
        slot_mapping=torch.tensor([7], dtype=torch.int64),
    )


def _make_proposer(
    num_speculative_tokens: int = 2,
) -> tuple[SpecDecodeBaseProposer, mock.Mock]:
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.method = "eagle3"
    proposer.hidden_size = 4
    proposer.parallel_drafting = False
    proposer.constant_draft_positions = False
    proposer.needs_extra_input_slots = False
    proposer.supports_mm_inputs = False
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0
    proposer.draft_uses_xdrope_dim = 0
    proposer.num_speculative_tokens = num_speculative_tokens
    proposer.model = object.__new__(Eagle3LlamaForCausalLM)
    proposer.max_batch_size = 1
    proposer.pass_hidden_states_to_model = True
    proposer.use_local_argmax_reduction = False
    proposer._enable_probabilistic_draft_probs = False
    proposer._share_mtp_indices = False
    proposer.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        lora_config=None,
    )
    proposer.speculative_config = SimpleNamespace(
        disable_padded_drafter_batch=False,
        enforce_eager=False,
    )
    proposer._draft_attn_layer_names = {"layer.0"}
    proposer.token_arange_np = np.arange(16, dtype=np.int32)

    builder = object.__new__(FlashAttentionMetadataBuilder)
    builder.layer_names = ["layer.0"]
    builder.dcp_world_size = 1
    builder.cp_kv_cache_interleave_size = 1
    builder.use_full_cuda_graph = True
    builder.max_cudagraph_size = 4
    builder.max_num_splits = 4
    builder.aot_schedule = True
    builder.kv_cache_spec = object()
    builder.block_size = 16
    builder.kv_cache_dtype = "auto"
    builder._test_scheduler_metadata = None

    def build_for_drafting(
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> FlashAttentionMetadata:
        del draft_index
        return FlashAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
            max_dcp_context_kv_len=0,
            dcp_context_kv_lens=None,
            scheduler_metadata=builder._test_scheduler_metadata,
            prefix_scheduler_metadata=None,
            max_num_splits=builder.max_num_splits,
            causal=common_attn_metadata.causal,
        )

    build_mock = mock.Mock(side_effect=build_for_drafting)
    builder.build_for_drafting = build_mock
    proposer.draft_attn_groups = [
        AttentionGroup(
            backend=FlashAttentionBackend,
            layer_names=builder.layer_names,
            kv_cache_spec=builder.kv_cache_spec,
            kv_cache_group_id=0,
            metadata_builders=[builder],
        )
    ]

    proposer._draft_attn_metadata_cache = {}
    proposer._disabled_draft_attn_metadata_cache_keys = set()
    proposer._draft_attn_metadata_cache_group_key = None
    proposer._draft_query_start_loc_cpu_cache = {}
    proposer.metadata_template_builds = 0
    proposer.metadata_template_hits = 0
    proposer.metadata_template_fallbacks = 0
    proposer._draft_chain_graphs = {}
    proposer._draft_chain_graph_capability_key = None
    proposer.draft_chain_graph_captures = 0
    proposer.draft_chain_graph_replays = 0
    proposer.draft_chain_graph_fallbacks = 0
    proposer._draft_chain_forward_events = 0
    proposer._draft_chain_graph_stats_by_k = {}
    proposer._initialize_draft_attn_metadata_cache_capability()
    return proposer, build_mock


def _make_triton_proposer(
    num_speculative_tokens: int = 2,
) -> tuple[SpecDecodeBaseProposer, mock.Mock, TritonAttentionMetadataBuilder]:
    proposer, _ = _make_proposer(num_speculative_tokens)
    proposer.vllm_config.parallel_config.decode_context_parallel_size = 1

    builder = object.__new__(TritonAttentionMetadataBuilder)
    builder.layer_names = ["layer.0"]
    builder.vllm_config = proposer.vllm_config
    builder.kv_cache_spec = object()
    builder.seq_threshold_3D = 4
    builder.num_par_softmax_segments = 64
    builder.softmax_segm_output = torch.empty((4, 1, 64, 4))
    builder.softmax_segm_max = torch.empty((4, 1, 64))
    builder.softmax_segm_expsum = torch.empty((4, 1, 64))

    def build_for_drafting(
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> TritonAttentionMetadata:
        del draft_index
        return TritonAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            seq_threshold_3D=builder.seq_threshold_3D,
            num_par_softmax_segments=builder.num_par_softmax_segments,
            softmax_segm_output=builder.softmax_segm_output,
            softmax_segm_max=builder.softmax_segm_max,
            softmax_segm_expsum=builder.softmax_segm_expsum,
            causal=common_attn_metadata.causal,
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
        )

    build_mock = mock.Mock(side_effect=build_for_drafting)
    builder.build_for_drafting = build_mock
    proposer.draft_attn_groups = [
        AttentionGroup(
            backend=TritonAttentionBackend,
            layer_names=builder.layer_names,
            kv_cache_spec=builder.kv_cache_spec,
            kv_cache_group_id=0,
            metadata_builders=[builder],
        )
    ]
    proposer._draft_attn_metadata_cache.clear()
    proposer._disabled_draft_attn_metadata_cache_keys.clear()
    proposer._draft_attn_metadata_cache_group_key = None
    proposer.metadata_template_builds = 0
    proposer.metadata_template_hits = 0
    proposer.metadata_template_fallbacks = 0
    proposer._initialize_draft_attn_metadata_cache_capability()
    return proposer, build_mock, builder


def _get_metadata(
    proposer: SpecDecodeBaseProposer,
    common: CommonAttentionMetadata,
    draft_index: int,
    phase: str,
    input_batch_size: int = 1,
    mode: CUDAGraphMode = CUDAGraphMode.PIECEWISE,
) -> tuple[list[object], dict[str, object]]:
    return proposer._get_draft_attn_metadata(
        common,
        draft_index=draft_index,
        phase=phase,
        input_batch_size=input_batch_size,
        cudagraph_runtime_mode=mode,
    )


def test_draft_attn_metadata_cache_refreshes_only_dynamic_scalar():
    proposer, build_mock = _make_proposer()
    common = _make_common_metadata()

    built = _get_metadata(proposer, common, 1, "remaining")
    common.seq_lens.add_(1)
    common.slot_mapping.add_(16)
    common.max_seq_len += 1
    cached = _get_metadata(proposer, common, 1, "remaining")

    assert cached is built
    assert build_mock.call_count == 1
    assert cached[0][0] is cached[1]["layer.0"]
    metadata = cast(FlashAttentionMetadata, cached[0][0])
    assert metadata.max_seq_len == common.max_seq_len
    assert metadata.seq_lens is common.seq_lens
    assert metadata.slot_mapping is common.slot_mapping
    assert torch.equal(metadata.seq_lens, torch.tensor([9], dtype=torch.int32))
    assert torch.equal(metadata.slot_mapping, torch.tensor([23]))
    assert proposer.get_draft_attn_metadata_cache_stats() == {
        "metadata_template_builds": 1,
        "metadata_template_hits": 1,
        "metadata_template_fallbacks": 0,
        "metadata_template_entries": 1,
    }


@pytest.mark.parametrize(
    "field",
    ["query_start_loc", "seq_lens", "block_table_tensor", "slot_mapping"],
)
def test_draft_attn_metadata_cache_disables_unstable_pointer(field: str):
    proposer, build_mock = _make_proposer()
    common = _make_common_metadata()
    _get_metadata(proposer, common, 1, "remaining")

    setattr(common, field, getattr(common, field).clone())
    rebuilt = _get_metadata(proposer, common, 1, "remaining")

    assert build_mock.call_count == 2
    metadata = cast(FlashAttentionMetadata, rebuilt[0][0])
    assert metadata.max_seq_len == common.max_seq_len
    stats = proposer.get_draft_attn_metadata_cache_stats()
    assert stats["metadata_template_builds"] == 1
    assert stats["metadata_template_hits"] == 0
    assert stats["metadata_template_fallbacks"] == 1
    assert stats["metadata_template_entries"] == 0

    _get_metadata(proposer, common, 1, "remaining")
    assert build_mock.call_count == 3
    assert proposer.metadata_template_fallbacks == 2


def test_draft_attn_metadata_cache_keys_k_phase_index_and_padding():
    proposer, build_mock = _make_proposer()
    common = _make_common_metadata()

    _get_metadata(proposer, common, 1, "remaining")
    _get_metadata(proposer, common, 1, "remaining")
    _get_metadata(proposer, common, 0, "first")
    proposer.num_speculative_tokens = 3
    _get_metadata(proposer, common, 1, "remaining")
    _get_metadata(proposer, common, 2, "remaining")
    _get_metadata(proposer, common, 2, "remaining", input_batch_size=2)

    assert build_mock.call_count == 5
    assert proposer.metadata_template_builds == 5
    assert proposer.metadata_template_hits == 1


def test_draft_attn_metadata_cache_keys_builder_capture_configuration():
    proposer, _ = _make_proposer()
    common = _make_common_metadata()
    builder = proposer.draft_attn_groups[0].get_metadata_builder()

    def cache_key():
        return proposer._draft_attn_metadata_cache_key_for(
            common,
            draft_index=1,
            phase="remaining",
            input_batch_size=1,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        )

    baseline = cache_key()
    for attr_name, changed_value in (
        ("use_full_cuda_graph", False),
        ("max_cudagraph_size", 8),
        ("max_num_splits", 8),
    ):
        original_value = getattr(builder, attr_name)
        setattr(builder, attr_name, changed_value)
        assert cache_key() != baseline
        setattr(builder, attr_name, original_value)

        delattr(builder, attr_name)
        assert cache_key() != baseline
        setattr(builder, attr_name, original_value)

    original_max_cudagraph_size = builder.max_cudagraph_size
    builder.max_cudagraph_size = None
    none_value_key = cache_key()
    del builder.max_cudagraph_size
    assert cache_key() != none_value_key
    builder.max_cudagraph_size = original_max_cudagraph_size

    with mock.patch(
        "vllm.v1.spec_decode.llm_base_proposer.envs.VLLM_BATCH_INVARIANT",
        new=not envs.VLLM_BATCH_INVARIANT,
    ):
        assert cache_key() != baseline


def test_draft_attn_metadata_cache_logs_periodic_cumulative_snapshots():
    proposer, _ = _make_proposer()
    common = _make_common_metadata()

    with (
        mock.patch(
            "vllm.v1.spec_decode.llm_base_proposer._DRAFT_ATTN_METADATA_LOG_INTERVAL",
            new=4,
        ),
        mock.patch("vllm.v1.spec_decode.llm_base_proposer.logger.info") as log_info,
    ):
        _get_metadata(proposer, common, 1, "remaining")
        for _ in range(3):
            _get_metadata(proposer, common, 1, "remaining")
        common.seq_lens = common.seq_lens.clone()
        _get_metadata(proposer, common, 1, "remaining")
        proposer.log_draft_attn_metadata_cache_stats("shutdown")

    assert [call.args[1] for call in log_info.call_args_list] == [
        "first-build",
        "first-hit",
        "periodic",
        "first-fallback",
        "shutdown",
    ]
    assert log_info.call_args_list[2].args[2:] == (1, 3, 0, 1)
    assert log_info.call_args_list[3].args[2:] == (1, 3, 1, 0)
    assert log_info.call_args_list[4].args[2:] == (1, 3, 1, 0)


@pytest.mark.parametrize("num_speculative_tokens", [2, 3])
def test_draft_attn_metadata_cache_uses_k_for_all_forwards(
    num_speculative_tokens: int,
):
    proposer, build_mock = _make_proposer(num_speculative_tokens)
    common = _make_common_metadata()

    for _ in range(2):
        _get_metadata(proposer, common, 0, "first")
        for draft_index in range(1, num_speculative_tokens):
            _get_metadata(proposer, common, draft_index, "remaining")

    assert build_mock.call_count == num_speculative_tokens
    assert proposer.metadata_template_builds == num_speculative_tokens
    assert proposer.metadata_template_hits == num_speculative_tokens


@pytest.mark.parametrize("num_speculative_tokens", [2, 3])
def test_triton_draft_attn_metadata_cache_uses_k_and_refreshes_dynamic_state(
    num_speculative_tokens: int,
):
    proposer, build_mock, builder = _make_triton_proposer(num_speculative_tokens)
    common = _make_common_metadata()
    built_by_index: dict[int, tuple[list[object], dict[str, object]]] = {}

    for draft_index in range(num_speculative_tokens):
        phase = "first" if draft_index == 0 else "remaining"
        built_by_index[draft_index] = _get_metadata(
            proposer, common, draft_index, phase
        )

    common.seq_lens.add_(1)
    common.slot_mapping.add_(16)
    common.max_seq_len += 1
    for draft_index in range(num_speculative_tokens):
        phase = "first" if draft_index == 0 else "remaining"
        cached = _get_metadata(proposer, common, draft_index, phase)
        assert cached is built_by_index[draft_index]
        metadata = cast(TritonAttentionMetadata, cached[0][0])
        assert type(metadata) is TritonAttentionMetadata
        assert metadata.max_seq_len == common.max_seq_len
        assert metadata.seq_lens is common.seq_lens
        assert metadata.slot_mapping is common.slot_mapping
        assert metadata.softmax_segm_output is builder.softmax_segm_output
        assert metadata.softmax_segm_max is builder.softmax_segm_max
        assert metadata.softmax_segm_expsum is builder.softmax_segm_expsum

    assert build_mock.call_count == num_speculative_tokens
    assert proposer.get_draft_attn_metadata_cache_stats() == {
        "metadata_template_builds": num_speculative_tokens,
        "metadata_template_hits": num_speculative_tokens,
        "metadata_template_fallbacks": 0,
        "metadata_template_entries": num_speculative_tokens,
    }


@pytest.mark.parametrize(
    "unsupported",
    ["derived_backend", "mixed_pair", "derived_builder", "dcp", "mode"],
)
def test_triton_draft_attn_metadata_cache_rejects_unsupported_capability(
    unsupported: str,
):
    proposer, build_mock, builder = _make_triton_proposer()
    mode = CUDAGraphMode.PIECEWISE
    if unsupported == "derived_backend":

        class DerivedTritonAttentionBackend(TritonAttentionBackend):
            pass

        proposer.draft_attn_groups[0].backend = DerivedTritonAttentionBackend
    elif unsupported == "mixed_pair":
        proposer.draft_attn_groups[0].backend = FlashAttentionBackend
    elif unsupported == "derived_builder":

        class DerivedTritonAttentionMetadataBuilder(TritonAttentionMetadataBuilder):
            pass

        derived_builder = object.__new__(DerivedTritonAttentionMetadataBuilder)
        derived_builder.__dict__.update(builder.__dict__)
        proposer.draft_attn_groups[0].metadata_builders = [derived_builder]
    elif unsupported == "dcp":
        builder.vllm_config.parallel_config.decode_context_parallel_size = 2
    else:
        assert unsupported == "mode"
        mode = CUDAGraphMode.FULL

    proposer._initialize_draft_attn_metadata_cache_capability()
    _get_metadata(proposer, _make_common_metadata(), 1, "remaining", mode=mode)

    assert build_mock.call_count == 1
    assert proposer.metadata_template_builds == 0
    assert proposer.metadata_template_hits == 0
    assert proposer.metadata_template_fallbacks == 1
    assert not proposer._draft_attn_metadata_cache


@pytest.mark.parametrize(
    "invalid_field",
    [
        "metadata_type",
        "use_cascade",
        "common_prefix_len",
        "cu_prefix_query_lens",
        "prefix_kv_lens",
        "suffix_kv_lens",
        "scheduler_metadata",
        "prefix_scheduler_metadata",
        "mm_prefix_range",
        "mm_prefix_range_tensor",
        "causal",
        "seq_threshold_3D",
        "num_par_softmax_segments",
        "softmax_segm_output",
        "softmax_segm_output_non_tensor",
        "softmax_segm_max",
        "softmax_segm_expsum",
    ],
)
def test_triton_draft_attn_metadata_cache_rejects_invalid_metadata(
    invalid_field: str,
):
    proposer, build_mock, _ = _make_triton_proposer()
    original_build = build_mock.side_effect
    assert callable(original_build)

    def build_invalid_metadata(*args, **kwargs):
        metadata = original_build(*args, **kwargs)
        if invalid_field == "metadata_type":

            class DerivedTritonAttentionMetadata(TritonAttentionMetadata):
                pass

            derived = object.__new__(DerivedTritonAttentionMetadata)
            derived.__dict__.update(metadata.__dict__)
            return derived
        field = invalid_field.removesuffix("_non_tensor")
        value = getattr(metadata, field)
        if invalid_field == "use_cascade":
            value = True
        elif invalid_field in (
            "cu_prefix_query_lens",
            "prefix_kv_lens",
            "suffix_kv_lens",
            "scheduler_metadata",
            "prefix_scheduler_metadata",
            "mm_prefix_range_tensor",
        ):
            value = torch.ones(1, dtype=torch.int32)
        elif invalid_field == "mm_prefix_range":
            value = {0: [(0, 1)]}
        elif invalid_field == "causal":
            value = False
        elif invalid_field.endswith("_non_tensor"):
            value = None
        elif isinstance(value, torch.Tensor):
            value = value.clone()
        else:
            value += 1
        setattr(
            metadata,
            field,
            value,
        )
        return metadata

    build_mock.side_effect = build_invalid_metadata
    _get_metadata(proposer, _make_common_metadata(), 1, "remaining")

    assert build_mock.call_count == 1
    assert proposer.metadata_template_builds == 0
    assert proposer.metadata_template_hits == 0
    assert proposer.metadata_template_fallbacks == 1
    assert not proposer._draft_attn_metadata_cache


@pytest.mark.parametrize(
    "field",
    ["query_start_loc", "seq_lens", "block_table_tensor", "slot_mapping"],
)
def test_triton_draft_attn_metadata_cache_disables_unstable_pointer(field: str):
    proposer, build_mock, _ = _make_triton_proposer()
    common = _make_common_metadata()
    _get_metadata(proposer, common, 1, "remaining")

    setattr(common, field, getattr(common, field).clone())
    _get_metadata(proposer, common, 1, "remaining")

    assert build_mock.call_count == 2
    assert proposer.metadata_template_builds == 1
    assert proposer.metadata_template_hits == 0
    assert proposer.metadata_template_fallbacks == 1
    assert not proposer._draft_attn_metadata_cache


@pytest.mark.parametrize(
    "unsupported",
    [
        "dcp",
        "mm",
        "causal",
        "backend",
        "builder",
        "model",
        "groups",
        "proposer_mm",
        "mrope",
        "dcp_metadata",
        "tree_causal",
    ],
)
def test_draft_attn_metadata_cache_falls_back_for_unsupported_state(
    unsupported: str,
):
    proposer, build_mock = _make_proposer()
    common = _make_common_metadata()
    builder = proposer.draft_attn_groups[0].get_metadata_builder()
    if unsupported == "dcp":
        builder.dcp_world_size = 2
    elif unsupported == "mm":
        common.mm_req_doc_ranges = {0: [(0, 1)]}
    elif unsupported == "causal":
        common.causal = False
    elif unsupported == "backend":

        class DerivedFlashAttentionBackend(FlashAttentionBackend):
            pass

        proposer.draft_attn_groups[0].backend = DerivedFlashAttentionBackend
    elif unsupported == "builder":

        class DerivedFlashAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
            pass

        derived_builder = object.__new__(DerivedFlashAttentionMetadataBuilder)
        derived_builder.__dict__.update(builder.__dict__)
        proposer.draft_attn_groups[0].metadata_builders = [derived_builder]
    elif unsupported == "model":
        proposer.model = object()
    elif unsupported == "groups":
        proposer.draft_attn_groups.append(proposer.draft_attn_groups[0])
    elif unsupported == "proposer_mm":
        proposer.supports_mm_inputs = True
    elif unsupported == "mrope":
        proposer.uses_mrope = True
    elif unsupported == "dcp_metadata":
        common.dcp_local_seq_lens = torch.ones(1, dtype=torch.int32)
    else:
        assert unsupported == "tree_causal"
        common.causal = torch.ones(1, dtype=torch.bool)

    proposer._initialize_draft_attn_metadata_cache_capability()

    _get_metadata(proposer, common, 1, "remaining")

    assert build_mock.call_count == (2 if unsupported == "groups" else 1)
    assert proposer.metadata_template_builds == 0
    assert proposer.metadata_template_hits == 0
    assert proposer.metadata_template_fallbacks == 1
    assert not proposer._draft_attn_metadata_cache


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_draft_attn_metadata_cache_falls_back_outside_piecewise(mode):
    proposer, build_mock = _make_proposer()
    common = _make_common_metadata()

    _get_metadata(proposer, common, 1, "remaining", mode=mode)

    assert build_mock.call_count == 1
    assert proposer.metadata_template_builds == 0
    assert proposer.metadata_template_fallbacks == 1


def test_draft_attn_metadata_cache_rejects_scheduler_metadata():
    proposer, build_mock = _make_proposer()
    common = _make_common_metadata()
    builder = proposer.draft_attn_groups[0].get_metadata_builder()
    builder._test_scheduler_metadata = torch.ones(1, dtype=torch.int32)

    _get_metadata(proposer, common, 1, "remaining")

    assert build_mock.call_count == 1
    assert proposer.metadata_template_builds == 0
    assert proposer.metadata_template_fallbacks == 1
    assert not proposer._draft_attn_metadata_cache


def test_draft_query_start_loc_cpu_is_cached_by_batch_size():
    proposer, _ = _make_proposer()

    first = proposer._get_draft_query_start_loc_cpu(2)
    second = proposer._get_draft_query_start_loc_cpu(2)
    other = proposer._get_draft_query_start_loc_cpu(3)

    assert first is second
    assert other is not first
    assert torch.equal(first, torch.tensor([0, 1, 2], dtype=torch.int32))
    assert torch.equal(other, torch.tensor([0, 1, 2, 3], dtype=torch.int32))


def test_propose_reaches_all_k_metadata_templates_with_piecewise_dispatch():
    proposer, build_mock = _make_proposer(num_speculative_tokens=3)
    common = _make_common_metadata()
    proposer._last_draft_probs = None
    proposer._share_mtp_indices = False
    proposer.allowed_attn_types = None
    proposer.block_size = 16
    proposer.pass_hidden_states_to_model = True
    proposer.arange = torch.arange(8, dtype=torch.int32)
    proposer.input_ids = torch.zeros(8, dtype=torch.int32)
    proposer.positions = torch.zeros(8, dtype=torch.int64)
    proposer.hidden_states = torch.zeros((8, 4), dtype=torch.float32)
    proposer._slot_mapping_buffer = torch.zeros(8, dtype=torch.int64)
    common.slot_mapping = proposer._slot_mapping_buffer[:1]
    second_common = CommonAttentionMetadata(
        query_start_loc=common.query_start_loc.view_as(common.query_start_loc),
        query_start_loc_cpu=common.query_start_loc_cpu,
        seq_lens=common.seq_lens.view_as(common.seq_lens),
        num_reqs=common.num_reqs,
        num_actual_tokens=common.num_actual_tokens,
        max_query_len=common.max_query_len,
        max_seq_len=common.max_seq_len,
        block_table_tensor=common.block_table_tensor.view_as(common.block_table_tensor),
        slot_mapping=common.slot_mapping.view_as(common.slot_mapping),
    )

    model = mock.MagicMock(spec=Eagle3LlamaForCausalLM)
    assert isinstance(model, Eagle3LlamaForCausalLM)
    model.combine_hidden_states.side_effect = lambda hidden_states: hidden_states
    model.side_effect = [
        (torch.zeros((1, 4)), torch.full((1, 4), step, dtype=torch.float32))
        for _ in range(2)
        for step in range(3)
    ]
    proposer.model = model
    proposer.set_inputs_first_pass = mock.Mock(
        side_effect=[
            (1, torch.tensor([0], dtype=torch.int64), common),
            (1, torch.tensor([0], dtype=torch.int64), second_common),
        ]
    )
    proposer.build_model_inputs_first_pass = mock.Mock(
        return_value=(
            {
                "input_ids": proposer.input_ids[:1],
                "positions": proposer.positions[:1],
                "inputs_embeds": None,
                "hidden_states": proposer.hidden_states[:1],
            },
            1,
        )
    )
    proposer._determine_batch_execution_and_padding = mock.Mock(
        return_value=(CUDAGraphMode.PIECEWISE, 1, None)
    )

    def update_positions(positions, common_attn_metadata, *args):
        del args
        common_attn_metadata.max_seq_len += 1
        common_attn_metadata.slot_mapping = proposer._slot_mapping_buffer[:1]
        return positions

    proposer._update_positions_dependent_metadata = mock.Mock(
        side_effect=update_positions
    )
    proposer._sample_draft_tokens = mock.Mock(
        side_effect=[
            (torch.tensor([10 + step]), None)
            for _ in range(2)
            for step in range(proposer.num_speculative_tokens)
        ]
    )

    with mock.patch(
        "vllm.v1.spec_decode.llm_base_proposer.set_forward_context",
        side_effect=lambda *args, **kwargs: nullcontext(),
    ) as forward_context:
        results = [
            proposer.propose(
                num_speculative_tokens=3,
                target_token_ids=torch.tensor([1], dtype=torch.int32),
                target_positions=torch.tensor([0], dtype=torch.int64),
                target_hidden_states=torch.zeros((1, 4)),
                next_token_ids=torch.tensor([2], dtype=torch.int32),
                token_indices_to_sample=None,
                common_attn_metadata=proposal_common,
                sampling_metadata=mock.MagicMock(),
            )
            for proposal_common in (common, second_common)
        ]

    assert all(torch.equal(result, torch.tensor([[10, 11, 12]])) for result in results)
    assert [call.kwargs["draft_index"] for call in build_mock.call_args_list] == [
        0,
        1,
        2,
    ]
    assert proposer._determine_batch_execution_and_padding.call_count == 4
    assert model.call_count == 6
    assert forward_context.call_count == 6
    assert all(
        call.kwargs["cudagraph_runtime_mode"] is CUDAGraphMode.PIECEWISE
        for call in forward_context.call_args_list
    )
    assert proposer.metadata_template_builds == 3
    assert proposer.metadata_template_hits == 3


@pytest.mark.parametrize("num_speculative_tokens", [2, 3])
def test_draft_chain_body_captures_all_serial_forwards(
    num_speculative_tokens: int,
):
    proposer, _, _ = _make_triton_proposer(num_speculative_tokens)
    proposer.vllm_config = SimpleNamespace()
    proposer.model_returns_tuple = mock.Mock(return_value=True)
    proposer.build_model_inputs_first_pass = mock.Mock(
        return_value=(
            {
                "input_ids": proposer.input_ids[:1],
                "positions": proposer.positions[:1],
                "inputs_embeds": None,
                "hidden_states": proposer.hidden_states[:1],
            },
            1,
        )
    )
    proposer.model = mock.Mock(
        side_effect=[
            (torch.zeros((1, 4)), torch.full((1, 4), step, dtype=torch.float32))
            for step in range(num_speculative_tokens)
        ]
    )
    proposer._greedy_sample = mock.Mock(
        side_effect=[
            torch.tensor([10 + step]) for step in range(num_speculative_tokens)
        ]
    )
    proposer._update_positions_dependent_metadata_for_graph = mock.Mock(
        side_effect=lambda positions, *_: positions
    )
    common = _make_common_metadata()

    with mock.patch(
        "vllm.v1.spec_decode.llm_base_proposer.set_forward_context",
        side_effect=lambda *args, **kwargs: nullcontext(),
    ) as forward_context:
        output = proposer._run_draft_chain_graph_body(
            num_speculative_tokens=num_speculative_tokens,
            first_input_batch_size=1,
            remaining_input_batch_size=1,
            token_indices_to_sample=torch.tensor([0]),
            num_rejected_tokens=torch.tensor([0], dtype=torch.int32),
            first_attn_metadata={},
            remaining_attn_metadata=[{} for _ in range(num_speculative_tokens - 1)],
            first_slot_mapping={},
            remaining_slot_mapping={},
            remaining_common_attn_metadata=common,
        )

    assert torch.equal(
        output,
        torch.tensor([[10 + step for step in range(num_speculative_tokens)]]),
    )
    assert proposer.model.call_count == num_speculative_tokens
    assert proposer._greedy_sample.call_count == num_speculative_tokens
    assert (
        proposer._update_positions_dependent_metadata_for_graph.call_count
        == num_speculative_tokens - 1
    )
    assert forward_context.call_count == num_speculative_tokens


def test_draft_chain_graph_key_separates_k_and_padded_sizes():
    proposer, _, _ = _make_triton_proposer()
    proposer._draft_chain_graph_capability_key = ("exact-triton",)
    proposer.max_model_len = 1024
    common = _make_common_metadata()

    k2 = proposer._draft_chain_graph_key_for(common, 2, 3, 4, 1)
    k3 = proposer._draft_chain_graph_key_for(common, 3, 4, 4, 1)
    other_padding = proposer._draft_chain_graph_key_for(common, 2, 3, 3, 1)

    assert k2 != k3
    assert k2 != other_padding


def test_draft_chain_graph_max_seq_legality_is_exact_triton_only():
    flash_proposer, _ = _make_proposer()
    triton_proposer, _, _ = _make_triton_proposer()

    with mock.patch(
        "vllm.v1.spec_decode.llm_base_proposer._DRAFT_CHAIN_CUDAGRAPH_ENABLED",
        new=True,
    ):
        flash_proposer._initialize_draft_chain_graph_capability()
        triton_proposer._initialize_draft_chain_graph_capability()

    assert flash_proposer._draft_chain_graph_capability_key is None
    assert triton_proposer._draft_chain_graph_capability_key is not None


def test_draft_chain_graph_is_disabled_by_default():
    proposer, _, _ = _make_triton_proposer()

    proposer._initialize_draft_chain_graph_capability()

    assert proposer._draft_chain_graph_capability_key is None


@pytest.mark.parametrize("num_speculative_tokens", [2, 3])
def test_draft_chain_graph_replay_prepares_and_reconciles_stable_state(
    num_speculative_tokens: int,
):
    proposer, _, _ = _make_triton_proposer(num_speculative_tokens)
    proposer._draft_chain_graph_capability_key = ("exact-triton",)
    proposer.max_model_len = 1024
    proposer.compilation_config = SimpleNamespace(
        max_cudagraph_capture_size=4,
        cudagraph_mode=CUDAGraphMode.FULL,
    )
    proposer.speculative_config.enforce_eager = False
    proposer._share_mtp_indices = False
    proposer._enable_probabilistic_draft_probs = False
    proposer._determine_batch_execution_and_padding = mock.Mock(
        side_effect=lambda num_tokens: (
            CUDAGraphMode.PIECEWISE,
            4 if num_tokens > 1 else 1,
            None,
        )
    )
    num_tokens = num_speculative_tokens + 1
    common = _make_common_metadata()
    common.query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32)
    common.query_start_loc_cpu = common.query_start_loc.clone()
    common.num_actual_tokens = num_tokens
    common.max_query_len = num_tokens
    common.slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
    common._seq_lens_cpu = torch.tensor([8], dtype=torch.int32)
    common._num_computed_tokens_cpu = torch.tensor([5], dtype=torch.int32)
    common.seq_lens_cpu_upper_bound = torch.tensor([8], dtype=torch.int32)
    key = proposer._draft_chain_graph_key_for(
        common,
        num_speculative_tokens,
        num_tokens,
        4,
        1,
    )
    output = torch.arange(num_speculative_tokens).view(1, -1)
    graph = mock.Mock()
    entry = SimpleNamespace(
        graph=graph,
        output=output,
        first_input_batch_size=4,
        token_indices_to_sample=torch.zeros(1, dtype=torch.int64),
        num_rejected_tokens=torch.zeros(1, dtype=torch.int32),
        seq_lens=common.seq_lens,
        block_table_tensor=common.block_table_tensor,
    )
    proposer._draft_chain_graphs[key] = entry

    with mock.patch(
        "vllm.v1.spec_decode.llm_base_proposer._DRAFT_CHAIN_CUDAGRAPH_ENABLED",
        new=True,
    ):
        result = proposer._try_replay_draft_chain_graph(
            num_speculative_tokens=num_speculative_tokens,
            num_tokens=num_tokens,
            token_indices_to_sample=torch.tensor([num_tokens - 1]),
            common_attn_metadata=common,
            mm_embed_inputs=None,
            num_rejected_tokens_gpu=torch.tensor([0], dtype=torch.int32),
        )

    assert result is output
    graph.replay.assert_called_once_with()
    assert torch.equal(entry.token_indices_to_sample, torch.tensor([num_tokens - 1]))
    assert torch.equal(
        proposer._slot_mapping_buffer[:num_tokens],
        torch.arange(num_tokens, dtype=torch.int64),
    )
    assert common.num_actual_tokens == 1
    assert common.max_query_len == 1
    assert common.max_seq_len == 8 + num_speculative_tokens - 1
    assert common._seq_lens_cpu is None
    assert common._num_computed_tokens_cpu is None
    assert proposer.draft_chain_graph_replays == 1
    assert proposer.draft_chain_graph_fallbacks == 0


def test_draft_chain_graph_falls_back_when_target_width_exceeds_capture_limit():
    proposer, _, _ = _make_triton_proposer(num_speculative_tokens=4)
    proposer._draft_chain_graph_capability_key = ("exact-triton",)
    proposer.max_model_len = 1024
    proposer.compilation_config = SimpleNamespace(
        max_cudagraph_capture_size=5,
        cudagraph_mode=CUDAGraphMode.FULL,
    )
    proposer.speculative_config.enforce_eager = False
    common = _make_common_metadata()
    common.query_start_loc = torch.tensor([0, 5], dtype=torch.int32)
    common.query_start_loc_cpu = torch.tensor([0, 5], dtype=torch.int32)
    common.num_actual_tokens = 5
    common.max_query_len = 5
    common.slot_mapping = torch.arange(5, dtype=torch.int64)
    proposer._determine_batch_execution_and_padding = mock.Mock(
        side_effect=lambda num_tokens: (
            CUDAGraphMode.PIECEWISE,
            5 if num_tokens > 1 else 1,
            None,
        )
    )

    with mock.patch(
        "vllm.v1.spec_decode.llm_base_proposer._DRAFT_CHAIN_CUDAGRAPH_ENABLED",
        new=True,
    ):
        assert (
            proposer._can_use_draft_chain_graph(
                common,
                num_speculative_tokens=4,
                first_num_actual_tokens=5,
                token_indices_to_sample=torch.tensor([4]),
                mm_embed_inputs=None,
            )
            is not None
        )
        proposer.compilation_config.max_cudagraph_capture_size = 4
        result = proposer._try_replay_draft_chain_graph(
            num_speculative_tokens=4,
            num_tokens=5,
            token_indices_to_sample=torch.tensor([4]),
            common_attn_metadata=common,
            mm_embed_inputs=None,
            num_rejected_tokens_gpu=torch.tensor([0], dtype=torch.int32),
        )

    assert not isinstance(result, torch.Tensor)
    assert proposer._determine_batch_execution_and_padding.call_count == 2
    assert proposer.draft_chain_graph_replays == 0
    assert proposer.draft_chain_graph_fallbacks == 1
    assert proposer.get_draft_chain_graph_stats(4) == {
        "draft_chain_graph_captures": 0,
        "draft_chain_graph_replays": 0,
        "draft_chain_graph_fallbacks": 1,
        "draft_chain_graph_entries": 0,
        "draft_chain_forward_events": 4,
    }


def test_draft_chain_graph_logs_forward_periodic_and_shutdown_snapshots():
    proposer, _ = _make_proposer()
    proposer._draft_chain_graphs[("k2",)] = SimpleNamespace(num_speculative_tokens=2)

    with (
        mock.patch(
            "vllm.v1.spec_decode.llm_base_proposer._DRAFT_ATTN_METADATA_LOG_INTERVAL",
            new=4,
        ),
        mock.patch(
            "vllm.v1.spec_decode.llm_base_proposer._DRAFT_CHAIN_CUDAGRAPH_ENABLED",
            new=True,
        ),
        mock.patch("vllm.v1.spec_decode.llm_base_proposer.logger.info") as log_info,
    ):
        proposer._record_draft_chain_graph_event("capture", num_speculative_tokens=2)
        proposer._record_draft_chain_graph_event(
            "replay", num_drafter_forwards=2, num_speculative_tokens=2
        )
        proposer._record_draft_chain_graph_event(
            "replay", num_drafter_forwards=2, num_speculative_tokens=2
        )
        proposer.log_draft_chain_graph_stats("shutdown")

    assert [call.args[1] for call in log_info.call_args_list] == [
        "first-capture",
        "first-replay",
        "periodic",
        "shutdown",
    ]
    assert log_info.call_args_list[-1].args[2:] == (2, True, 1, 2, 0, 1)


def test_draft_chain_graph_logs_k2_expected_counts_include_first_fallback():
    proposer, _ = _make_proposer(num_speculative_tokens=2)
    proposer._draft_chain_graphs[("k2",)] = SimpleNamespace(num_speculative_tokens=2)

    with (
        mock.patch(
            "vllm.v1.spec_decode.llm_base_proposer._DRAFT_CHAIN_CUDAGRAPH_ENABLED",
            new=True,
        ),
        mock.patch("vllm.v1.spec_decode.llm_base_proposer.logger.info") as log_info,
    ):
        proposer._record_draft_chain_graph_event("capture", num_speculative_tokens=2)
        proposer._record_draft_chain_graph_event(
            "fallback", num_drafter_forwards=2, num_speculative_tokens=2
        )
        for _ in range(8192):
            proposer._record_draft_chain_graph_event(
                "replay", num_drafter_forwards=2, num_speculative_tokens=2
            )
        proposer.log_draft_chain_graph_stats("shutdown", 2)

    assert log_info.call_args_list[-1].args[2:] == (2, True, 1, 8192, 1, 1)


def test_draft_chain_graph_disabled_shutdown_stats_are_zero():
    proposer, _ = _make_proposer(num_speculative_tokens=2)

    with mock.patch("vllm.v1.spec_decode.llm_base_proposer.logger.info") as log_info:
        proposer.log_draft_chain_graph_stats("shutdown")

    assert log_info.call_args.args[2:] == (2, False, 0, 0, 0, 0)
