# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe as fi_moe
from vllm.model_executor.layers.fused_moe.activation import MoEActivation


@pytest.fixture(autouse=True)
def reset_mnt_call_proof():
    fi_moe._AUTORESEARCH_FLASHINFER_MOE_MNT_CALL_PROOF_EMITTED = False
    yield
    fi_moe._AUTORESEARCH_FLASHINFER_MOE_MNT_CALL_PROOF_EMITTED = False


def make_expert(ep_rank: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        quant_config=SimpleNamespace(use_nvfp4_w4a4=False),
        quant_dtype=None,
        weight_quant_dtype=None,
        use_deepseek_fp8_block_scale=False,
        gemm1_clamp_limit=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        out_dtype=torch.float32,
        tp_size=1,
        tp_rank=0,
        ep_size=4,
        ep_rank=ep_rank,
        max_capture_size=68,
        tune_max_num_tokens=272,
    )


def apply(expert: SimpleNamespace, input_dim0: int) -> None:
    hidden_states = torch.zeros((input_dim0, 8), dtype=torch.float32)
    output = torch.empty_like(hidden_states)
    topk_ids = torch.zeros((input_dim0, 1), dtype=torch.int64)
    topk_weights = torch.ones((input_dim0, 1), dtype=torch.float32)
    weights = torch.zeros((1, 8, 8), dtype=torch.uint8)
    fi_moe.FlashInferExperts.apply(
        expert,
        output,
        hidden_states,
        weights,
        weights,
        topk_weights,
        topk_ids,
        MoEActivation.SILU,
        4,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


@pytest.mark.parametrize("ep_rank", range(4))
def test_terminal_mnt_call_emits_post_return_proof(monkeypatch, ep_rank: int):
    calls = []
    logs = []

    def fake_flashinfer(**kwargs):
        calls.append(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(fi_moe, "flashinfer_cutlass_fused_moe", fake_flashinfer)
    monkeypatch.setattr(
        fi_moe.logger,
        "info",
        lambda message, *args: logs.append((message, args)),
    )

    apply(make_expert(ep_rank), 272)

    assert len(calls) == 1
    assert calls[0]["tune_max_num_tokens"] == 272
    assert calls[0]["use_fused_finalize"] is False
    assert len(logs) == 1
    assert logs[0][0] == "AUTORESEARCH_FLASHINFER_MOE_MNT_CALL_PROOF_JSON=%s"
    record = json.loads(logs[0][1][0])
    assert record == {
        "ep_rank": ep_rank,
        "ep_size": 4,
        "input_dim0": 272,
        "max_capture_size": 68,
        "schema_version": 2,
        "status": "flashinfer_python_binding_returned",
        "tp_rank": 0,
        "tp_size": 1,
        "tune_max_num_tokens": 272,
        "use_fused_finalize": False,
        "use_w4_group_scaling": False,
    }


def test_nonterminal_call_does_not_emit_proof(monkeypatch):
    logs = []
    monkeypatch.setattr(
        fi_moe, "flashinfer_cutlass_fused_moe", lambda **kwargs: kwargs["output"]
    )
    monkeypatch.setattr(fi_moe.logger, "info", lambda *args: logs.append(args))

    apply(make_expert(), 256)

    assert logs == []
    assert not fi_moe._AUTORESEARCH_FLASHINFER_MOE_MNT_CALL_PROOF_EMITTED


def test_terminal_call_emits_once(monkeypatch):
    logs = []
    monkeypatch.setattr(
        fi_moe, "flashinfer_cutlass_fused_moe", lambda **kwargs: kwargs["output"]
    )
    monkeypatch.setattr(fi_moe.logger, "info", lambda *args: logs.append(args))

    expert = make_expert()
    apply(expert, 272)
    apply(expert, 272)

    assert len(logs) == 1


def test_failed_binding_call_does_not_emit_proof(monkeypatch):
    logs = []

    def fail_flashinfer(**kwargs):
        raise RuntimeError("injected FlashInfer failure")

    monkeypatch.setattr(fi_moe, "flashinfer_cutlass_fused_moe", fail_flashinfer)
    monkeypatch.setattr(fi_moe.logger, "info", lambda *args: logs.append(args))

    with pytest.raises(RuntimeError, match="injected FlashInfer failure"):
        apply(make_expert(), 272)

    assert logs == []
    assert not fi_moe._AUTORESEARCH_FLASHINFER_MOE_MNT_CALL_PROOF_EMITTED
