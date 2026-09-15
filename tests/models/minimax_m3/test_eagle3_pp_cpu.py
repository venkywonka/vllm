# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only routing tests without importing vLLM, torch, or CUDA extensions.

Run directly: PYTHONDONTWRITEBYTECODE=1 python3 -B <this file> -v
AST extraction executes the real model/runner methods with small tensor and
distributed fakes. This checks routing, not kernels or GPU graph capture.
"""

from __future__ import annotations

import ast
import copy
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
MODEL = "vllm/models/minimax_m3/nvidia/model.py"
INTERFACES = "vllm/model_executor/models/interfaces.py"
UTILS = "vllm/model_executor/models/utils.py"
RUNNER = "vllm/v1/worker/gpu_model_runner.py"


def source_definition(path, name):
    tree = ast.parse((ROOT / path).read_text())
    return copy.deepcopy(next(n for n in tree.body if getattr(n, "name", None) == name))


def execute_definition(node, namespace):
    module = ast.Module(
        body=[*ast.parse("from __future__ import annotations").body, node],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), "<actual-vllm-source>", "exec"),
        namespace,
    )
    return namespace[node.name]


class Tensor:
    def __init__(self, value=0, shape=(3, 4), dtype="bf16", device="cpu", rows=None):
        self.rows = (
            rows if rows is not None else [[value] * shape[1] for _ in range(shape[0])]
        )
        self.dtype = dtype
        self.device = device

    @property
    def shape(self):
        return len(self.rows), len(self.rows[0]) if self.rows else 4

    @property
    def value(self):
        return self.rows[0][0]

    def __add__(self, other):
        other_rows = other.rows if isinstance(other, Tensor) else None
        rows = [
            [
                value + (other_rows[i][j] if other_rows else other)
                for j, value in enumerate(row)
            ]
            for i, row in enumerate(self.rows)
        ]
        return Tensor(dtype=self.dtype, device=self.device, rows=rows)

    def __getitem__(self, key):
        assert isinstance(key, slice)
        return Tensor(dtype=self.dtype, device=self.device, rows=self.rows[key])

    def copy_(self, other, non_blocking=False):
        assert self.shape == other.shape
        for destination, source in zip(self.rows, other.rows):
            destination[:] = source
        return self


class Module:
    forward: Callable[..., Any]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Embedding:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def __call__(self, input_ids):
        assert input_ids is not None
        self.calls += 1
        return input_ids


class MissingLayer:
    def __call__(self, *args, **kwargs):
        raise AssertionError("This PP stage executed a missing layer")


class Norm:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def __call__(self, hidden_states, residual):
        self.calls += 1
        if residual is not None:
            hidden_states = hidden_states + residual
        return hidden_states + 1000, None


class Layer:
    def __call__(self, positions, hidden_states, residual):
        if residual is not None:
            hidden_states = hidden_states + residual
        return hidden_states + 1, Tensor(10, shape=hidden_states.shape)


class Harness:
    def __init__(self):
        self.rank = 0
        self.world_size = 1
        self.gathered = []
        self.residual_scattered = False
        self.namespace: dict[str, Any] = {
            "torch": SimpleNamespace(
                zeros=lambda shape, dtype, device: Tensor(
                    shape=shape, dtype=dtype, device=device
                ),
                equal=lambda a, b: a.rows == b.rows,
            ),
            "nn": SimpleNamespace(Module=Module),
            "VocabParallelEmbedding": Embedding,
            "ParallelLMHead": Embedding,
            "PPMissingLayer": MissingLayer,
            "MiniMAXGemmaRMSNorm": Norm,
            "MiniMaxM3DecoderLayer": lambda **kwargs: Layer(),
            "make_layers": self.make_layers,
            "get_pp_group": self.pp_group,
            "get_tensor_model_parallel_world_size": lambda: 4,
            "get_tp_group": lambda: SimpleNamespace(all_gather=self.all_gather),
            "is_residual_scattered_for_sp": lambda config, count: (
                self.residual_scattered
            ),
            "fused_allreduce_gemma_rms_norm": lambda hidden, residual, norm: norm(
                hidden, residual
            ),
            "maybe_prefix": lambda prefix, name: f"{prefix}.{name}" if prefix else name,
            "LogitsProcessor": lambda *args: None,
            "logger": SimpleNamespace(info=lambda *args: None),
            "SupportsPP": type("SupportsPP", (), {}),
            "SupportsMultiModal": type("SupportsMultiModal", (), {}),
            "WeightsMapper": lambda **kwargs: None,
        }
        for path, name in (
            ("vllm/sequence.py", "IntermediateTensors"),
            (INTERFACES, "EagleModelMixin"),
            (INTERFACES, "SupportsEagle3"),
            (UTILS, "make_empty_intermediate_tensors_factory"),
            (MODEL, "MiniMaxM3Model"),
            (MODEL, "MiniMaxM3SparseForCausalLM"),
            (MODEL, "MiniMaxM3SparseForConditionalGeneration"),
        ):
            node = source_definition(path, name)
            if isinstance(node, ast.ClassDef):
                node.decorator_list = []
                if name == "SupportsEagle3":
                    node.bases = []
                    node.keywords = []
            execute_definition(node, self.namespace)
        runner = source_definition(RUNNER, "GPUModelRunner")
        self.receive = execute_definition(
            next(
                n
                for n in runner.body
                if getattr(n, "name", None) == "sync_and_gather_intermediate_tensors"
            ),
            self.namespace,
        )

    def pp_group(self):
        return SimpleNamespace(
            rank_in_group=self.rank,
            world_size=self.world_size,
            is_first_rank=self.rank == 0,
            is_last_rank=self.rank == self.world_size - 1,
        )

    def make_layers(self, count, layer_fn, prefix):
        start = self.rank * count // self.world_size
        end = (self.rank + 1) * count // self.world_size
        return start, end, [layer_fn(prefix=f"{prefix}.{i}") for i in range(count)]

    def all_gather(self, tensor, dim):
        self.gathered.append(tensor)
        return Tensor(rows=[row[:] for _ in range(4) for row in tensor.rows])

    def build(self, rank, world_size, taps=()):
        self.rank, self.world_size = rank, world_size
        config = SimpleNamespace(
            vocab_size=64, hidden_size=4, num_hidden_layers=60, rms_norm_eps=1e-6
        )
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=config),
            quant_config=None,
            parallel_config=SimpleNamespace(pipeline_parallel_size=world_size),
            speculative_config=None,
        )
        causal = self.namespace["MiniMaxM3SparseForCausalLM"](vllm_config=vllm_config)
        causal.set_aux_hidden_state_layers(taps)
        return causal

    def run(self, world_size, taps, *, inputs_embeds=False):
        stages = [self.build(rank, world_size, taps) for rank in range(world_size)]
        intermediates = []
        output = None
        for rank, stage in enumerate(stages):
            self.rank = rank
            output = stage(
                input_ids=Tensor(100) if rank == 0 else None,
                positions=None,
                intermediate_tensors=output,
                inputs_embeds=Tensor(100) if inputs_embeds else None,
            )
            if rank < world_size - 1:
                intermediates.append(output)
        return output, intermediates, stages


class PipelineProtocolTests(unittest.TestCase):
    def test_forward_signatures_support_intermediate_hidden_and_auxiliary_states(self):
        expected = ast.parse(
            "def forward(self, input_ids: Tensor | None, positions: Tensor, *, "
            "intermediate_tensors: IntermediateTensors | None) "
            "-> Tensor | IntermediateTensors | tuple[Tensor, list[Tensor]]: ..."
        ).body[0]
        assert isinstance(expected, ast.FunctionDef)
        assert expected.returns is not None
        for name in ("SupportsPP", "_SupportsPPType"):
            with self.subTest(protocol=name):
                protocol = source_definition(INTERFACES, name)
                assert isinstance(protocol, ast.ClassDef)
                forward = next(
                    node
                    for node in protocol.body
                    if isinstance(node, ast.FunctionDef) and node.name == "forward"
                )
                assert forward.returns is not None
                self.assertEqual(ast.dump(forward.args), ast.dump(expected.args))
                self.assertEqual(ast.dump(forward.returns), ast.dump(expected.returns))


class Eagle3PipelineTests(unittest.TestCase):
    def setUp(self):
        self.harness = Harness()

    def test_default_taps_for_sixty_layers(self):
        self.assertEqual(
            self.harness.build(0, 1).get_eagle3_default_aux_hidden_state_layers(),
            (2, 30, 57),
        )

    def test_pp1_pp2_pp3_forward_and_tap_order_match(self):
        taps = (57, 2, 30, 2)
        for world_size in (1, 2, 3):
            with self.subTest(world_size=world_size):
                (hidden, auxiliary), _, stages = self.harness.run(world_size, taps)
                self.assertEqual(hidden.value, 1760)
                self.assertEqual(
                    [tensor.value for tensor in auxiliary], [122, 430, 727]
                )
                self.assertEqual(stages[0].model.embed_tokens.calls, 1)
                for stage in stages[1:]:
                    self.assertIsInstance(stage.model.embed_tokens, MissingLayer)
                for stage in stages[:-1]:
                    self.assertIsInstance(stage.model.norm, MissingLayer)
                    self.assertIsInstance(stage.lm_head, MissingLayer)
                self.assertEqual(stages[-1].model.norm.calls, 1)

    def test_boundary_tap_belongs_to_earlier_stage(self):
        (_, auxiliary), intermediates, _ = self.harness.run(2, (2, 30, 57))
        stage_zero = intermediates[0]
        self.assertEqual(
            set(stage_zero.tensors),
            {
                "hidden_states",
                "residual",
                "aux_hidden_states_2",
                "aux_hidden_states_30",
            },
        )
        self.assertIs(stage_zero["aux_hidden_states_30"], auxiliary[1])

    def test_embedding_and_final_taps_preserve_pp1_order(self):
        for world_size in (1, 2, 3):
            with self.subTest(world_size=world_size):
                (_, auxiliary), intermediates, _ = self.harness.run(
                    world_size, (60, 30, 0, 20, 40, 0), inputs_embeds=True
                )
                self.assertEqual(
                    [tensor.value for tensor in auxiliary], [100, 320, 430, 540, 760]
                )
                if intermediates:
                    self.assertIs(intermediates[0]["aux_hidden_states_0"], auxiliary[0])

    def test_no_auxiliary_outputs_preserves_plain_forward(self):
        for world_size in (1, 2):
            with self.subTest(world_size=world_size):
                hidden, intermediates, stages = self.harness.run(world_size, ())
                self.assertEqual(hidden.value, 1760)
                for stage in stages[1:]:
                    self.assertIsInstance(stage.model.embed_tokens, MissingLayer)
                for intermediate in intermediates:
                    self.assertEqual(
                        set(intermediate.tensors), {"hidden_states", "residual"}
                    )

    def test_factory_rebuilt_after_setter_and_both_wrappers_delegate(self):
        for rank, world_size in ((0, 1), (0, 2), (1, 2)):
            with self.subTest(rank=rank, world_size=world_size):
                causal = self.harness.build(rank, world_size)
                wrapper_cls = self.harness.namespace[
                    "MiniMaxM3SparseForConditionalGeneration"
                ]
                wrapper = wrapper_cls.__new__(wrapper_cls)
                wrapper.language_model = causal
                factory = wrapper.make_empty_intermediate_tensors
                self.assertEqual(
                    set(factory(5, "bf16", "cpu").tensors),
                    {"hidden_states", "residual"},
                )
                wrapper.set_aux_hidden_state_layers((57, 2, 30))
                expected = {
                    "hidden_states",
                    "residual",
                    "aux_hidden_states_2",
                    "aux_hidden_states_30",
                }
                if world_size == 1 or rank == 1:
                    expected.add("aux_hidden_states_57")
                for owner in (causal.model, causal, wrapper):
                    tensors = owner.make_empty_intermediate_tensors(5, "bf16", "cpu")
                    self.assertEqual(set(tensors.tensors), expected)
                    for tensor in tensors.tensors.values():
                        self.assertEqual(tensor.shape, (5, 4))
                        self.assertEqual((tensor.dtype, tensor.device), ("bf16", "cpu"))
                wrapper.set_aux_hidden_state_layers((0,))
                self.assertEqual(
                    set(
                        wrapper.make_empty_intermediate_tensors(
                            5, "bf16", "cpu"
                        ).tensors
                    ),
                    {"hidden_states", "residual", "aux_hidden_states_0"},
                )

    def test_conditional_wrapper_forwards_intermediate_tensors(self):
        causal = self.harness.build(1, 2, (2, 30, 57))
        wrapper_cls = self.harness.namespace["MiniMaxM3SparseForConditionalGeneration"]
        wrapper = wrapper_cls.__new__(wrapper_cls)
        wrapper.language_model = causal
        intermediate_cls = self.harness.namespace["IntermediateTensors"]
        incoming = intermediate_cls(
            {
                "hidden_states": Tensor(420),
                "residual": Tensor(10),
                "aux_hidden_states_2": Tensor(122),
                "aux_hidden_states_30": Tensor(430),
            }
        )
        hidden, auxiliary = wrapper(None, None, intermediate_tensors=incoming)
        self.assertEqual(hidden.value, 1760)
        self.assertEqual([tensor.value for tensor in auxiliary], [122, 430, 727])

    def test_runner_receive_copies_auxiliary_keys_and_only_gathers_residual(self):
        causal = self.harness.build(1, 2, (2, 30, 57))
        intermediate_cls = self.harness.namespace["IntermediateTensors"]
        incoming = intermediate_cls(
            {
                "hidden_states": Tensor(420, (4, 4)),
                "residual": Tensor(10, (1, 4)),
                "aux_hidden_states_2": Tensor(122, (4, 4)),
                "aux_hidden_states_30": Tensor(430, (4, 4)),
            }
        )
        runner = SimpleNamespace(
            intermediate_tensors=causal.make_empty_intermediate_tensors(
                8, "bf16", "cpu"
            ),
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(tensor_parallel_size=4)
            ),
        )
        self.harness.residual_scattered = True
        received = self.harness.receive(runner, 4, incoming, sync_self=True)
        self.assertEqual(len(self.harness.gathered), 1)
        self.assertEqual(self.harness.gathered[0].value, 10)
        for key, expected in (
            ("hidden_states", 420),
            ("residual", 10),
            ("aux_hidden_states_2", 122),
            ("aux_hidden_states_30", 430),
        ):
            self.assertEqual(received[key].shape, (4, 4))
            self.assertEqual(received[key].value, expected)
            self.assertEqual(runner.intermediate_tensors[key][4:].value, 0)
        self.assertEqual(received["aux_hidden_states_57"].value, 0)
        hidden, auxiliary = causal(None, None, intermediate_tensors=received)
        self.assertEqual(hidden.value, 1760)
        self.assertEqual([tensor.value for tensor in auxiliary], [122, 430, 727])
        self.assertIs(auxiliary[0], received["aux_hidden_states_2"])
        self.assertIs(auxiliary[1], received["aux_hidden_states_30"])


if __name__ == "__main__":
    unittest.main()
