"""Stdlib-only C8 contracts; run with .venv/bin/python -B <file> -v.

Executes extracted source without importing torch or Triton. These tests check
host decisions and launch-index arithmetic, not GPU kernel correctness.
"""

from __future__ import annotations

import ast
import os
import unittest
from collections.abc import Callable
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "vllm/v1/attention/backends/triton_attn.py"
ATTENTION = ROOT / "vllm/v1/attention/ops/triton_unified_attention.py"
UBATCH = ROOT / "vllm/v1/worker/ubatch_utils.py"
FLAG = "VLLM_TRITON_ATTN_PREFILL_REORDER"


def definition(path, name, parent=None):
    tree = ast.parse(path.read_text())
    if parent is not None:
        tree = next(node for node in tree.body if getattr(node, "name", None) == parent)
    return next(node for node in tree.body if getattr(node, "name", None) == name)


def execute(nodes, namespace):
    module = ast.Module(
        body=[*ast.parse("from __future__ import annotations").body, *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "<vllm-source>", "exec"), namespace)


def extract(path, name, namespace, parent=None):
    node = definition(path, name, parent)
    node.decorator_list = []
    execute([node], namespace)
    return namespace[name]


def assigns(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets
    )


class CpuVector:
    def __init__(self, values, device="cpu"):
        self.values = list(values)
        self.device = SimpleNamespace(type=device)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        value = self.values[key]
        return CpuVector(value, self.device.type) if isinstance(key, slice) else value

    def __setitem__(self, key, value):
        self.values[key] = value.values if isinstance(value, CpuVector) else value

    def __sub__(self, other):
        other_values = (
            other.values if isinstance(other, CpuVector) else [other] * len(self)
        )
        return CpuVector(
            [value - rhs for value, rhs in zip(self.values, other_values)],
            self.device.type,
        )

    def any(self):
        return SimpleNamespace(item=lambda: any(self.values))

    def max(self):
        return max(self.values)

    def clone(self):
        return CpuVector(self.values, self.device.type)


class PrefillReorderMetadataTests(unittest.TestCase):
    def setUp(self):
        namespace = {"TritonAttentionMetadata": SimpleNamespace}
        build = extract(BACKEND, "build", namespace, "TritonAttentionMetadataBuilder")
        self.capture = extract(
            BACKEND,
            "build_for_cudagraph_capture",
            namespace,
            "TritonAttentionMetadataBuilder",
        )
        self.builder = SimpleNamespace(
            reorder_causal_prefill=True,
            seq_threshold_3D=128,
            num_par_softmax_segments=64,
            softmax_segm_output=None,
            softmax_segm_max=None,
            softmax_segm_expsum=None,
        )
        self.builder.build = MethodType(build, self.builder)
        self.common = SimpleNamespace(
            num_reqs=2,
            num_actual_tokens=8,
            max_query_len=4,
            max_seq_len=16,
            query_start_loc=object(),
            seq_lens=SimpleNamespace(fill_=Mock()),
            block_table_tensor=object(),
            slot_mapping=object(),
            causal=True,
            mm_req_doc_ranges=None,
            is_prefilling=CpuVector([True, False]),
        )

    def test_environment_is_default_off_and_explicitly_enabled(self):
        enabled = extract(BACKEND, "_prefill_reorder_enabled", {"os": os})
        for value, expected in ((None, False), ("0", False), ("1", True), ("2", True)):
            with self.subTest(value=value), patch.dict(os.environ, clear=True):
                if value is not None:
                    os.environ[FLAG] = value
                self.assertIs(enabled(), expected)

    def test_invalid_environment_value_is_not_silently_enabled(self):
        enabled = extract(BACKEND, "_prefill_reorder_enabled", {"os": os})
        with patch.dict(os.environ, {FLAG: "true"}), self.assertRaises(ValueError):
            enabled()

    def test_metadata_default_and_builder_environment_binding(self):
        metadata = definition(BACKEND, "TritonAttentionMetadata")
        field = next(
            node
            for node in metadata.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "reorder_causal_prefill"
        )
        assert field.value is not None
        self.assertIs(ast.literal_eval(field.value), False)
        initializer = definition(BACKEND, "__init__", "TritonAttentionMetadataBuilder")
        bindings = [
            node
            for node in ast.walk(initializer)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "reorder_causal_prefill"
                for target in node.targets
            )
        ]
        self.assertEqual(len(bindings), 1)
        self.assertEqual(ast.unparse(bindings[0].value), "_prefill_reorder_enabled()")

    def test_host_eligibility_requires_real_active_prefill(self):
        cases = (
            (True, False, 4, [True, False], True),
            (True, False, 4, [False, True], True),
            (True, False, 4, [False, False], False),
            (True, False, 4, [False, False, True], False),
            (True, False, 4, None, False),
            (True, False, 1, [True, True], False),
            (True, True, 4, [True, True], False),
            (False, False, 4, [True, True], False),
        )
        for enabled, fast, query_len, phases, expected in cases:
            with self.subTest(
                enabled=enabled, fast=fast, query=query_len, phases=phases
            ):
                self.builder.reorder_causal_prefill = enabled
                self.common.max_query_len = query_len
                self.common.is_prefilling = (
                    None if phases is None else CpuVector(phases)
                )
                metadata = self.builder.build(0, self.common, fast_build=fast)
                self.assertIs(metadata.reorder_causal_prefill, expected)

    def test_device_phase_requires_cpu_without_decode_or_fast_build_sync(self):
        self.common.is_prefilling = CpuVector([True, True], device="cuda")
        with self.assertRaisesRegex(AssertionError, "must be a CPU tensor"):
            self.builder.build(0, self.common)
        for enabled, fast, query_len in (
            (False, False, 4),
            (True, True, 4),
            (True, False, 1),
        ):
            with self.subTest(enabled=enabled, fast=fast, query=query_len):
                self.builder.reorder_causal_prefill = enabled
                self.common.max_query_len = query_len
                metadata = self.builder.build(0, self.common, fast_build=fast)
                self.assertFalse(metadata.reorder_causal_prefill)

    def test_full_graph_capture_disables_reordering(self):
        self.assertTrue(self.builder.build(0, self.common).reorder_causal_prefill)
        captured = self.capture(self.builder, self.common)
        self.assertFalse(captured.reorder_causal_prefill)
        self.common.seq_lens.fill_.assert_called_once_with(1)

    def test_forward_passes_the_metadata_decision_to_wrapper(self):
        forward = definition(BACKEND, "forward", "TritonAttentionImpl")
        call = next(
            node
            for node in ast.walk(forward)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "unified_attention"
        )
        value = next(
            keyword.value
            for keyword in call.keywords
            if keyword.arg == "reorder_causal_prefill"
        )
        self.assertEqual(ast.unparse(value), "attn_metadata.reorder_causal_prefill")


class PrefillReorderLaunchTests(unittest.TestCase):
    def setUp(self):
        self.wrapper = definition(ATTENTION, "unified_attention")
        self.kernel = definition(ATTENTION, "kernel_unified_attention")
        self.eligibility = next(
            node
            for node in self.wrapper.body
            if assigns(node, "reorder_causal_prefill")
        )
        self.flags = dict(
            reorder_causal_prefill=True,
            use_causal=True,
            use_per_seq_causal=False,
            use_mm_prefix=False,
            sliding_window_val=-1,
            chunk_lookback=-1,
            use_3d=False,
        )

    def test_wrapper_excludes_each_incompatible_kernel_mode(self):
        exclusions = dict(
            reorder_causal_prefill=False,
            use_causal=False,
            use_per_seq_causal=True,
            use_mm_prefix=True,
            sliding_window_val=32,
            chunk_lookback=0,
            use_3d=True,
        )
        execute([self.eligibility], self.flags)
        self.assertTrue(self.flags["reorder_causal_prefill"])
        for name, value in exclusions.items():
            with self.subTest(flag=name):
                flags = self.flags | {name: value}
                execute([self.eligibility], flags)
                self.assertFalse(flags["reorder_causal_prefill"])

    def test_grid_collapses_only_for_eligible_2d_attention(self):
        grid_selection = next(
            node
            for node in self.wrapper.body
            if isinstance(node, ast.If)
            and any(assigns(child, "grid") for child in node.body)
        )
        cases: tuple[tuple[dict[str, bool | int], tuple[int, ...]], ...] = (
            ({}, (28,)),
            ({"sliding_window_val": 0}, (28,)),
            ({"reorder_causal_prefill": False}, (7, 4)),
            ({"use_causal": False}, (7, 4)),
            ({"use_3d": True}, (7, 4, 16)),
        )
        for overrides, expected in cases:
            with self.subTest(overrides=overrides):
                flags = (
                    self.flags
                    | overrides
                    | dict(
                        total_num_q_blocks=7,
                        num_kv_heads=4,
                        num_par_softmax_segments=16,
                        TILE_SIZE_PREFILL=32,
                        TILE_SIZE_DECODE=16,
                    )
                )
                execute([self.eligibility, grid_selection], flags)
                self.assertEqual(flags["grid"], expected)

    def test_kernel_rejects_incompatible_modes_when_enabled(self):
        guard = next(
            node.value.args[0]
            for node in self.kernel.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "static_assert"
            and "REORDER_CAUSAL_PREFILL" in ast.unparse(node)
        )
        expression = compile(ast.Expression(guard), str(ATTENTION), "eval")
        flags = dict(
            REORDER_CAUSAL_PREFILL=True,
            USE_CAUSAL=True,
            USE_PER_SEQ_CAUSAL=False,
            USE_MM_PREFIX=False,
            SLIDING_WINDOW=-1,
            CHUNK_LOOKBACK=-1,
            IS_3D=False,
        )
        self.assertTrue(eval(expression, flags))
        for name, value in dict(
            USE_CAUSAL=False,
            USE_PER_SEQ_CAUSAL=True,
            USE_MM_PREFIX=True,
            SLIDING_WINDOW=1,
            CHUNK_LOOKBACK=0,
            IS_3D=True,
        ).items():
            with self.subTest(flag=name):
                incompatible = flags | {name: value}
                self.assertFalse(eval(expression, incompatible))
                incompatible["REORDER_CAUSAL_PREFILL"] = False
                self.assertTrue(eval(expression, incompatible))

    def test_linear_grid_visits_each_query_block_and_head_once(self):
        mapping = next(
            node
            for node in self.kernel.body
            if isinstance(node, ast.If)
            and any(assigns(child, "linear_program_idx") for child in node.body)
        )
        pairs = []
        for program in range(28):
            flags = dict(
                REORDER_CAUSAL_PREFILL=True,
                num_query_heads=32,
                num_queries_per_kv=8,
                tl=SimpleNamespace(program_id=lambda axis, index=program: index),
            )
            execute([mapping], flags)
            pairs.append((flags["q_block_global_idx"], flags["kv_head_idx"]))
        self.assertEqual(
            pairs, [(block, head) for block in range(7) for head in range(4)]
        )
        flags.update(
            REORDER_CAUSAL_PREFILL=False,
            tl=SimpleNamespace(program_id=lambda axis: (5, 3)[axis]),
        )
        execute([mapping], flags)
        self.assertEqual((flags["q_block_global_idx"], flags["kv_head_idx"]), (5, 3))

    def test_local_reversal_preserves_ragged_blocks_and_rejects_padding(self):
        padding_index = next(
            index
            for index, node in enumerate(self.kernel.body)
            if isinstance(node, ast.If)
            and "q_block_local_idx * BLOCK_Q" in ast.unparse(node.test)
        )
        remap = ast.parse(
            "def remap(q_block_local_idx, cur_batch_query_len, "
            "BLOCK_Q, REORDER_CAUSAL_PREFILL): pass"
        ).body[0]
        assert isinstance(remap, ast.FunctionDef)
        remap.body = [
            *self.kernel.body[padding_index : padding_index + 2],
            ast.Return(value=ast.Name(id="q_block_local_idx", ctx=ast.Load())),
        ]
        namespace: dict[str, Callable[..., int | None]] = {
            "cdiv_fn": lambda numerator, denominator: (numerator + denominator - 1)
            // denominator
        }
        execute([remap], namespace)
        for query_len in (1, 16, 17, 63, 64, 65, 129):
            blocks = (query_len + 15) // 16
            for enabled in (False, True):
                with self.subTest(query_len=query_len, enabled=enabled):
                    visited = [
                        namespace["remap"](block, query_len, 16, enabled)
                        for block in range(blocks + 1)
                    ]
                    expected = list(range(blocks))
                    if enabled:
                        expected.reverse()
                    self.assertEqual(visited, [*expected, None])

    def test_wrapper_forwards_the_qualified_flag_to_kernel(self):
        values = [
            keyword.value
            for node in ast.walk(self.wrapper)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "REORDER_CAUSAL_PREFILL"
        ]
        self.assertEqual(len(values), 1)
        self.assertEqual(ast.unparse(values[0]), "reorder_causal_prefill")


class PrefillReorderMicrobatchTests(unittest.TestCase):
    def test_phase_slicing_preserves_absence_and_partial_request_phase(self):
        namespace = {
            "CommonAttentionMetadata": SimpleNamespace,
            "torch": SimpleNamespace(
                abs=lambda vector: CpuVector([abs(value) for value in vector.values]),
                max=lambda vector: SimpleNamespace(item=vector.max),
            ),
        }
        extract(UBATCH, "slice_query_start_locs", namespace)
        make_slice = extract(UBATCH, "_make_metadata_with_slice", namespace)
        for phases in (None, [True, False, True]):
            for request_slice, token_slice in (
                (slice(1, 3), slice(3, 9)),
                (slice(0, 2), slice(1, 4)),
                (slice(2, 3), slice(5, 9)),
            ):
                with self.subTest(
                    phases=phases, requests=request_slice, tokens=token_slice
                ):
                    common = SimpleNamespace(
                        query_start_loc_cpu=CpuVector([0, 3, 5, 9]),
                        query_start_loc=CpuVector([0, 3, 5, 9]),
                        seq_lens=CpuVector([13, 10, 14]),
                        _seq_lens_cpu=CpuVector([13, 10, 14]),
                        seq_lens_cpu_upper_bound=CpuVector([13, 10, 14]),
                        _num_computed_tokens_cpu=CpuVector([10, 8, 10]),
                        max_seq_len=32,
                        max_query_len=4,
                        block_table_tensor=CpuVector([0, 1, 2]),
                        slot_mapping=CpuVector(range(9)),
                        is_prefilling=None if phases is None else CpuVector(phases),
                    )
                    sliced = make_slice(
                        SimpleNamespace(
                            request_slice=request_slice,
                            token_slice=token_slice,
                            is_empty=lambda: False,
                        ),
                        common,
                    )
                    if phases is None:
                        self.assertIsNone(sliced.is_prefilling)
                    else:
                        self.assertEqual(
                            sliced.is_prefilling.values, phases[request_slice]
                        )
                        self.assertEqual(sliced.is_prefilling.device.type, "cpu")
                        self.assertEqual(common.is_prefilling.values, phases)
                    self.assertEqual(
                        sliced.num_reqs, request_slice.stop - request_slice.start
                    )
                    self.assertEqual(sliced.max_seq_len, 32)
                    self.assertEqual(common.seq_lens.values, [13, 10, 14])


if __name__ == "__main__":
    unittest.main()
