# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only runner contracts, runnable without importing torch or vLLM.

Run directly with ``PYTHONDONTWRITEBYTECODE=1 python3 -B <this file>``.
The tests execute selected production AST nodes with rank/config stand-ins;
they do not exercise GPU model execution or communication.
"""

from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
SOURCE_PATH = REPO / "vllm/v1/worker/gpu_model_runner.py"
SOURCE = ast.parse(SOURCE_PATH.read_text())
RUNNER = next(
    node
    for node in SOURCE.body
    if isinstance(node, ast.ClassDef) and node.name == "GPUModelRunner"
)
METHODS = {node.name: node for node in RUNNER.body if isinstance(node, ast.FunctionDef)}


def load_method(name, namespace):
    method = copy.deepcopy(METHODS[name])
    method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(SOURCE_PATH), "exec"), namespace
    )
    return namespace[name]


def contains_attribute(node, name):
    return any(
        isinstance(child, ast.Attribute) and child.attr == name
        for child in ast.walk(node)
    )


class RunnerPPContracts(unittest.TestCase):
    def setUp(self):
        self.pp = SimpleNamespace(is_last_rank=False, world_size=2)
        self.namespace = {
            "get_pp_group": lambda: self.pp,
            "supports_eagle3": lambda model: model.supports_eagle3,
            "logger": SimpleNamespace(info=lambda *args: None),
        }

    @staticmethod
    def spec_config(method="eagle3", eagle_config=None):
        return SimpleNamespace(
            method=method,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(eagle_config=eagle_config)
            ),
        )

    def initialize_aux_flags(self, spec_config):
        body = METHODS["__init__"].body
        start = next(
            i
            for i, node in enumerate(body)
            if isinstance(node, ast.Assign)
            and contains_attribute(node.targets[0], "use_aux_hidden_state_outputs")
        )
        # Execute the contiguous production flag assignments and relay condition.
        nodes = body[start : start + 3]
        self.assertIsInstance(nodes[2], ast.If)
        runner = SimpleNamespace(speculative_config=spec_config)
        namespace = dict(self.namespace, self=runner)
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE_PATH), "exec"),
            namespace,
        )
        return runner

    def test_aux_relay_follows_draft_config_without_a_drafter(self):
        for config, expected in (
            (None, True),
            ({}, True),
            ({"use_aux_hidden_state": True}, True),
            ({"use_aux_hidden_state": False}, False),
        ):
            with self.subTest(eagle_config=config):
                runner = self.initialize_aux_flags(
                    self.spec_config(eagle_config=config)
                )
                self.assertEqual(runner.relay_aux_hidden_states, expected)
                self.assertFalse(runner.use_aux_hidden_state_outputs)
                self.assertFalse(hasattr(runner, "drafter"))

    def test_relay_is_disabled_on_last_rank_pp1_and_other_methods(self):
        for pp_size, last_rank, spec_config in (
            (1, True, self.spec_config()),
            (2, True, self.spec_config()),
            (2, False, self.spec_config(method="eagle")),
            (2, False, None),
        ):
            with self.subTest(pp_size=pp_size, last_rank=last_rank, config=spec_config):
                self.pp.world_size = pp_size
                self.pp.is_last_rank = last_rank
                runner = self.initialize_aux_flags(spec_config)
                self.assertFalse(runner.relay_aux_hidden_states)
                self.assertFalse(runner.use_aux_hidden_state_outputs)

    def test_speculative_feedback_is_limited_to_async_eagle3_pp(self):
        assignment = next(
            node
            for node in METHODS["__init__"].body
            if isinstance(node, ast.Assign)
            and contains_attribute(node.targets[0], "use_pp_async_spec_decode")
        )
        for pp_size, async_spec, method, broadcast, expected in (
            (2, True, "eagle3", False, True),
            (1, True, "eagle3", False, False),
            (2, False, "eagle3", False, False),
            (2, True, "eagle", False, False),
            (2, True, "eagle3", True, False),
        ):
            with self.subTest(pp_size=pp_size, async_spec=async_spec, method=method):
                runner = SimpleNamespace(
                    use_async_spec_decode=async_spec,
                    parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
                    speculative_config=self.spec_config(method=method),
                    broadcast_pp_output=broadcast,
                )
                exec(
                    compile(
                        ast.Module(body=[assignment], type_ignores=[]),
                        str(SOURCE_PATH),
                        "exec",
                    ),
                    {"self": runner},
                )
                self.assertEqual(runner.use_pp_async_spec_decode, expected)

    def make_runner(self, *, relay=True, outputs=False, pp_support=True, layers=None):
        selected: list[tuple[int, ...]] = []
        model = SimpleNamespace(
            model=SimpleNamespace(supports_aux_hidden_states_over_pp=pp_support),
            supports_eagle3=True,
            get_eagle3_default_aux_hidden_state_layers=lambda: (2, 30, 57),
            set_aux_hidden_state_layers=selected.append,
        )
        runner = SimpleNamespace(
            model=model,
            get_model=lambda: model,
            relay_aux_hidden_states=relay,
            use_aux_hidden_state_outputs=outputs,
            _get_eagle3_aux_layers_from_config=lambda: layers,
        )
        return runner, selected

    def test_early_and_last_ranks_select_identical_global_taps(self):
        setup = load_method("_setup_eagle3_aux_hidden_state_outputs", self.namespace)
        for layers, expected in ((None, (2, 30, 57)), ((0, 30, 60), (0, 30, 60))):
            for last_rank in (False, True):
                with self.subTest(layers=layers, last_rank=last_rank):
                    self.pp.is_last_rank = last_rank
                    runner, selected = self.make_runner(
                        relay=not last_rank, outputs=last_rank, layers=layers
                    )
                    setup(runner)
                    self.assertEqual(selected, [expected])
                    self.assertEqual(runner.use_aux_hidden_state_outputs, last_rank)

    def test_pp_capability_is_required_only_when_aux_is_enabled_and_pp_gt_one(self):
        setup = load_method("_setup_eagle3_aux_hidden_state_outputs", self.namespace)
        for last_rank in (False, True):
            self.pp.is_last_rank = last_rank
            runner, selected = self.make_runner(
                relay=not last_rank, outputs=last_rank, pp_support=False
            )
            with self.assertRaisesRegex(RuntimeError, "with pipeline parallelism"):
                setup(runner)
            self.assertEqual(selected, [])

        self.pp.world_size = 1
        runner, selected = self.make_runner(relay=False, outputs=True, pp_support=False)
        setup(runner)
        self.assertEqual(selected, [(2, 30, 57)])

        self.pp.world_size = 2
        runner, selected = self.make_runner(
            relay=False, outputs=False, pp_support=False
        )
        setup(runner)
        self.assertEqual(selected, [])

    def test_drafter_guards_short_circuit_on_early_rank(self):
        # Each rank guard must reject an early rank before reading self.drafter,
        # config-specific methods, or proposer classes (none exist in this scope).
        expected_counts = {
            "_build_attention_metadata": 3,
            "sample_tokens": 1,
            "_dummy_run": 1,
            "initialize_metadata_builders": 1,
            "_check_and_update_cudagraph_mode": 1,
            "initialize_kv_cache": 1,
        }
        runner = SimpleNamespace(speculative_config=self.spec_config())
        namespace = dict(
            self.namespace, self=runner, spec_config=runner.speculative_config
        )
        for name, expected_count in expected_counts.items():
            guards = [
                node.test
                for node in ast.walk(METHODS[name])
                if isinstance(node, ast.If)
                and contains_attribute(node.test, "is_last_rank")
                and contains_attribute(node, "drafter")
            ]
            with self.subTest(method=name):
                self.assertEqual(len(guards), expected_count)
                for guard in guards:
                    self.assertFalse(
                        eval(
                            compile(ast.Expression(guard), str(SOURCE_PATH), "eval"),
                            namespace,
                        )
                    )

    def test_proposal_entry_rejects_early_rank_before_accessing_drafter(self):
        first = METHODS["propose_draft_token_ids"].body[0]
        self.assertIsInstance(first, ast.Assert)
        with self.assertRaisesRegex(AssertionError, "last PP rank"):
            exec(
                compile(
                    ast.Module(body=[first], type_ignores=[]), str(SOURCE_PATH), "exec"
                ),
                self.namespace,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
