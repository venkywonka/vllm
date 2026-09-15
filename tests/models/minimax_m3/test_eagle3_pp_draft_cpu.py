# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only contracts using source methods with framework dependencies stubbed.

Run with python3 -B tests/models/minimax_m3/test_eagle3_pp_draft_cpu.py.
No torch/vLLM import, model loading, or accelerator calls are performed.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]


def load_method(path, class_name, method_name, namespace):
    tree = ast.parse((REPO / path).read_text())
    cls = next(node for node in tree.body if getattr(node, "name", None) == class_name)
    method = next(
        node for node in cls.body if getattr(node, "name", None) == method_name
    )
    method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


class DraftPPContractTests(unittest.TestCase):
    def test_draft_pp_is_one_and_target_tp_equality_is_valid(self):
        namespace = {"ParallelConfig": SimpleNamespace}
        make_config = load_method(
            "vllm/config/speculative.py",
            "SpeculativeConfig",
            "create_draft_parallel_config",
            namespace,
        )
        verify_tp = load_method(
            "vllm/config/speculative.py",
            "SpeculativeConfig",
            "_verify_and_get_draft_tp",
            {},
        )
        for pp_size in (1, 2):
            with self.subTest(pp_size=pp_size):
                target = SimpleNamespace(
                    pipeline_parallel_size=pp_size,
                    tensor_parallel_size=4,
                    distributed_executor_backend="mp",
                    max_parallel_loading_workers=None,
                    disable_custom_all_reduce=True,
                    ray_workers_use_nsight=False,
                    placement_group=None,
                )
                draft = make_config(target, 4)
                self.assertEqual(draft.pipeline_parallel_size, 1)
                self.assertEqual(draft.tensor_parallel_size, 4)
                self.assertEqual(target.pipeline_parallel_size, pp_size)
                draft_hf_config = SimpleNamespace(model_type="llama")
                self.assertEqual(verify_tp(target, 4, draft_hf_config), 4)
                self.assertEqual(verify_tp(target, None, draft_hf_config), 4)
                with self.assertRaises(ValueError):
                    verify_tp(target, 8, draft_hf_config)

    def test_draft_layer_prefix_uses_total_target_layers(self):
        class FakeModule:
            def __init__(self):
                pass

        namespace = {
            "nn": SimpleNamespace(
                Module=FakeModule, Parameter=lambda value, **kwargs: value
            ),
            "torch": SimpleNamespace(zeros=lambda *args, **kwargs: (), long="long"),
            "LlamaModel": lambda **kwargs: SimpleNamespace(**kwargs),
            "ParallelLMHead": lambda *args, **kwargs: None,
            "LogitsProcessor": lambda *args, **kwargs: None,
            "get_draft_quant_config": lambda config: None,
            "maybe_prefix": lambda prefix, name: f"{prefix}.{name}" if prefix else name,
        }
        initialize = load_method(
            "vllm/model_executor/models/llama_eagle3.py",
            "Eagle3LlamaForCausalLM",
            "__init__",
            namespace,
        )
        for pp_size in (1, 2):
            with self.subTest(pp_size=pp_size):
                draft_hf_config = SimpleNamespace(vocab_size=200064, hidden_size=6144)
                vllm_config = SimpleNamespace(
                    speculative_config=SimpleNamespace(
                        draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
                        parallel_drafting=False,
                    ),
                    model_config=SimpleNamespace(
                        get_total_num_hidden_layers=lambda: 60,
                        get_num_layers=lambda config: (
                            60 // config.pipeline_parallel_size
                        ),
                    ),
                    parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
                )
                model = SimpleNamespace()
                initialize(model, vllm_config=vllm_config)
                self.assertEqual(model.model.start_layer_id, 60)
                self.assertEqual(model.config.target_layer_count, 60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
