# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute actual async PP feedback/routing code using only the Python stdlib.

Run: PYTHONDONTWRITEBYTECODE=1 python3 -B <this file> -v
The tensor, transport, stream, and event objects below are CPU fakes; importing
this test does not import torch, NumPy, vLLM, or GPU extensions.
"""

from __future__ import annotations

import ast
import copy
import itertools
import operator
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "vllm/v1/worker/gpu_model_runner.py"


def compile_nodes(nodes, namespace):
    module = ast.Module(
        body=[
            *ast.parse("from __future__ import annotations").body,
            *copy.deepcopy(nodes),
        ],
        type_ignores=[],
    )
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            node.decorator_list = []
    exec(
        compile(ast.fix_missing_locations(module), "<actual-vllm-source>", "exec"),
        namespace,
    )


class Scalar(int):
    def item(self):
        return int(self)


class Tensor:
    def __init__(self, values, *, shape=None, data=None, indices=None):
        if data is not None:
            self.data, self.indices, self.shape = data, indices, shape
        else:
            if shape is None:
                shape = (
                    (len(values), len(values[0]))
                    if values and isinstance(values[0], list)
                    else (len(values),)
                )
            self.shape = shape
            self.data = (
                list(itertools.chain.from_iterable(values))
                if values and isinstance(values[0], list)
                else list(values)
            )
            self.indices = list(range(len(self.data)))
        self.device = "cpu"

    @classmethod
    def empty(cls, shape, **kwargs):
        shape = (shape,) if isinstance(shape, int) else shape
        size = 1
        for dim in shape:
            size *= dim
        return cls([-999] * size, shape=shape)

    def flat(self):
        return [self.data[i] for i in self.indices]

    def tolist(self):
        values = self.flat()
        return (
            [
                values[i : i + self.shape[1]]
                for i in range(0, len(values), self.shape[1])
            ]
            if len(self.shape) == 2
            else values
        )

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        return iter(Scalar(value) for value in self.flat())

    def __getitem__(self, key):
        keys = key if isinstance(key, tuple) else (key,)
        keys += (slice(None),) * (len(self.shape) - len(keys))
        selections, shape = [], []
        for size, part in zip(self.shape, keys):
            if isinstance(part, Tensor):
                selected = part.flat()
            elif isinstance(part, slice):
                selected = list(range(size))[part]
            else:
                selections.append([int(part)])
                continue
            selections.append(selected)
            shape.append(len(selected))
        indices = []
        for coords in itertools.product(*selections):
            offset = 0
            for coordinate, size in zip(coords, self.shape):
                offset = offset * size + coordinate
            indices.append(self.indices[offset])
        if not shape:
            return Scalar(self.data[indices[0]])
        return Tensor([], data=self.data, indices=indices, shape=tuple(shape))

    def __setitem__(self, key, value):
        destination = self[key]
        if isinstance(destination, Tensor):
            values = (
                value.flat()
                if isinstance(value, Tensor)
                else [value] * len(destination.indices)
            )
            for index, item in zip(destination.indices, values):
                self.data[index] = item
        else:
            keys = key if isinstance(key, tuple) else (key,)
            offset = 0
            for coordinate, size in zip(keys, self.shape):
                offset = offset * size + int(coordinate)
            self.data[self.indices[offset]] = value

    def copy_(self, other, **kwargs):
        assert self.shape == other.shape, (self.shape, other.shape)
        for index, value in zip(self.indices, other.flat()):
            self.data[index] = value
        return self

    def to(self, *args, **kwargs):
        return self

    def contiguous(self):
        return self

    def dim(self):
        return len(self.shape)

    def unsqueeze(self, dim):
        shape = list(self.shape)
        shape.insert(dim, 1)
        return Tensor(self.flat(), shape=tuple(shape))

    def flatten(self):
        return Tensor(self.flat())

    def index_select(self, dim, indices):
        assert dim == 0
        selected = self[indices]
        return Tensor(selected.flat(), shape=selected.shape)

    def scatter_(self, dim, index, src):
        assert dim == 0
        for destination, value in zip(index.flat(), src.flat()):
            self[destination] = value
        return self

    def binary(self, other, operation):
        rhs = other.flat() if isinstance(other, Tensor) else [other] * len(self.indices)
        return Tensor(
            [operation(a, b) for a, b in zip(self.flat(), rhs)], shape=self.shape
        )

    def __add__(self, other):
        return self.binary(other, operator.add)

    def __ge__(self, other):
        return self.binary(other, operator.ge)

    def __gt__(self, other):
        return self.binary(other, operator.gt)

    def __and__(self, other):
        return self.binary(other, operator.and_)

    def clamp(self, min):
        return Tensor([max(value, min) for value in self.flat()], shape=self.shape)

    def int(self):
        return self


class Event:
    def __init__(self):
        self.records = self.waits = 0

    def record(self):
        self.records += 1

    def synchronize(self):
        self.waits += 1


class FeedbackHarness:
    METHODS = (
        "_pp_broadcast_spec_decode_state",
        "_pp_receive_spec_decode_state",
        "_pp_receive_prev_sampled_token_ids_to_input_batch",
        "_copy_valid_sampled_token_count",
        "_get_valid_sampled_token_count",
        "_prepare_input_ids",
    )

    def __init__(self):
        self.last_rank = True
        self.sent = []
        self.namespace: dict[str, Any] = {
            "get_pp_group": self.pp_group,
            "torch": SimpleNamespace(
                Tensor=Tensor,
                int32="int32",
                int64="int64",
                empty=Tensor.empty,
                tensor=lambda values, **kwargs: Tensor(values),
                cuda=SimpleNamespace(
                    current_stream=lambda: None, stream=lambda stream: nullcontext()
                ),
                where=lambda condition, a, b: Tensor(
                    [
                        x if flag else y
                        for flag, x, y in zip(condition.flat(), a.flat(), b.flat())
                    ]
                ),
            ),
            "np": SimpleNamespace(
                nonzero=lambda mask: (
                    Tensor([i for i, value in enumerate(mask) if value]),
                )
            ),
        }
        tree = ast.parse(RUNNER.read_text())
        runner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "GPUModelRunner"
        )
        self.methods = {
            node.name: node for node in runner.body if isinstance(node, ast.FunctionDef)
        }
        compile_nodes([self.methods[name] for name in self.METHODS], self.namespace)
        utils = ast.parse((ROOT / "vllm/v1/spec_decode/utils.py").read_text())
        correction = next(
            node
            for node in utils.body
            if getattr(node, "name", None)
            == "update_num_computed_tokens_for_batch_change"
        )
        compile_nodes([correction], self.namespace)

    def pp_group(self):
        return SimpleNamespace(
            is_last_rank=self.last_rank,
            world_size=2,
            broadcast_tensor_dict=self.broadcast,
        )

    def broadcast(self, state=None, src=None):
        assert src == 1
        if self.last_rank:
            assert state is not None
            self.sent.append(copy.deepcopy(state))
            return state
        assert state is None
        return copy.deepcopy(self.sent[-1])

    def runner(self, req_ids=("a", "b", "prefill"), discard=(False, False, True)):
        count = len(req_ids)
        runner = SimpleNamespace(
            device="cpu",
            pin_memory=False,
            use_pp_async_spec_decode=True,
            use_async_spec_decode=True,
            use_async_scheduling=True,
            enable_prompt_embeds=False,
            num_spec_tokens=3,
            prev_num_spec_tokens=0,
            _draft_token_ids=None,
            valid_sampled_token_count_gpu=None,
            valid_sampled_token_count_cpu=Tensor([0] * count),
            valid_sampled_token_count_event=Event(),
            valid_sampled_token_count_copy_stream=SimpleNamespace(
                wait_stream=lambda stream: None
            ),
            input_batch=SimpleNamespace(
                req_ids=list(req_ids),
                num_reqs=count,
                prev_sampled_token_ids=None,
                prev_req_id_to_index=None,
                num_tokens_no_spec=Tensor([100] * count),
                is_token_ids=Tensor([[False] * 512 for _ in req_ids]),
            ),
            discard_request_mask=SimpleNamespace(np=Tensor(list(discard))),
            requests={
                req_id: SimpleNamespace(output_token_ids=[]) for req_id in req_ids
            },
            _is_all_reqs_chunked_prefill=lambda: all(discard),
        )
        for name in self.METHODS:
            setattr(runner, name, MethodType(self.namespace[name], runner))
        return runner

    def exchange(
        self, *, local_order=("a", "b", "prefill"), discard=(False, False, True)
    ):
        sender = self.runner()
        sender.input_batch.prev_sampled_token_ids = Tensor([[10], [23], [99]])
        sender._draft_token_ids = Tensor(
            [[110, 111, 112], [210, 211, 212], [310, 311, 312]]
        )
        sender.valid_sampled_token_count_gpu = Tensor([1, 4, 0])
        self.last_rank = True
        sender._pp_broadcast_spec_decode_state()
        receiver = self.runner(local_order, discard)
        self.last_rank = False
        receiver._pp_receive_prev_sampled_token_ids_to_input_batch()
        return receiver


class PipelineFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.harness = FeedbackHarness()

    def test_mixed_acceptance_feedback_and_prefill_discard(self):
        receiver = self.harness.exchange()
        self.assertEqual(len(self.harness.sent), 1)
        self.assertEqual(
            receiver.input_batch.prev_sampled_token_ids.tolist(), [[10], [23], [99]]
        )
        self.assertEqual(
            receiver._draft_token_ids.tolist(),
            [[110, 111, 112], [210, 211, 212], [310, 311, 312]],
        )
        self.assertEqual(receiver.prev_num_spec_tokens, 3)
        self.assertEqual(receiver.valid_sampled_token_count_gpu.tolist(), [1, 4, 0])
        self.assertEqual(receiver._get_valid_sampled_token_count(), [1, 4, 0])
        self.assertEqual(receiver.valid_sampled_token_count_event.records, 1)
        self.assertEqual(receiver.input_batch.prev_req_id_to_index, {"a": 0, "b": 1})
        self.assertEqual(
            receiver.input_batch.num_tokens_no_spec.tolist(), [101, 101, 100]
        )
        self.assertEqual(
            [
                len(receiver.requests[key].output_token_ids)
                for key in ("a", "b", "prefill")
            ],
            [1, 1, 0],
        )

    def test_stage_local_reordering_reorders_all_feedback_tensors(self):
        receiver = self.harness.exchange(local_order=("b", "a", "prefill"))
        self.assertEqual(
            receiver.input_batch.prev_sampled_token_ids.tolist(), [[23], [10], [99]]
        )
        self.assertEqual(receiver._draft_token_ids.tolist()[0], [210, 211, 212])
        self.assertEqual(receiver._get_valid_sampled_token_count(), [4, 1, 0])
        self.assertEqual(receiver.input_batch.prev_req_id_to_index, {"b": 0, "a": 1})

    def test_chunked_prefill_still_participates_but_has_no_cached_decode_rows(self):
        receiver = self.harness.exchange(discard=(True, True, True))
        self.assertEqual(len(self.harness.sent), 1)
        self.assertEqual(receiver.input_batch.prev_req_id_to_index, {})
        self.assertEqual(
            receiver.input_batch.num_tokens_no_spec.tolist(), [100, 100, 100]
        )
        self.assertTrue(
            all(not request.output_token_ids for request in receiver.requests.values())
        )

    def test_unknown_request_ids_fail_before_caching_feedback(self):
        with self.assertRaisesRegex(RuntimeError, "request IDs do not match"):
            self.harness.exchange(local_order=("a", "other", "prefill"))

    def test_prepare_inputs_uses_real_drafts_after_reorder_retirement_and_new_prefill(
        self,
    ):
        receiver = self.harness.exchange()
        receiver.input_batch.req_ids = ["b", "new", "a"]
        receiver.prev_positions = SimpleNamespace(np=Tensor([1, -1, 0]))
        gpu = Tensor([-999] * 10)
        cpu = Tensor([-1, -1, -1, -1, 77, 78, -1, -1, -1, -1])
        receiver.input_ids = SimpleNamespace(
            gpu=gpu, copy_to_gpu=lambda count: gpu[:count].copy_(cpu[:count])
        )
        scheduler = SimpleNamespace(
            scheduled_spec_decode_tokens={"a": [-1] * 3, "b": [-1] * 3}
        )
        receiver._prepare_input_ids(scheduler, 3, 10, Tensor([4, 6, 10]))
        self.assertEqual(gpu.tolist(), [23, 210, 211, 212, 77, 78, 10, 110, 111, 112])

    def test_actual_accepted_count_correction_handles_reorder_and_new_requests(self):
        receiver = self.harness.exchange()
        computed = Tensor([100, 200, 300])
        accepted = Tensor([1, 1, 1])
        self.harness.namespace["update_num_computed_tokens_for_batch_change"](
            computed,
            accepted,
            Tensor([1, -1, 0]),
            receiver.valid_sampled_token_count_gpu,
            Tensor([3, 3, 0]),
            Tensor([204, 15, 104]),
        )
        self.assertEqual(computed.tolist(), [204, 15, 101])
        self.assertEqual(accepted.tolist(), [4, 1, 1])

    def test_optimistic_extension_and_trim_match_last_stage_repeatedly(self):
        update = self.harness.methods["_update_states"]
        loop = next(
            node
            for node in ast.walk(update)
            if isinstance(node, ast.For)
            and ast.unparse(node.iter) == "enumerate(req_data.req_ids)"
        )
        selected = []
        for node in loop.body:
            if isinstance(node, ast.If):
                condition = ast.unparse(node.test)
                if (
                    condition.startswith("req_state.prev_num_draft_len")
                    or condition == "not is_last_rank"
                    or "num_output_tokens < len(req_state.output_token_ids)"
                    in condition
                    or (
                        condition.startswith("not is_last_rank and")
                        and "use_pp_async_spec_decode" in condition
                    )
                ):
                    selected.append(node)
        self.assertEqual(len(selected), 4)
        for scheduler_output_count in (1, 2, 4, 5, 8):
            stage_results = []
            for is_last_rank in (False, True):
                request = SimpleNamespace(
                    output_token_ids=[-1] * scheduler_output_count, prev_num_draft_len=3
                )
                runner = SimpleNamespace(
                    use_async_scheduling=True,
                    use_pp_async_spec_decode=True,
                    prev_num_draft_tokens=SimpleNamespace(np=Tensor([0])),
                    input_batch=SimpleNamespace(
                        prev_req_id_to_index={"a": 0},
                        num_prompt_tokens=Tensor([100]),
                        num_tokens_no_spec=Tensor([100 + scheduler_output_count]),
                    ),
                )
                namespace: dict[str, Any] = {
                    "self": runner,
                    "req_state": request,
                    "req_index": 0,
                    "req_id": "a",
                    "is_last_rank": is_last_rank,
                    "is_ngram_gpu": False,
                    "deferred_spec_decode_corrections": [],
                    "req_data": SimpleNamespace(new_token_ids=[]),
                    "num_output_tokens": scheduler_output_count,
                    "num_computed_tokens": 100 + scheduler_output_count + 3,
                }
                compile_nodes(selected, namespace)
                stage_results.append(
                    (
                        len(request.output_token_ids),
                        runner.input_batch.num_tokens_no_spec.tolist(),
                    )
                )
                self.assertEqual(runner.prev_num_draft_tokens.np.tolist(), [3])
                self.assertEqual(len(namespace["deferred_spec_decode_corrections"]), 1)
            self.assertEqual(stage_results[0], stage_results[1])
            self.assertEqual(
                stage_results[0],
                (scheduler_output_count, [100 + scheduler_output_count]),
            )

    def test_actual_deferred_cpu_correction_uses_received_counts_after_reorder(self):
        receiver = self.harness.exchange()
        receiver.input_batch.req_id_to_index = {"b": 0, "a": 1}
        receiver.input_batch.num_computed_tokens_cpu = Tensor([204, 104])
        requests = [
            SimpleNamespace(num_computed_tokens=104),
            SimpleNamespace(num_computed_tokens=204),
        ]
        update = self.harness.methods["_update_states"]
        closure = next(
            node
            for node in ast.walk(update)
            if isinstance(node, ast.FunctionDef)
            and node.name == "correct_spec_decode_token_counts"
        )
        namespace = {
            "self": receiver,
            "is_ngram_gpu": False,
            "deferred_spec_decode_corrections": [
                ("a", 3, requests[0]),
                ("b", 3, requests[1]),
            ],
        }
        compile_nodes([closure], namespace)
        namespace[closure.name]()
        self.assertEqual(
            [request.num_computed_tokens for request in requests], [101, 204]
        )
        self.assertEqual(
            receiver.input_batch.num_computed_tokens_cpu.tolist(), [204, 101]
        )

    def test_feedback_and_early_return_source_order_preserves_async_state(self):
        execute = self.harness.methods["execute_model"]
        early_stage = next(
            node
            for node in ast.walk(execute)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "not get_pp_group().is_last_rank"
            and any(
                isinstance(child, ast.Return)
                and child.value is not None
                and ast.unparse(child.value) == "hidden_states"
                for child in node.body
            )
        )
        correction = next(
            node
            for node in ast.walk(early_stage)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "deferred_state_corrections_fn"
        )
        stage_return = next(
            node for node in early_stage.body if isinstance(node, ast.Return)
        )
        self.assertLess(correction.lineno, stage_return.lineno)
        sample = self.harness.methods["sample_tokens"]
        calls = [node for node in ast.walk(sample) if isinstance(node, ast.Call)]
        feedback = next(
            node
            for node in calls
            if ast.unparse(node.func) == "self._pp_broadcast_spec_decode_state"
        )
        for node in calls:
            if ast.unparse(node.func) in (
                "self._bookkeeping_sync",
                "propose_draft_token_ids",
            ):
                self.assertLess(node.lineno, feedback.lineno)
        old_send_guard = next(
            node
            for node in ast.walk(sample)
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Expr)
                and isinstance(child.value, ast.Call)
                and ast.unparse(child.value.func)
                == "self._pp_broadcast_prev_sampled_token_ids"
                for child in node.body
            )
        )
        self.assertIn(
            "not self.use_pp_async_spec_decode", ast.unparse(old_send_guard.test)
        )


if __name__ == "__main__":
    unittest.main()
