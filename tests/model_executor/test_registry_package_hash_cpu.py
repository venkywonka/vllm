# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stdlib-only registry tests; run with .venv/bin/python -B <file> -v.

AST extraction exercises the real helpers without importing vLLM or torch.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import tempfile
import unittest
from _hashlib import UnsupportedDigestmodError
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "vllm/model_executor/models/registry.py"


def extract_definition(path, name, namespace, parent=None):
    tree = ast.parse(path.read_text())
    if parent is not None:
        tree = next(node for node in tree.body if getattr(node, "name", None) == parent)
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    node.decorator_list = []
    module = ast.Module(
        body=[*ast.parse("from __future__ import annotations").body, node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class RegistryPackageHashTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.package = self.root / "package"
        (self.package / "nvidia").mkdir(parents=True)
        self.init = self.package / "__init__.py"
        self.model = self.package / "nvidia/model.py"
        self.init.write_bytes(b"from .nvidia.model import Model\n")
        self.model.write_bytes(b"supports_pp = False\n")
        self.logger = logging.getLogger("vllm.registry.package_hash_cpu")
        self.namespace: dict[str, Any] = {
            "Path": Path,
            "os": os,
            "hashlib": hashlib,
            "UnsupportedDigestmodError": UnsupportedDigestmodError,
            "logger": self.logger,
        }
        self.safe_hash = extract_definition(
            ROOT / "vllm/utils/hashing.py", "safe_hash", self.namespace
        )
        self.hash = extract_definition(
            REGISTRY, "_get_model_module_hash", self.namespace
        )

    def test_nested_implementation_edit_invalidates_unchanged_entrypoint(self):
        before = self.hash(self.init)
        entrypoint = self.init.read_bytes()
        self.model.write_bytes(b"supports_pp = True\n")
        self.assertIsNotNone(before)
        self.assertNotEqual(before, self.hash(self.init))
        self.assertEqual(entrypoint, self.init.read_bytes())

    def test_relative_path_rename_invalidates(self):
        before = self.hash(self.init)
        self.model.rename(self.model.with_name("mtp.py"))
        self.assertNotEqual(before, self.hash(self.init))

    def test_python_file_addition_and_removal_invalidate(self):
        before = self.hash(self.init)
        extra = self.package / "common.py"
        extra.write_bytes(b"")
        self.assertNotEqual(before, self.hash(self.init))
        extra.unlink()
        self.assertEqual(before, self.hash(self.init))

    def test_non_python_files_are_ignored(self):
        before = self.hash(self.init)
        (self.package / "README.md").write_text("documentation")
        (self.package / "model.pyc").write_bytes(b"bytecode")
        self.assertEqual(before, self.hash(self.init))

    def test_hash_is_independent_of_root_and_traversal_order(self):
        other = self.root / "other"
        (other / "nvidia").mkdir(parents=True)
        for source in (self.model, self.init):
            (other / source.relative_to(self.package)).write_bytes(source.read_bytes())
        before = self.hash(self.init)
        self.assertEqual(before, self.hash(other / "__init__.py"))
        entries = [
            (directory, directories[::-1], filenames[::-1])
            for directory, directories, filenames in os.walk(self.package)
        ]
        with patch.object(os, "walk", return_value=iter(entries[::-1])):
            self.assertEqual(before, self.hash(self.init))

    def test_plain_module_retains_legacy_digest(self):
        expected = self.safe_hash(
            self.model.read_bytes(), usedforsecurity=False
        ).hexdigest()
        self.assertEqual(expected, self.hash(self.model))

    def test_package_spec_flag_hashes_siblings(self):
        entry = self.init.rename(self.init.with_name("entry.py"))
        before = self.hash(entry, is_package=True)
        self.model.write_bytes(b"supports_pp = True\n")
        self.assertNotEqual(before, self.hash(entry, is_package=True))

    def test_missing_source_disables_cache(self):
        self.assertIsNone(self.hash(self.package / "missing.py"))

    def test_read_failure_disables_cache(self):
        for source in (self.init, self.model):
            with self.subTest(source=source):
                error = OSError("read failed")
                with (
                    patch.object(Path, "read_bytes", side_effect=error),
                    self.assertLogs(self.logger, level="DEBUG") as captured,
                ):
                    self.assertIsNone(self.hash(source))
                self.assertEqual(len(captured.records), 1)
                record = captured.records[0]
                self.assertIn(str(source), record.getMessage())
                self.assertIn("skipping cache", record.getMessage())
                assert record.exc_info is not None
                self.assertIs(record.exc_info[1], error)
                self.assertIsNotNone(record.exc_info[2])

    def test_traversal_failure_disables_cache(self):
        error = OSError("directory unreadable")

        def failed_walk(path, *, onerror):
            onerror(error)

        with (
            patch.object(os, "walk", side_effect=failed_walk),
            self.assertLogs(self.logger, level="DEBUG") as captured,
        ):
            self.assertIsNone(self.hash(self.init))
        exception_info = captured.records[0].exc_info
        assert exception_info is not None
        self.assertIs(exception_info[1], error)

    def test_digest_failure_disables_cache(self):
        error = ValueError("digest unavailable")
        with (
            patch.dict(self.namespace, safe_hash=Mock(side_effect=error)),
            self.assertLogs(self.logger, level="DEBUG") as captured,
        ):
            self.assertIsNone(self.hash(self.init))
            self.assertIsNone(self.hash(self.model))
        self.assertEqual(len(captured.records), 2)
        for record in captured.records:
            assert record.exc_info is not None
            self.assertIs(record.exc_info[1], error)

    def test_safe_hash_fips_fallback_is_preserved(self):
        expected = hashlib.sha256(self.model.read_bytes()).hexdigest()
        with patch.object(hashlib, "md5", side_effect=ValueError):
            self.assertEqual(expected, self.hash(self.model))
            self.assertIsNotNone(self.hash(self.init))

    def prepare_inspection(self):
        fresh = object()
        self.namespace.update(
            __file__=str(REGISTRY),
            importlib=SimpleNamespace(
                util=SimpleNamespace(
                    find_spec=Mock(
                        return_value=SimpleNamespace(
                            origin=str(self.init),
                            submodule_search_locations=[str(self.package)],
                        )
                    )
                )
            ),
            logger=Mock(),
            _ModelInfo=SimpleNamespace(from_model_cls=Mock(return_value=fresh)),
            _run_in_subprocess=Mock(side_effect=lambda function: function()),
        )
        inspect = extract_definition(
            REGISTRY, "inspect_model_cls", self.namespace, "_LazyRegisteredModel"
        )
        model = SimpleNamespace(
            module_name="vllm.models.example",
            class_name="ExampleModel",
            load_model_cls=Mock(return_value=object()),
            _load_modelinfo_from_cache=Mock(return_value=None),
            _save_modelinfo_to_cache=Mock(),
        )
        return inspect, model, fresh

    def test_inspection_skips_cache_reads_and_writes_after_hash_failure(self):
        inspect, model, fresh = self.prepare_inspection()
        with patch.object(Path, "read_bytes", side_effect=OSError("read failed")):
            self.assertIs(fresh, inspect(model))
        model._load_modelinfo_from_cache.assert_not_called()
        model._save_modelinfo_to_cache.assert_not_called()
        self.namespace["_run_in_subprocess"].assert_called_once()
        model.load_model_cls.assert_called_once()
        self.namespace["logger"].debug.assert_any_call(
            "Cannot hash model source at %s; skipping cache",
            self.init,
            exc_info=True,
        )

    def test_resolution_failure_still_inspects_without_cache(self):
        inspect, model, fresh = self.prepare_inspection()
        self.namespace["importlib"].util.find_spec.side_effect = ImportError
        self.assertIs(fresh, inspect(model))
        model._load_modelinfo_from_cache.assert_not_called()
        model._save_modelinfo_to_cache.assert_not_called()
        self.namespace["_run_in_subprocess"].assert_called_once()
        self.namespace["logger"].debug.assert_any_call(
            "Cannot hash model source for class %s.%s; skipping cache",
            model.module_name,
            model.class_name,
            exc_info=True,
        )

    def test_inspection_rejects_legacy_package_cache_and_reuses_fresh_cache(self):
        inspect, model, fresh = self.prepare_inspection()
        legacy = self.safe_hash(
            self.init.read_bytes(), usedforsecurity=False
        ).hexdigest()
        cache = {"hash": legacy, "modelinfo": object()}
        model._load_modelinfo_from_cache.side_effect = lambda digest: (
            cache["modelinfo"] if cache["hash"] == digest else None
        )
        self.assertIs(fresh, inspect(model))
        digest = self.hash(self.init)
        self.assertNotEqual(legacy, digest)
        model._save_modelinfo_to_cache.assert_called_once_with(fresh, digest)
        cache.update(hash=digest, modelinfo=fresh)
        self.assertIs(fresh, inspect(model))
        self.namespace["_run_in_subprocess"].assert_called_once()


if __name__ == "__main__":
    unittest.main()
