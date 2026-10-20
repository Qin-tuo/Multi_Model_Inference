# SPDX-License-Identifier: Apache-2.0
"""Reference inputs survive removal of disposable validation directories."""

from deploy.validation_inputs import input_path, is_local_input


def test_retained_input_resolves_without_rewriting_old_metadata(tmp_path):
    old = tmp_path / "05_runtime/validation/old/case.npz"
    saved = tmp_path / "05_runtime/datasets/validation-inputs/old/case.npz"
    saved.parent.mkdir(parents=True)
    saved.write_bytes(b"unchanged reference")
    assert input_path(old, tmp_path) == saved
    assert input_path(old, tmp_path).read_bytes() == b"unchanged reference"
    assert is_local_input(saved, tmp_path)


def test_current_input_takes_precedence_over_retained_copy(tmp_path):
    current = tmp_path / "05_runtime/validation/new/case.npz"
    current.parent.mkdir(parents=True)
    current.write_bytes(b"new experiment")
    assert input_path(current, tmp_path) == current


def test_input_boundary_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"outside")
    link = tmp_path / "05_runtime/datasets/validation-inputs/escape.npz"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    assert not is_local_input(link, tmp_path)
    assert input_path(outside, tmp_path) == outside
