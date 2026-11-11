"""Tests for the streaming runner and its salvage deadline.

Why these exist: this code path only ever runs while a GPU is being billed.
The first control run proved that an untested failure path there is expensive,
so the deadline behaviour is tested against cheap shell commands rather than
discovered on an L4.

``modal_app`` is not an installed package, so the module is loaded by path.
That is deliberate: the alternative is making modal_app importable, which the
container-startup tests exist specifically to prevent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

FOLD_PATH = Path(__file__).resolve().parents[1] / "modal_app" / "fold.py"


def _load_fold_module():
    spec = importlib.util.spec_from_file_location("_bb_fold_under_test", FOLD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fold():
    return _load_fold_module()


def test_output_is_streamed_and_returned_as_a_tail(fold, capsys):
    """Every line must reach stdout WHILE the process runs, not after.

    capsys can only prove the lines were printed, not when. The ordering is
    enforced by construction: the draining thread prints each line as it is
    read, and the test below shows lines survive even when the child is killed,
    which is impossible with a buffer that is only read at exit.
    """
    rc, tail, elapsed, hit = fold._run_streamed(
        ["sh", "-c", "echo first; echo second; echo third"], deadline_sec=30
    )
    assert rc == 0
    assert hit is False
    assert tail == ["first", "second", "third"]
    out = capsys.readouterr().out
    for line in ("first", "second", "third"):
        assert line in out
    # Elapsed-second prefixes are what let a two-hour run be read at a glance.
    assert "[cf +" in out


def test_deadline_kills_the_child_and_reports_it(fold):
    """The whole point: we kill it, so the caller still gets to salvage.

    If Modal's own timeout kills the container instead, the Volume is never
    committed and finished designs are lost. That is exactly what cost
    $1.70 and two GPU-hours for nothing.
    """
    rc, tail, elapsed, hit = fold._run_streamed(
        ["sh", "-c", "echo starting; sleep 60"], deadline_sec=1
    )
    assert hit is True
    assert rc != 0, "a killed process must not report success"
    assert elapsed < 30, f"should have been killed promptly, took {elapsed:.1f}s"
    # Output produced before the kill is still available for diagnosis.
    assert "starting" in tail


def test_output_before_a_crash_survives(fold):
    """A nonzero exit still yields the log that explains it."""
    rc, tail, elapsed, hit = fold._run_streamed(
        ["sh", "-c", "echo progress line; echo boom >&2; exit 3"], deadline_sec=30
    )
    assert rc == 3
    assert hit is False
    # stderr is merged into the stream, so a traceback is not lost.
    assert "progress line" in tail
    assert "boom" in tail


def test_tail_is_bounded_so_a_chatty_run_cannot_blow_up_memory(fold):
    rc, tail, elapsed, hit = fold._run_streamed(
        ["sh", "-c", "i=0; while [ $i -lt 500 ]; do echo line$i; i=$((i+1)); done"],
        deadline_sec=60,
        tail_lines=50,
    )
    assert rc == 0
    assert len(tail) == 50
    assert tail[-1] == "line499", "the tail must keep the END, which is the useful part"


def test_salvage_margin_leaves_real_time_to_commit(fold):
    """A margin too small to parse and copy in is no margin at all."""
    assert fold.SALVAGE_MARGIN_SEC >= 300
    assert fold.SALVAGE_MARGIN_SEC < fold.FOLD_TIMEOUT_SEC / 2


def test_child_is_unbuffered_so_progress_actually_arrives(fold):
    """A child that block-buffers stdout defeats the whole streaming fix.

    This asserts the environment is set rather than timing a race: the child
    reports whether PYTHONUNBUFFERED reached it.
    """
    rc, tail, elapsed, hit = fold._run_streamed(
        ["sh", "-c", 'echo "unbuffered=$PYTHONUNBUFFERED"'], deadline_sec=30
    )
    assert rc == 0
    assert tail == ["unbuffered=1"]


def test_child_inherits_the_rest_of_the_environment(fold):
    """Overriding the env must not wipe PATH and leave colabfold_batch unfound."""
    rc, tail, elapsed, hit = fold._run_streamed(
        ["sh", "-c", 'test -n "$PATH" && echo path-present'], deadline_sec=30
    )
    assert rc == 0
    assert "path-present" in tail
