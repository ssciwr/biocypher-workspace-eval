"""Configuration, prompts and mutant patches stay consistent."""

import os

import pytest

from evaluation import config as cfg
from evaluation import grader
from evaluation.mutants import MUTANTS, apply


def test_levels_and_prompts():
    config = cfg.load()
    levels = cfg.levels(config)
    assert tuple(levels) == cfg.LEVELS
    for level in levels.values():
        prompt = level.prompt_file.read_text()
        assert level.final_marker in prompt
        assert "<" not in prompt.split("\n")[0]  # no unexpanded placeholders
        if level.dataset_file is not None:
            assert level.dataset_file.exists()
    assert set(grader.REQUIRED) == set(cfg.LEVELS)


def test_mutant_expectations_are_known_checks():
    for mutant in MUTANTS:
        assert mutant.expected_failures <= set(grader.ADAPTER_CHECKS)


REFERENCE = cfg.reference_repo(cfg.load())


@pytest.mark.skipif(not REFERENCE.exists(), reason="reference repo not available")
def test_mutants_apply_to_reference():
    source = grader.reference_adapter_source(REFERENCE)
    for mutant in MUTANTS:
        mutated = apply(mutant, source)
        assert mutated != source
        compile(mutated, mutant.name, "exec")


def test_patch_must_match_once():
    with pytest.raises(ValueError, match="found 0 times"):
        apply(MUTANTS[0], "class SyntheticPpiAdapter: pass")


@pytest.mark.skipif(
    not (REFERENCE.exists() and os.getenv("EVAL_SLOW")),
    reason="set EVAL_SLOW=1 (needs uv, network on first run)",
)
def test_grader_validation_end_to_end():
    assert grader.validate()
