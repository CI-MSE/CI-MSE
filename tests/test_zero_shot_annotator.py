import json

import pytest

from vlm_annotator.gemini_zero_shot_annotator import (
    ZERO_SHOT_PROMPT,
    build_zero_shot_prompt,
    parse_annotation_response,
)


def test_build_zero_shot_prompt_is_unchanged_without_task_instruction():
    assert build_zero_shot_prompt() == ZERO_SHOT_PROMPT


def test_build_zero_shot_prompt_includes_task_instruction():
    prompt = build_zero_shot_prompt("  Place the cup on the coaster.  ")

    assert "Task instruction:\nPlace the cup on the coaster." in prompt
    assert prompt.index("Task instruction:") < prompt.index(
        "Your task is to identify"
    )
    assert "do not assume that the entire instructed task is critical" in prompt


def test_parse_annotation_response_normalizes_timestamps():
    response = json.dumps(
        {
            "intervals": [
                {"label": " final grasp ", "start": 1.24, "end": 2.36},
                {"label": "precise placement", "start": 3, "end": 4.04},
            ]
        }
    )

    assert parse_annotation_response(response, 5.0) == {
        "intervals": [
            {"label": "final grasp", "start": 1.2, "end": 2.4},
            {"label": "precise placement", "start": 3.0, "end": 4.0},
        ]
    }


def test_parse_annotation_response_accepts_empty_intervals():
    assert parse_annotation_response('{"intervals": []}', 5.0) == {"intervals": []}


@pytest.mark.parametrize(
    "response",
    [
        '{"intervals": [{"label": "grasp", "start": -0.1, "end": 1.0}]}',
        '{"intervals": [{"label": "grasp", "start": 2.0, "end": 1.0}]}',
        '{"intervals": [{"label": "grasp", "start": 1.0, "end": 6.0}]}',
        (
            '{"intervals": ['
            '{"label": "grasp", "start": 1.0, "end": 2.0},'
            '{"label": "place", "start": 1.9, "end": 3.0}'
            ']}'
        ),
    ],
)
def test_parse_annotation_response_rejects_invalid_intervals(response):
    with pytest.raises(ValueError):
        parse_annotation_response(response, 5.0)
