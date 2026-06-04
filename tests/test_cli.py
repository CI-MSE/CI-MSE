from vlm_annotator.gemini_annotator import parse_episode_range
from val_metrics.validate import build_parser



def test_parse_episode_range_accepts_two_ints():
    assert parse_episode_range("[3, 5]") == (3, 5)


def test_parse_episode_range_rejects_code():
    try:
        parse_episode_range("__import__('os').system('echo bad')")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe episode range was accepted")
