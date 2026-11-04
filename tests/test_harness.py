"""The agent harness's policy gate rejects invalid controller calls (no API calls needed)."""

from types import SimpleNamespace

import numpy as np

from vericell.agent import Agent, Budget, Session


def session(n=3) -> Session:
    s = Session("t", None, "q", None, None)
    s.cands = [SimpleNamespace(inst=SimpleNamespace(box=(i, i, i + 5, i + 5))) for i in range(n)]
    s.p = np.full(n, 0.5)
    s.tried_phrases = ["organoid"]
    return s


def gate(s, name, **args):
    return Agent.__new__(Agent)._gate(s, {"name": name, "args": args})


def test_unknown_tool_rejected():
    assert gate(session(), "delete_everything")[0] is False


def test_objects_must_exist():
    assert gate(session(), "judge_objects", object_ids=[99], why="x")[0] is False
    assert gate(session(), "judge_objects", object_ids=[1], why="x")[0] is True


def test_already_judged_objects_rejected():
    s = session()
    s.verdicts[str([1, 1, 6, 6])] = {"verdict": "one_object", "confidence": 0.9}
    assert gate(s, "judge_objects", object_ids=[1], why="x")[0] is False


def test_repeated_phrase_and_budget():
    s = session()
    assert gate(s, "try_sam_prompt", phrase="Organoid", why="x")[0] is False
    s.budget = Budget(max_prompts=0)
    assert gate(s, "try_sam_prompt", phrase="cell cluster", why="x")[0] is False


def test_judge_budget():
    s = session()
    s.budget = Budget(max_judged=1)
    assert gate(s, "judge_objects", object_ids=[0, 1], why="x")[0] is False


def test_sam_prompts_come_from_the_measured_vocabulary_when_the_image_type_is_known():
    from vericell.agent import sam_phrases
    vocab = {"organoid": {"phrases": ["small round blobs", "circular cyst"]}}
    scene = {"target_object": "organoid", "sam_concepts": ["organoid", "round cell cluster"]}
    assert sam_phrases(scene, vocab) == (["small round blobs", "circular cyst"], "organoid")
    other = {"target_object": "bacterium", "sam_concepts": ["rod-shaped bacterium", "rod"]}
    assert sam_phrases(other, vocab) == (["rod-shaped bacterium", "rod"], "muse_spark")
