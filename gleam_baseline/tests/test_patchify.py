import numpy as np

from envs.patchify import coverage_ratio, mark_patch_visited, position_to_patch


def test_position_to_patch_center():
    idx = position_to_patch((0.0, 0.0), (10.0, 10.0), (10, 10))
    assert idx == (5, 5)


def test_mark_patch_visited_and_coverage():
    visited = np.zeros((2, 2), dtype=bool)
    assert mark_patch_visited(visited, (0, 1)) is True
    assert mark_patch_visited(visited, (0, 1)) is False
    assert coverage_ratio(visited) == 0.25
