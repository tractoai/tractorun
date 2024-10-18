from typing import Optional

import pytest

from tractorun.private.process_manager import (
    PoolStatus,
    check_status,
    has_failed,
    is_success,
)


@pytest.mark.parametrize(
    "exit_codes,expected",
    [
        ([1], True),
        ([0], False),
        ([None], False),
        ([None, 0], False),
        ([None, 1], True),
        ([None, 0, 1], True),
        ([0, 0], False),
        ([1, 0], True),
        ([1, 1], True),
    ],
)
def test_has_failed(exit_codes: list[Optional[int]], expected: bool) -> None:
    assert has_failed(exit_codes) == expected


@pytest.mark.parametrize(
    "exit_codes,expected",
    [
        ([1], False),
        ([0], True),
        ([None], False),
        ([None, 0], False),
        ([None, 1], False),
        ([None, 0, 1], False),
        ([0, 0], True),
        ([1, 0], False),
        ([1, 1], False),
    ],
)
def test_is_success(exit_codes: list[Optional[int]], expected: bool) -> None:
    assert is_success(exit_codes) == expected


@pytest.mark.parametrize(
    "exit_codes,expected",
    [
        ([1], PoolStatus.failed),
        ([0], PoolStatus.success),
        ([None], PoolStatus.running),
        ([None, None], PoolStatus.running),
        ([None, 0], PoolStatus.running),
        ([None, 1], PoolStatus.failed),
        ([None, 0, 1], PoolStatus.failed),
        ([0, 0], PoolStatus.success),
        ([1, 0], PoolStatus.failed),
        ([1, 1], PoolStatus.failed),
    ],
)
def test_check_status(exit_codes: list[Optional[int]], expected: bool) -> None:
    assert check_status(exit_codes) == expected
