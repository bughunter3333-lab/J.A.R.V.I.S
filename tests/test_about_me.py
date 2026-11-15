"""Offline tests for about_me: build time and exact durations, done in code."""
from datetime import datetime

import pytest

import config
import tools


@pytest.mark.parametrize("seconds,expected", [
    (0, "0 seconds"), (1, "1 second"), (61, "1 minute and 1 second"), (3600, "1 hour"),
    (3640, "1 hour and 40 seconds"),
    (2 * 86400 + 3 * 3600 + 12 * 60 + 5, "2 days, 3 hours, 12 minutes and 5 seconds"),
])
def test_exact_duration(seconds, expected):
    start = datetime(2026, 11, 13, 21, 0, 0)
    from datetime import timedelta
    assert tools.exact_duration(start, start + timedelta(seconds=seconds)) == expected


def test_built_at_setting_wins(monkeypatch):
    monkeypatch.setattr(config, "BUILT_AT", "2026-11-13 21:30:00")
    built, source = tools.build_time()
    assert built == datetime(2026, 11, 13, 21, 30) and "BUILT_AT" in source


def test_earliest_source_is_used_when_not_set(monkeypatch):
    monkeypatch.setattr(config, "BUILT_AT", None)
    monkeypatch.setattr(tools.sys, "platform", "win32")

    class FakeStat:
        st_ctime = datetime(2026, 11, 13, 20, 15, 0).timestamp()

    monkeypatch.setattr(type(config.BASE_DIR), "stat", lambda self, **kw: FakeStat())
    monkeypatch.setattr(tools, "_git", lambda *a: "abc" if a[0] == "rev-list" else "2026-11-15T10:56:55+11:00")
    built, source = tools.build_time()
    assert built == datetime(2026, 11, 13, 20, 15) and "folder was created" in source


def test_about_me_reports_exact_age(monkeypatch):
    monkeypatch.setattr(config, "BUILT_AT", "2026-11-13 21:30:00")

    class FrozenNow(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 15, 11, 58, 7)

    monkeypatch.setattr(tools, "datetime", FrozenNow)
    out = tools.about_me()
    assert "Built on Friday 13 November 2026 at 9:30:00 PM" in out
    assert "exactly 1 day, 14 hours, 28 minutes and 7 seconds ago" in out
