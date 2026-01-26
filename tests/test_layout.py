"""Tests for layout helpers."""

import fsspec

from radarbase_io.layout import list_participants

UUID1 = "11111111-1111-1111-1111-111111111111"
UUID2 = "22222222-2222-2222-2222-222222222222"


def test_list_participants_local(tmp_path):
    root = tmp_path / "project"
    (root / UUID1).mkdir(parents=True)
    (root / UUID2).mkdir()
    (root / "not-a-uuid").mkdir()

    out = list_participants(root)
    assert set(out.tolist()) == {UUID1, UUID2}


def test_list_participants_explicit_fs(tmp_path):
    fs = fsspec.filesystem("file")
    root = tmp_path / "project"
    root.mkdir()
    (root / UUID1).mkdir()

    out = list_participants(root.as_uri(), fs=fs)
    assert out.tolist() == [UUID1]
