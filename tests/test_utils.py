"""Tests for utility helpers."""

import numpy as np
import pytest

from radarbase_io.utils import parse_participant_uuids, parse_radar_path

UUID1 = "11111111-1111-1111-1111-111111111111"
UUID2 = "22222222-2222-2222-2222-222222222222"


def test_parse_participant_uuids_single_valid():
    assert parse_participant_uuids(UUID1) == UUID1


def test_parse_participant_uuids_single_invalid():
    assert parse_participant_uuids("not-a-uuid") is None


def test_parse_participant_uuids_trailing_slash():
    assert parse_participant_uuids(f"/x/{UUID1}/") == UUID1


def test_parse_participant_uuids_sequence_filters_invalid():
    paths = [f"/root/{UUID1}", "/root/not-a-uuid", f"/root/{UUID2}/"]
    out = parse_participant_uuids(paths)
    assert isinstance(out, np.ndarray)
    assert set(out.tolist()) == {UUID1, UUID2}


def test_parse_radar_path_basic():
    path = f"/data/proj/{UUID1}/accel"
    result = parse_radar_path(path)

    assert result["project_id"] == "proj"
    assert result["user_id"] == UUID1
    assert result["data_type"] == "accel"
    assert result["schema_file"] == "schema-accel.json"
    assert result["path"] == path
    assert result["schema_path"] == f"/data/proj/{UUID1}/schema-accel.json"


def test_parse_radar_path_trailing_slash():
    result = parse_radar_path(f"/data/proj/{UUID1}/accel/")
    assert result["data_type"] == "accel"


def test_parse_radar_path_too_short_raises():
    with pytest.raises(ValueError, match="Expected .*<project_id>"):
        parse_radar_path("a/b")
