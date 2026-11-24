"""Tests for the duplicated-data checks. Each case is written to an HDF5 file and read back, since the checks
look at HDF5 links and chunking."""

from datetime import datetime, timezone

import numpy as np
import pytest
from hdmf.backends.hdf5 import H5DataIO
from hdmf.common import DynamicTable, VectorData
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.epoch import TimeIntervals

from nwbinspector import Importance
from nwbinspector.checks import (
    _duplicates,
    check_identical_time_series,
    check_redundant_time_series_copies,
    check_table_duplicate_rows,
    check_table_identical_columns,
    check_table_ragged_times_repeated,
    check_time_series_identical_columns,
    check_timestamps_repeated,
    check_units_identical_spike_trains,
)


def _new_nwbfile() -> NWBFile:
    return NWBFile(
        session_description="test", identifier="id", session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc)
    )


@pytest.fixture
def roundtrip(tmp_path):
    """Write an NWBFile to disk and return it read back, closing the file afterwards."""
    ios = []

    def _roundtrip(nwbfile: NWBFile) -> NWBFile:
        path = tmp_path / f"test_{len(ios)}.nwb"
        with NWBHDF5IO(path, "w") as io:
            io.write(nwbfile)
        io = NWBHDF5IO(path, "r")
        ios.append(io)
        return io.read()

    yield _roundtrip
    for io in ios:
        io.close()


def _messages(result):
    if result is None:
        return []
    return [result] if not isinstance(result, list) else [m for m in result if m is not None]


def test_units_identical_spike_trains(roundtrip):
    rng = np.random.default_rng(0)
    nwbfile = _new_nwbfile()
    trains = [np.sort(rng.uniform(0, 100, 50)) for _ in range(4)]
    for unit_id, train in enumerate(trains):
        nwbfile.add_unit(id=unit_id, spike_times=train)
    nwbfile.add_unit(id=2, spike_times=trains[2])  # the same unit stored twice under one id
    nwbfile.add_unit(id=7, spike_times=trains[3])  # the same train under a new id
    nwbfile.add_unit(id=8, spike_times=np.sort(rng.uniform(0, 100, 50)))  # same count, different train
    units = roundtrip(nwbfile).units

    messages = _messages(check_units_identical_spike_trains(units))
    assert len(messages) == 2
    assert all(m.importance == Importance.CRITICAL for m in messages)
    assert "id 2 (rows 2, 4)" in messages[0].message
    assert "ids 3, 7" in messages[1].message


def test_units_identical_spike_trains_pass_for_empty_and_distinct(roundtrip):
    nwbfile = _new_nwbfile()
    for unit_id in range(5):
        nwbfile.add_unit(id=unit_id, spike_times=[])
    for unit_id in range(5, 10):
        nwbfile.add_unit(id=unit_id, spike_times=[float(unit_id)])
    assert check_units_identical_spike_trains(roundtrip(nwbfile).units) is None


def test_table_duplicate_rows_events_stored_several_times(roundtrip):
    rng = np.random.default_rng(1)
    nwbfile = _new_nwbfile()
    events = TimeIntervals(name="events", description="events")
    events.add_column(name="label", description="label")
    times = np.sort(rng.uniform(0, 100, 30))
    for _ in range(5):
        for t in times:
            events.add_row(start_time=t, stop_time=t + 0.1, label="cue")
    nwbfile.add_time_intervals(events)
    messages = _messages(check_table_duplicate_rows(roundtrip(nwbfile).intervals["events"]))
    assert len(messages) == 1
    assert "120 of 150 rows repeat another row" in messages[0].message


def test_table_duplicate_rows_pass_for_clean_trials(roundtrip):
    nwbfile = _new_nwbfile()
    for i in range(20):
        nwbfile.add_trial(start_time=float(i), stop_time=i + 0.5)
    assert check_table_duplicate_rows(roundtrip(nwbfile).trials) is None


def test_table_duplicate_rows_array_column_but_not_axis_column(roundtrip):
    rng = np.random.default_rng(2)
    nwbfile = _new_nwbfile()
    module = nwbfile.create_processing_module(name="ophys", description="ophys")
    curve = rng.normal(size=36)
    table = DynamicTable(
        name="tuning_curves",
        description="tuning curves",
        columns=[
            VectorData(name="curve", description="curve", data=np.tile(curve, (10, 1))),
            VectorData(name="bin", description="bin centers", data=np.tile(np.arange(36.0), (10, 1))),
        ],
    )
    module.add(table)
    messages = _messages(check_table_duplicate_rows(roundtrip(nwbfile).processing["ophys"]["tuning_curves"]))
    assert len(messages) == 1
    assert "Column 'curve'" in messages[0].message


def test_identical_time_series(roundtrip):
    rng = np.random.default_rng(3)
    nwbfile = _new_nwbfile()
    raw = rng.normal(size=(1000, 4)).astype("float32")
    nwbfile.add_acquisition(TimeSeries(name="raw", data=raw, unit="V", rate=1000.0))
    module = nwbfile.create_processing_module(name="ecephys", description="ecephys")
    module.add(TimeSeries(name="lfp", data=raw.copy(), unit="V", rate=1000.0))  # processed series equal to raw
    nwbfile.add_acquisition(TimeSeries(name="zeros1", data=np.zeros(500), unit="V", rate=10.0))
    nwbfile.add_acquisition(TimeSeries(name="zeros2", data=np.zeros(500), unit="V", rate=10.0))
    timestamps = np.sort(rng.uniform(0, 100, 200))
    nwbfile.add_acquisition(TimeSeries(name="lick", data=np.ones(200), unit="n.a.", timestamps=timestamps))
    nwbfile.add_acquisition(TimeSeries(name="reward", data=np.ones(200), unit="n.a.", timestamps=timestamps.copy()))
    nwbfile.add_acquisition(
        TimeSeries(name="other_lick", data=np.ones(200), unit="n.a.", timestamps=np.sort(rng.uniform(0, 100, 200)))
    )
    read_nwbfile = roundtrip(nwbfile)

    messages = _messages(check_identical_time_series(read_nwbfile))
    assert len(messages) == 2
    assert all(m.importance == Importance.CRITICAL for m in messages)
    text = " ".join(m.message for m in messages)
    assert "raw" in text and "ecephys/lfp" in text
    assert "lick" in text and "reward" in text and "zeros" not in text
    assert check_redundant_time_series_copies(read_nwbfile) is None


def test_redundant_time_series_copies(roundtrip):
    rng = np.random.default_rng(4)
    nwbfile = _new_nwbfile()
    module = nwbfile.create_processing_module(name="ecephys", description="ecephys")
    phase = rng.uniform(-np.pi, np.pi, size=(2000, 2))
    module.add(TimeSeries(name="up_phase", data=phase, unit="radians", rate=1250.0))
    module.add(TimeSeries(name="spindle_phase", data=phase.copy(), unit="radians", rate=1250.0))
    read_nwbfile = roundtrip(nwbfile)

    assert check_identical_time_series(read_nwbfile) is None
    messages = _messages(check_redundant_time_series_copies(read_nwbfile))
    assert len(messages) == 1
    assert messages[0].importance == Importance.BEST_PRACTICE_VIOLATION


def test_identical_time_series_pass_for_link(roundtrip):
    rng = np.random.default_rng(5)
    nwbfile = _new_nwbfile()
    series = TimeSeries(name="a", data=rng.normal(size=500), unit="V", rate=10.0)
    nwbfile.add_acquisition(series)
    nwbfile.add_acquisition(TimeSeries(name="b", data=series, unit="V", rate=10.0))
    read_nwbfile = roundtrip(nwbfile)
    assert check_identical_time_series(read_nwbfile) is None
    assert check_redundant_time_series_copies(read_nwbfile) is None


def test_table_identical_columns(roundtrip):
    rng = np.random.default_rng(6)
    nwbfile = _new_nwbfile()
    y = rng.normal(size=64)
    table = DynamicTable(
        name="coordinates",
        description="coordinates",
        columns=[
            VectorData(name="x", description="x", data=rng.normal(size=64)),
            VectorData(name="y", description="y", data=y),
            VectorData(name="z", description="z", data=y.copy()),
            VectorData(name="good", description="flag", data=np.ones(64, dtype=bool)),
            VectorData(name="good2", description="flag", data=np.ones(64, dtype=bool)),
            VectorData(name="stop_time", description="time", data=y + 10),
            VectorData(name="return_stop_time", description="time", data=y + 10),
        ],
    )
    nwbfile.add_acquisition(table)
    messages = _messages(check_table_identical_columns(roundtrip(nwbfile).acquisition["coordinates"]))
    assert len(messages) == 1
    assert "'y', 'z'" in messages[0].message


def test_time_series_identical_columns(roundtrip):
    rng = np.random.default_rng(7)
    nwbfile = _new_nwbfile()
    xy = rng.normal(size=(1000, 2))
    position = Position(
        spatial_series=SpatialSeries(name="position", data=np.c_[xy, xy[:, 1]], reference_frame="origin", rate=30.0)
    )
    nwbfile.create_processing_module(name="behavior", description="behavior").add(position)
    series = roundtrip(nwbfile).processing["behavior"]["Position"]["position"]
    message = check_time_series_identical_columns(series)
    assert message is not None
    assert "columns 1 = 2" in message.message


def test_time_series_identical_columns_sampled_and_chunked(roundtrip, monkeypatch):
    monkeypatch.setattr(_duplicates, "FULL_READ_BYTES", 64 * 1024)
    rng = np.random.default_rng(8)
    nwbfile = _new_nwbfile()
    data = rng.normal(size=(100_000, 6)).astype("float32")
    data[:, 4] = data[:, 2]
    data[90_000:, 5] = data[90_000:, 1]  # matches column 1 only at the end, so the first block rules it out
    nwbfile.add_acquisition(TimeSeries(name="big", data=H5DataIO(data, chunks=(16384, 2)), unit="V", rate=1000.0))
    series = roundtrip(nwbfile).acquisition["big"]

    message = check_time_series_identical_columns(series)
    assert message is not None
    assert "columns 2 = 4" in message.message and "sampled" in message.message
    # A budget smaller than one block of chunks skips the series.
    assert check_time_series_identical_columns(series, max_fetch_bytes=1024) is None


def test_table_ragged_times_repeated(roundtrip):
    rng = np.random.default_rng(9)
    nwbfile = _new_nwbfile()
    events = DynamicTable(name="annotated_events", description="events")
    events.add_column(name="event_times", description="times", index=True)
    events.add_column(name="label", description="label")
    base = np.sort(rng.uniform(0, 100, 40))
    events.add_row(event_times=np.sort(np.concatenate([base] * 5)), label="cue")
    events.add_row(event_times=np.sort(rng.uniform(0, 100, 40)), label="reward")
    nwbfile.add_acquisition(events)
    read_nwbfile = roundtrip(nwbfile)
    messages = _messages(check_table_ragged_times_repeated(read_nwbfile.acquisition["annotated_events"]))
    assert len(messages) == 1
    assert "160 of 200 times are repeats" in messages[0].message


def test_table_ragged_times_repeated_pass_for_millisecond_times(roundtrip):
    rng = np.random.default_rng(10)
    nwbfile = _new_nwbfile()
    events = DynamicTable(name="annotated_events", description="events")
    events.add_column(name="event_times", description="times", index=True)
    times = np.sort(np.round(rng.uniform(0, 1, 500), 2))  # 10 ms resolution, so many times coincide
    events.add_row(event_times=times)
    nwbfile.add_acquisition(events)
    read_nwbfile = roundtrip(nwbfile)
    assert check_table_ragged_times_repeated(read_nwbfile.acquisition["annotated_events"]) is None


def test_timestamps_repeated(roundtrip):
    rng = np.random.default_rng(11)
    nwbfile = _new_nwbfile()
    timestamps = np.sort(rng.uniform(0, 100, 300))
    nwbfile.add_acquisition(
        TimeSeries(name="doubled", data=np.arange(600), unit="n.a.", timestamps=np.sort(np.r_[timestamps, timestamps]))
    )
    nwbfile.add_acquisition(TimeSeries(name="clean", data=np.arange(300), unit="n.a.", timestamps=timestamps))
    read_nwbfile = roundtrip(nwbfile)
    message = check_timestamps_repeated(read_nwbfile.acquisition["doubled"])
    assert message is not None
    assert "300 of 600 timestamps" in message.message
    assert check_timestamps_repeated(read_nwbfile.acquisition["clean"]) is None
