from typing import NamedTuple
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

from livenodes import Graph

from livenodes_io_h5_csv.out_h5_csv import Out_h5_csv
from livenodes_io_h5_csv.in_playback_h5_csv import In_playback_h5_csv
from livenodes_io_python.out_python import Out_python
from livenodes_io_python.in_python import In_python


class Out_nodes(NamedTuple):
    ts: Out_python
    channels: Out_python
    annot: Out_python


# Example annotation includes small 1 and 2 size blocks to test these edge cases.
_anot = ["1"] * 5 + ["2"] * 2 + ["3"] * 1 + ["1"] * 2 + ["2"] * 3 + ["3"] * 7


def _prepare_data(tmp_path, generate_annot=True):
    data = np.arange(100).reshape((1, 20, 5))  # 20 samples with 5 channels each

    data_in = In_python(name="A", data=data)
    channels_in = In_python(name="Channels", data=[["A", "B", "C", "D", "E"]])

    collect_data = Out_python(name="B")
    collect_data.add_input(data_in, emit_port=data_in.ports_out.any, recv_port=collect_data.ports_in.any)

    write_data = Out_h5_csv(name="C", folder=f"{tmp_path}/")
    write_data.add_input(data_in, emit_port=data_in.ports_out.any, recv_port=write_data.ports_in.ts)
    write_data.add_input(channels_in, emit_port=channels_in.ports_out.any, recv_port=write_data.ports_in.channels)

    if generate_annot:
        annot_in = In_python(name="D", data=[_anot])
        write_data.add_input(annot_in, emit_port=annot_in.ports_out.any, recv_port=write_data.ports_in.annot)

    g = Graph(start_node=data_in)
    g.start_all()
    g.join_all()
    g.stop_all()

    return collect_data.get_state()


def _run_test_pipeline(tmp_path, emit_at_once, channel_names=None):
    # Set sample rate very high since we don't want actual real-time simulation here
    read_data = In_playback_h5_csv(
        name="A", files=f"{tmp_path}/*.h5", loop=False, emit_at_once=emit_at_once, meta={'channels': channel_names, 'sample_rate': 1000000}
    )

    collect_data = Out_python(name="B")
    collect_data.add_input(read_data, emit_port=read_data.ports_out.ts, recv_port=collect_data.ports_in.any)

    collect_channels = Out_python(name="C")
    collect_channels.add_input(read_data, emit_port=read_data.ports_out.channels, recv_port=collect_channels.ports_in.any)

    collect_anot = Out_python(name="D")
    collect_anot.add_input(read_data, emit_port=read_data.ports_out.annot, recv_port=collect_anot.ports_in.any)

    g = Graph(start_node=read_data)
    g.start_all()
    g.join_all()
    g.stop_all()

    return Out_nodes(collect_data, collect_channels, collect_anot)


def _run_single_test(tmp_path, emit_at_once, exp_data_shape, empty_annot=False, exp_annot_shape=None):
    if empty_annot:
        expected_data = np.array(_prepare_data(tmp_path, generate_annot=False)).reshape(exp_data_shape)
        expected_annot = []
    else:
        expected_data = np.array(_prepare_data(tmp_path)).reshape(exp_data_shape)
        expected_annot = np.array(_anot).reshape(exp_annot_shape)
    channels = ["CH1", "CH2", "CH3", "CH4", "CH5"]

    results = _run_test_pipeline(tmp_path, emit_at_once=emit_at_once, channel_names=channels)

    actual_data = np.array(results.ts.get_state())
    actual_annot = results.annot.get_state()
    actual_channels = results.channels.get_state()[0]

    np.testing.assert_equal(actual_data, expected_data)
    np.testing.assert_equal(actual_annot, expected_annot)
    np.testing.assert_equal(actual_channels, channels)


class TestProcessing:

    def test_emit_1(self, tmp_path):
        _run_single_test(tmp_path, emit_at_once=1, exp_data_shape=(20, 1, 5), exp_annot_shape=(20, 1, 1))

    def test_emit_2(self, tmp_path):
        _run_single_test(tmp_path, emit_at_once=2, exp_data_shape=(10, 2, 5), exp_annot_shape=(10, 2, 1))

    def test_emit_5(self, tmp_path):
        _run_single_test(tmp_path, emit_at_once=5, exp_data_shape=(4, 5, 5), exp_annot_shape=(4, 5, 1))

    def test_emit_20(self, tmp_path):
        _run_single_test(tmp_path, emit_at_once=20, exp_data_shape=(1, 20, 5), exp_annot_shape=(1, 20, 1))

    def test_emit_more_than_data(self, tmp_path):  # Should behave the same as data length for emit_at_once
        _run_single_test(tmp_path, emit_at_once=1000, exp_data_shape=(1, 20, 5), exp_annot_shape=(1, 20, 1))

    def test_annot_empty(self, tmp_path):
        _run_single_test(tmp_path, emit_at_once=1, exp_data_shape=(20, 1, 5), empty_annot=True)
