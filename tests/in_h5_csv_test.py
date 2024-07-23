from typing import NamedTuple
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

from livenodes import Graph

from livenodes_io_h5_csv.out_h5_csv import Out_h5_csv
from livenodes_io_h5_csv.in_h5_csv import In_h5_csv
from livenodes_io_python.out_python import Out_python
from livenodes_io_python.in_python import In_python


class Out_nodes(NamedTuple):
    ts: Out_python
    channels: Out_python
    annot: Out_python
    percent: Out_python


# Example annotation includes small 1 and 2 size blocks to test these edge cases.
_anot = ["1"] * 5 + ["2"] * 2 + ["3"] * 1 + ["1"] * 2 + ["2"] * 3 + ["3"] * 7


def _prepare_data(tmp_path, generate_annot=False):
    data = np.arange(100).reshape((1, 20, 5))  # 20 samples with 5 channels each

    data_in = In_python(name="A", data=data)
    # Channels defined here not part of test; only needed for Out_h5_csv to work
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


def _run_test_pipeline(tmp_path, channel_names=None):
    read_data = In_h5_csv(name="A", files=f"{tmp_path}/*.h5", meta={'channels': channel_names})

    collect_data = Out_python(name="B")
    collect_data.add_input(read_data, emit_port=read_data.ports_out.ts, recv_port=collect_data.ports_in.any)

    collect_channels = Out_python(name="C")
    collect_channels.add_input(read_data, emit_port=read_data.ports_out.channels, recv_port=collect_channels.ports_in.any)

    collect_anot = Out_python(name="D")
    collect_anot.add_input(read_data, emit_port=read_data.ports_out.annot, recv_port=collect_anot.ports_in.any)

    collect_percent = Out_python(name="D")
    collect_percent.add_input(read_data, emit_port=read_data.ports_out.percent, recv_port=collect_percent.ports_in.any)

    g = Graph(start_node=read_data)
    g.start_all()
    g.join_all()
    g.stop_all()

    return Out_nodes(collect_data, collect_channels, collect_anot, collect_percent)


class TestProcessing:

    def test_data_only(self, tmp_path):

        expected_data = _prepare_data(tmp_path)
        expected_channels = ["0", "1", "2", "3", "4"]

        results = _run_test_pipeline(tmp_path)

        actual_data = np.array(results.ts.get_state())
        actual_channels = results.channels.get_state()[0]

        np.testing.assert_equal(expected_data, actual_data)
        np.testing.assert_equal(expected_channels, actual_channels)

    def test_data_and_channels(self, tmp_path):

        expected_data = _prepare_data(tmp_path)

        channels = ["CH1", "CH2", "CH3", "CH4", "CH5"]

        results = _run_test_pipeline(tmp_path, channels)

        actual_data = np.array(results.ts.get_state())
        actual_channels = results.channels.get_state()[0]

        np.testing.assert_equal(expected_data, actual_data)
        np.testing.assert_equal(channels, actual_channels)

    def test_data_and_more_channels(self, tmp_path):

        expected_data = _prepare_data(tmp_path)

        channels = ["CH1", "CH2", "CH3", "CH4", "CH5", "CH6", "CH7", "CH8"]

        results = _run_test_pipeline(tmp_path, channels)

        actual_data = np.array(results.ts.get_state())
        actual_channels = results.channels.get_state()[0]

        np.testing.assert_equal(expected_data, actual_data)
        np.testing.assert_equal(channels[:5], actual_channels)

    def test_data_and_fewer_channels(self, tmp_path):

        expected_data = _prepare_data(tmp_path)

        channels = ["CH1", "CH2"]
        expected_channels = channels + ["2", "3", "4"]

        results = _run_test_pipeline(tmp_path, channels)

        actual_data = np.array(results.ts.get_state())
        actual_channels = results.channels.get_state()[0]

        np.testing.assert_equal(expected_data, actual_data)
        np.testing.assert_equal(expected_channels, actual_channels)

    def test_annot(self, tmp_path):
        _prepare_data(tmp_path, generate_annot=True)

        results = _run_test_pipeline(tmp_path)

        actual_annot = results.annot.get_state()[0]

        np.testing.assert_equal(_anot, actual_annot)

    def test_annot_empty(self, tmp_path):
        _prepare_data(tmp_path)

        results = _run_test_pipeline(tmp_path)

        actual_annot = results.annot.get_state()[0]

        np.testing.assert_equal([], actual_annot)

    def test_percent_single(self, tmp_path):
        _prepare_data(tmp_path)

        results = _run_test_pipeline(tmp_path)

        expected_percent = [1.0]
        actual_percent = results.percent.get_state()

        np.testing.assert_equal(expected_percent, actual_percent)

    def test_percent_multiple(self, tmp_path):
        for _ in range(4):
            _prepare_data(tmp_path)

        results = _run_test_pipeline(tmp_path)

        expected_percent = [0.25, 0.5, 0.75, 1.0]
        actual_percent = results.percent.get_state()

        np.testing.assert_equal(expected_percent, actual_percent)
