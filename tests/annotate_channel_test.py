from typing import NamedTuple
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

from livenodes import Graph

from livenodes_io_h5_csv.annotate_channel import Annotate_channel
from livenodes_io_python.out_python import Out_python
from livenodes_io_python.in_python import In_python


class Results(NamedTuple):
    ts: np.ndarray
    channels: np.ndarray
    annot: np.ndarray


def _run_test_pipeline(data, channels):
    data_in = In_python(name="A", data=data)
    # Channels defined here not part of test; only needed for Out_h5_csv to work
    channels_in = In_python(name="B", data=channels)

    annotate_channel = Annotate_channel(channel_name="Annot", targets=["Idle", "Tap"])
    annotate_channel.add_input(data_in, emit_port=data_in.ports_out.any, recv_port=annotate_channel.ports_in.ts)
    annotate_channel.add_input(channels_in, emit_port=channels_in.ports_out.any, recv_port=annotate_channel.ports_in.channels)

    collect_data = Out_python(name="C")
    collect_data.add_input(annotate_channel, emit_port=annotate_channel.ports_out.ts, recv_port=collect_data.ports_in.any)

    collect_channels = Out_python(name="D")
    collect_channels.add_input(annotate_channel, emit_port=annotate_channel.ports_out.channels, recv_port=collect_channels.ports_in.any)

    collect_annot = Out_python(name="E")
    collect_annot.add_input(annotate_channel, emit_port=annotate_channel.ports_out.annot, recv_port=collect_annot.ports_in.any)

    g = Graph(start_node=data_in)
    g.start_all()
    g.join_all()
    g.stop_all()

    return Results(collect_data.get_state(), collect_channels.get_state(), collect_annot.get_state())


class TestProcessing:

    def test_emit_1(self):
        data = np.arange(500).reshape((100, 1, 5))

        rng = np.random.default_rng(seed=0)
        annot = rng.integers(low=0, high=2, size=(100, 1, 1))
        expected_annot = np.where(annot.flatten() > 0, "Tap", "Idle").reshape(100, 1)

        merged_data = np.append(data, annot, axis=2)

        channels = np.array([["CH1", "CH2", "CH3", "CH4", "CH5", "Annot"]])
        expected_channels = channels[:, :-1]

        actual_data, actual_channels, actual_annot = _run_test_pipeline(merged_data, channels)

        np.testing.assert_equal(actual_data, data)
        np.testing.assert_equal(actual_channels, expected_channels)
        np.testing.assert_equal(actual_annot, expected_annot)

    def test_emit_2(self):
        data = np.arange(500).reshape((50, 2, 5))

        rng = np.random.default_rng(seed=1)
        annot = rng.integers(low=0, high=2, size=(50, 2, 1))
        expected_annot = np.where(annot.flatten() > 0, "Tap", "Idle").reshape(50, 2)

        merged_data = np.append(data, annot, axis=2)

        channels = np.array([["CH1", "CH2", "CH3", "CH4", "CH5", "Annot"]])
        expected_channels = channels[:, :-1]

        actual_data, actual_channels, actual_annot = _run_test_pipeline(merged_data, channels)

        np.testing.assert_equal(actual_data, data)
        np.testing.assert_equal(actual_channels, expected_channels)
        np.testing.assert_equal(actual_annot, expected_annot)

    def test_emit_5(self):
        data = np.arange(500).reshape((20, 5, 5))

        rng = np.random.default_rng(seed=2)
        annot = rng.integers(low=0, high=2, size=(20, 5, 1))
        expected_annot = np.where(annot.flatten() > 0, "Tap", "Idle").reshape(20, 5)

        merged_data = np.append(data, annot, axis=2)

        channels = np.array([["CH1", "CH2", "CH3", "CH4", "CH5", "Annot"]])
        expected_channels = channels[:, :-1]

        actual_data, actual_channels, actual_annot = _run_test_pipeline(merged_data, channels)

        np.testing.assert_equal(actual_data, data)
        np.testing.assert_equal(actual_channels, expected_channels)
        np.testing.assert_equal(actual_annot, expected_annot)

    def test_annot_first(self):
        data = np.arange(500).reshape((20, 5, 5))

        rng = np.random.default_rng(seed=3)
        annot = rng.integers(low=0, high=2, size=(20, 5, 1))
        expected_annot = np.where(annot.flatten() > 0, "Tap", "Idle").reshape(20, 5)

        merged_data = np.append(annot, data, axis=2)

        channels = np.array([["Annot", "CH1", "CH2", "CH3", "CH4", "CH5"]])
        expected_channels = channels[:, 1:]

        actual_data, actual_channels, actual_annot = _run_test_pipeline(merged_data, channels)

        np.testing.assert_equal(actual_data, data)
        np.testing.assert_equal(actual_channels, expected_channels)
        np.testing.assert_equal(actual_annot, expected_annot)

    def test_annot_middle(self):
        data1 = np.arange(200).reshape((20, 5, 2))
        data2 = np.arange(300).reshape((20, 5, 3))
        expected_data = np.append(data1, data2, axis=2)

        rng = np.random.default_rng(seed=3)
        annot = rng.integers(low=0, high=2, size=(20, 5, 1))
        expected_annot = np.where(annot.flatten() > 0, "Tap", "Idle").reshape(20, 5)

        merged_data = np.append(data1, annot, axis=2)
        merged_data = np.append(merged_data, data2, axis=2)

        channels = np.array([["CH1", "CH2", "Annot", "CH3", "CH4", "CH5"]])
        expected_channels = np.delete(channels, 2, axis=1)

        actual_data, actual_channels, actual_annot = _run_test_pipeline(merged_data, channels)

        np.testing.assert_equal(actual_data, expected_data)
        np.testing.assert_equal(actual_channels, expected_channels)
        np.testing.assert_equal(actual_annot, expected_annot)
