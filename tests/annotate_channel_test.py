import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

from livenodes import Graph

from livenodes_io_h5_csv.annotate_channel import Annotate_channel
from livenodes_io_python.out_python import Out_python
from livenodes_io_python.in_python import In_python


class TestProcessing:

    def test_data(self):
        data = np.arange(100).reshape((20, 1, 5))  # 20 samples with 5 channels each

        data_in = In_python(name="A", data=data)
        # Channels defined here not part of test; only needed for Out_h5_csv to work
        channels_in = In_python(name="B", data=[["A", "B", "C", "D", "E"]])

        annotate_channel = Annotate_channel(channel_name="A", targets=["Idle", "Tap"])
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

        print(collect_data.get_state())
        print(collect_channels.get_state())
        print(collect_annot.get_state())
