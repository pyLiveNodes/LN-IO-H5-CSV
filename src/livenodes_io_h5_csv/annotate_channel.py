import numpy as np

from livenodes.node import Node

from livenodes_common_ports.ports import Ports_ts_channels, Port_List_Str, Port_Timeseries
from typing import NamedTuple


class Ports_out(NamedTuple):
    ts: Port_Timeseries = Port_Timeseries("TimeSeries")
    channels: Port_List_Str = Port_List_Str("Channel Names")
    annot: Port_List_Str = Port_List_Str("Annotation")


class Annotate_channel(Node):
    ports_in = Ports_ts_channels()
    ports_out = Ports_out()

    category = "Annotation"
    description = ""

    example_init = {'name': 'Channel Annotation', 'channel_name': 'Pushbutton', 'targets': ['Pressed', 'Released']}

    def __init__(self, channel_name, targets, name="Channel Annotation", **kwargs):
        super().__init__(name=name, **kwargs)

        self.channel_name = channel_name
        self.targets = targets
        self.name = name

        self.idx = None

    def _settings(self):
        return {
            "name": self.name,
            "channel_name": self.channel_name,
            "targets": self.targets,
        }

    def _should_process(self, ts=None, channels=None):
        return ts is not None and (self.idx is not None or channels is not None)

    def process(self, ts, channels=None, **kwargs):
        if channels is not None:
            self.idx = np.array(channels) == self.channel_name
            self.ret_accu(np.array(channels)[~self.idx], port=self.ports_out.channels)

        self.ret_accu(ts[:, ~self.idx], port=self.ports_out.ts)
        self.ret_accu(np.where(ts[:, self.idx].flatten() > 0, self.targets[1], self.targets[0]), port=self.ports_out.annot)
        return self.ret_accumulated()
