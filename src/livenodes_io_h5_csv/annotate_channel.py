import numpy as np

from livenodes.node import Node

from livenodes_common_ports.ports import Ports_ts_channels, Port_List_Str, Port_Timeseries
from typing import NamedTuple


class Ports_out(NamedTuple):
    ts: Port_Timeseries = Port_Timeseries("TimeSeries")
    channels: Port_List_Str = Port_List_Str("Channel Names")
    annot: Port_List_Str = Port_List_Str("Annotation")


class Annotate_channel(Node):
    """Creates annotation based on the specified channel and target names.

    Two target names can be set via the `targets` attribute. Semantically, this
    is often a pair of an idle and an active state. Examples are
    `["Off", "On"]`, `["Stand", "Walk"]` or `["Nothing", "Tap"]`.

    The sample value of the specified channel then determines which of the two
    target names to send via the Annotation port. If the value is zero or less,
    the first name is selected, otherwise the second. The specified channel is
    also removed from the TimeSeries, leaving only the "actual" data channels.

    Note that this assumes that the specified channel actually contains the
    relevant information like from a hardware button with two different states.
    Using a regular signal data channel may produce nonsensical results. The
    channel must also be a part of those input via the Channel Names port.

    Attributes
    ----------
    channel_name : str
        Name of the input channel used to generate annotation.
    targets : List of str
        List of two annotation target names. Further list elements are ignored.

    Ports In
    --------
    ts : Port_TimeSeries
        Input data batch.
    channels : Port_ListUnique_Str
        List of channel names. Sent only once on the first batch.

    Ports Out
    ---------
    ts : Port_TimeSeries
        Input data batch without the specified annotation channel.
    channels : Port_ListUnique_Str
        List of channel names without the specified annotation channel. Sent
        only once on the first batch.
    annot : Port_List_Str
        List of annotation strings corresponding to data batch, with one string
        per data sample.
    """

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
