from abc import ABC, abstractmethod
import glob
import json
import pandas as pd
import h5py
import numpy as np

from typing import NamedTuple

from livenodes.producer_async import Producer_async
from livenodes_common_ports.ports import Ports_empty, Port_Timeseries, Port_ListUnique_Str


class Ports_out(NamedTuple):
    ts: Port_Timeseries = Port_Timeseries("TimeSeries")
    channels: Port_ListUnique_Str = Port_ListUnique_Str("Channel Names")
    annot: Port_Timeseries = Port_Timeseries("Annotation")


class Abstract_in_h5_csv(Producer_async, ABC):
    """Abstract base class for `in_(playback_)h5_csv` nodes."""

    ports_in = Ports_empty()
    ports_out = Ports_out()

    category = "Data Source"
    description = ""

    def __init__(self, name="In h5 CSV", files='data', meta={}, **kwargs):
        super(Producer_async, self).__init__(name, **kwargs)
        self.files = files
        self.meta = meta
        self.channels = meta.get('channels')

    def _settings(self):
        return {"files": self.files, "meta": self.meta}

    @abstractmethod
    async def _async_run(self):
        raise NotImplementedError

    def _overwrite_channels(self, channels, n_channels):
        new_channels = [str(x) for x in list(range(n_channels))]  # Default values
        if self.channels is not None and self.channels != [""]:  # Single empty string in Smart Studio equivalent to nothing set
            new_channels = self.channels
        elif channels:
            new_channels = channels

        if len(new_channels) != n_channels:
            raise ValueError(
                f"Number of new channel names must equal number of actual channels. List {channels} is length {len(channels)}, but should be {n_channels}"
            )
        return new_channels

    @staticmethod
    def _read_data(f):
        try:
            with h5py.File(f, 'r') as data_file:
                data = data_file.get('data')[:]  # Load into mem

            channels = []
            if glob.glob(json_file := f.replace(".h5", ".json")):
                with open(json_file, 'r') as json_f:
                    entries = json.load(json_f)
                    if "channels" in entries:
                        channels = entries.get("channels")

            annot = []
            if glob.glob(csv_file := f.replace(".h5", ".csv")):
                if (ref := pd.read_csv(csv_file, delimiter=',')).size > 0:
                    last_end = 0
                    rows = ref.iterrows()
                    for _, row in rows:
                        annot.append([""] * (row['start'] - last_end))
                        annot.append([str(row['act'])] * (row['end'] - row['start']))
                        last_end = row['end']
                    annot.append([""] * (len(data) - last_end))
                    annot = list(np.concatenate(annot))

            return data, channels, annot

        except (OSError, TypeError):
            print('Could not open file, skipping', f)
            return np.array([[]]), [], np.array([[]])
