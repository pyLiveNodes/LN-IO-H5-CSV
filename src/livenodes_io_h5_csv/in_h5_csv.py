import asyncio
import glob
import pandas as pd
import h5py
import os
import numpy as np

from typing import NamedTuple

from livenodes.producer_async import Producer_async
from livenodes_common_ports.ports import Ports_empty, Port_Timeseries, Port_Number, Port_List_Str, Port_ListUnique_Str


class Ports_out(NamedTuple):
    ts: Port_Timeseries = Port_Timeseries("TimeSeries")
    channels: Port_ListUnique_Str = Port_ListUnique_Str("Channel Names")
    annot: Port_List_Str = Port_List_Str("Annotation")
    percent: Port_Number = Port_Number("Percent")


def read_data(f):
    try:
        with h5py.File(f, 'r') as data_file:
            data = data_file.get('data')[:]  # Load into mem

        annot = []

        if glob.glob(csv_file := f.replace(".h5", ".csv")):
            if (annot_read := pd.read_csv(csv_file, delimiter=',').to_numpy()).size > 0:
                annot = list(np.concatenate([[str(act)] * (end - start) for start, end, act in annot_read]))

        return data, annot

    except (OSError, TypeError):
        print('Could not open file, skipping', f)
        return np.array([[]]), []


class In_h5_csv(Producer_async):
    """Reads and sends HDF5/.h5 data and corresponding .csv annotation.

    Each batch contains the entire dataset of a file. For custom batch sizes
    and real-time simulation use the `In_playback_h5_csv` node with its
    `emit_at_once` and `sample_rate` settings instead.

    By default, channels sent via the Channel Names port are named ascending
    from 0. These can be overwritten by passing a new list of names as a meta
    parameter.

    If a valid annotation CSV file with the same base name is found, its
    content is sent via the Annotation port. .h5 and .csv files created via
    the `Out_h5_csv` node automatically follow this format.

    After each file is processed, the Percent port is also updated accordingly.
    One common usage example is triggering model training once all files are
    sent.

    Attributes
    ----------
    files : str
        glob pattern for files to include. Should end with ".h5" extension.
        Common examples are single files ("../data/data.h5") or all files in a
        directory ("../data/*.h5").
    meta : dict
        Dict of meta parameters.

        * 'sample_rate' : int
            Sample rate to simulate.
        * 'channel_names' : list of unique str, optional
            List of channel names for `channels` port.

    Ports Out
    ---------
    ts : Port_TimeSeries
        HDF5/.h5 data file contents as TimeSeries.
    channels : Port_ListUnique_Str
        List of channel names. Can be overwritten using the `meta` attribute.
    annot : Port_List_Str
        List of annotation strings, one per data sample. Only sent if valid
        .csv annotation file found. Otherwise empty list.
    percent : Port_Number
        Percentage of files sent so far. Float values from 0.0 to 1.0.
    """

    ports_in = Ports_empty()
    ports_out = Ports_out()

    category = "Data Source"
    description = ""

    example_init = {'name': 'In h5 CSV', 'files': 'data/*.h5', 'meta': {'channels': ["Channel 1"]}}

    def __init__(self, name="In h5 CSV", files='data', meta={}, **kwargs):
        super().__init__(name, **kwargs)
        self.files = files
        self.meta = meta
        self.channels = meta.get('channels')

    def _settings(self):
        return {"files": self.files, "meta": self.meta}

    async def _async_run(self):
        files = glob.glob(self.files)
        n_files = len(files)
        self.info(f'Files found: {n_files}, {os.getcwd()}')

        for i, f in enumerate(files):
            self.info(f'Processing {f}')

            ts, annot = read_data(f)

            channels = [str(x) for x in list(range(ts.shape[1]))]
            percent = round((i + 1) / n_files, 2)

            if self.channels is not None:
                # Fill up, but only as far as smaller list allows
                until = min(len(channels), len(self.channels))
                channels[:until] = self.channels[:until]

            yield self.ret(ts=ts, channels=channels, annot=annot, percent=percent)

            await asyncio.sleep(0)  # so other tasks can run
