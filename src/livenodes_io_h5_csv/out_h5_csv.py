import time
import datetime
import h5py
import json
import os
import numpy as np

from livenodes.node import Node


from livenodes_common_ports.ports import Port_ListUnique_Str, Port_Timeseries, Port_List_Str, Ports_empty
from typing import NamedTuple


class Ports_in(NamedTuple):
    ts: Port_Timeseries = Port_Timeseries("TimeSeries")  # ie (time, channel)
    channels: Port_ListUnique_Str = Port_ListUnique_Str("Channel Names")
    annot: Port_List_Str = Port_List_Str("Annotation")  # ie (time, channel), where there should be only one channel
    # NOTE: Simplify annot or turn into TimeSeries? -> TS would allow compatibility with Window node


class Out_h5_csv(Node):
    """Writes data to HDF5/.h5 files and (optionally) annotation to .csv files.

    Once processing has finished, data is written to a HDF5/.h5 file with the
    current timestamp string as base name. More specifically, a dataset named
    "data" is created with data samples in rows and channels in columns.

    If the Annotation port is connected, the annotation will also be saved to a
    .csv file with the same base name. Each line contains a triple of the start
    sample number, the end sample number (exclusive), and the respective
    annotation string.

    Files created using this node are automatically compatible with the
    `In_h5_csv` and `In_playback_h5_csv` nodes.

    Attributes
    ----------
    folder : str
        folder to save data files to.
    compute_on : str
        Multiprocessing/-threading location to run node on. Advanced feature;
        see LiveNodes core docs for details.

    Ports In
    --------
    ts : Port_TimeSeries
        Data batch to be saved to HDF5/.h5 file.
    channels : Port_ListUnique_Str
        List of channel names. Only required on the first process invocation.
    annot : Port_List_Str, optional
        List of annotation strings corresponding to data batch to be saved to
        .csv file, with one string per data sample. Ignored if not connected.
    """

    ports_in = Ports_in()
    ports_out = Ports_empty()

    category = "Save"
    description = ""

    example_init = {'name': 'Save', 'folder': './data/'}

    def __init__(self, folder, name="Save", compute_on="", **kwargs):
        # NOTE: Previous default compute_on="1:1" often caused file write failures, investigate before changing back.
        super().__init__(name, compute_on=compute_on, **kwargs)

        self.folder = folder

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # NOTE: we can create the filename here (although debatable)
        # but we cannot create the file here, as no processing is being done or even planned yet (this might just be create_pipline)
        self.outputFilename = f"{self.folder}{datetime.datetime.fromtimestamp(time.time())}"
        print("Saving to:", self.outputFilename)

        self.outputFile = None
        self.outputDataset = None

        self.outputFileAnnotation = None
        self.last_annotation = None

        self.channels = None

        self.running = False

        self.buffer = []

    def _settings(self):
        return {"folder": self.folder}

    def _onstart(self):
        if not self.running:
            self.running = True
            self.outputFile = h5py.File(self.outputFilename + '.h5', 'w')
            if self._is_input_connected(self.ports_in.annot):
                self.outputFileAnnotation = open(f"{self.outputFilename}.csv", "w")
                self.outputFileAnnotation.write("start,end,act\n")
            self.info('Created Files')

    def _onstop(self):
        if self.running:
            self.running = False
            if self._is_input_connected(self.ports_in.annot) and self.last_annotation is not None:
                self.outputFileAnnotation.write(f"{self.last_annotation[1]},{self.last_annotation[2]},{self.last_annotation[0]}")
                self.outputFileAnnotation.close()

            self._append_buffer_to_file()
            self.outputFile.close()
            self.info('Stopped writing out and closed files')

    def _should_process(self, ts=None, channels=None, annot=None):
        return (
            ts is not None
            and (self.channels is not None or channels is not None)
            and (annot is not None or not self._is_input_connected(self.ports_in.annot))
        )

    def process(self, ts, channels=None, annot=None, **kwargs):

        if channels is not None:
            self.channels = channels

            if self.outputDataset is None:

                self.outputDataset = self.outputFile.create_dataset(
                    "data", (0, len(self.channels)), maxshape=(None, len(self.channels)), dtype=np.array(ts).dtype
                )

        if channels is not None:
            self._write_meta({'channels': channels})

        if annot is not None:
            self.receive_annotation(annot)

        self.buffer.append(ts)
        if len(self.buffer) > 100:
            self._append_buffer_to_file()

    def _append_buffer_to_file(self):
        if len(self.buffer) >= 1:
            d = np.concatenate(self.buffer, axis=0)  # concat buffer and write to file
            self.buffer = []
            self.outputDataset.resize(self.outputDataset.shape[0] + len(d), axis=0)
            self.outputDataset[-len(d) :] = d

    def receive_annotation(self, data_frame, **kwargs):
        if self.last_annotation is None:
            self.last_annotation = (data_frame[0], 0, 0)

        # Group succeding entries together
        # TODO: consider using sparcity libraries instead of custom implementation
        for annotation in data_frame:
            if annotation == self.last_annotation[0]:
                self.last_annotation = (annotation, self.last_annotation[1], self.last_annotation[2] + 1)
            else:
                self.outputFileAnnotation.write(f"{self.last_annotation[1]},{self.last_annotation[2]},{self.last_annotation[0]}\n")
                self.last_annotation = (annotation, self.last_annotation[2], self.last_annotation[2] + 1)

    def _read_meta(self):
        if not os.path.exists(f"{self.outputFilename}.json"):
            return {}
        with open(f"{self.outputFilename}.json", 'r') as f:
            return json.load(f)

    def _write_meta(self, setting):
        with open(f"{self.outputFilename}.json", 'w') as f:
            json.dump(setting, f, indent=2)
