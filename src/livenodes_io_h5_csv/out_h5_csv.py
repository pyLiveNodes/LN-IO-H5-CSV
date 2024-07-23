import time
import datetime
import h5py
import json
import os
import numpy as np

from livenodes.node import Node


from livenodes_common_ports.ports import Port_Dict, Port_ListUnique_Str, Port_Timeseries, Port_List_Str, Ports_empty
from typing import NamedTuple


class Ports_in(NamedTuple):
    ts: Port_Timeseries = Port_Timeseries("TimeSeries")  # ie (time, channel)
    channels: Port_ListUnique_Str = Port_ListUnique_Str("Channel Names")
    annot: Port_List_Str = Port_List_Str("Annotation")  # ie (time, channel), where there should be only one channel
    meta: Port_Dict = Port_Dict("Meta")


class Out_h5_csv(Node):
    ports_in = Ports_in()
    ports_out = Ports_empty()

    category = "Save"
    description = ""

    example_init = {'name': 'Save', 'folder': './data/Debug/'}

    def __init__(self, folder, name="Save", compute_on="1:1", **kwargs):
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
        self.rec_meta = False

        self.running = False

        self.buffer = []

    def _settings(self):
        return {"folder": self.folder}

    def _onstart(self):
        if not self.running:
            self.running = True
            self.outputFile = h5py.File(self.outputFilename + '.h5', 'w')
            self.outputFileAnnotation = open(f"{self.outputFilename}.csv", "w")
            self.outputFileAnnotation.write("start,end,act\n")
            self.info('Created Files')

    def _onstop(self):
        if self.running:
            self.running = False
            if self.last_annotation is not None:
                self.outputFileAnnotation.write(f"{self.last_annotation[1]},{self.last_annotation[2]},{self.last_annotation[0]}")
            self.outputFileAnnotation.close()

            self._append_buffer_to_file()
            self.outputFile.close()
            self.info('Stopped writing out and closed files')

    def _should_process(self, ts=None, channels=None, meta=None, annot=None):
        return (
            ts is not None
            and (self.channels is not None or channels is not None)
            and ((meta is not None or self.rec_meta) or not self._is_input_connected(self.ports_in.meta))
            and (annot is not None or not self._is_input_connected(self.ports_in.annot))
        )

    def process(self, ts, channels=None, meta=None, annot=None, **kwargs):

        if channels is not None:
            self.channels = channels

            m_dict = self._read_meta()
            m_dict['channels'] = channels
            self._write_meta(m_dict)

            if self.outputDataset is None:

                self.outputDataset = self.outputFile.create_dataset(
                    "data", (0, len(self.channels)), maxshape=(None, len(self.channels)), dtype=np.array(ts).dtype
                )

        if meta is not None:
            self.rec_meta = True
            m_dict = self._read_meta()
            for key, val in meta.items():
                # We'll assume that the channels are always hooked up
                if key != "channels":
                    m_dict[key] = val
            self._write_meta(m_dict)

        if annot is not None:
            # self.receive_annotation(np.vstack(annotation))
            self.receive_annotation(annot)

        # self.outputDataset.resize(self.outputDataset.shape[0] + len(data),
        #                           axis=0)
        # self.outputDataset[-len(data):] = data
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
        # For now lets assume the file is always open before this is called.
        # TODO: re-consider that assumption

        if self.last_annotation is None:
            self.last_annotation = (data_frame[0], 0, 0)

        # Group succeding entries together
        # TODO: consider using sparcity libraries instead of custom implementation
        for annotation in data_frame:
            if annotation == self.last_annotation[0]:
                self.last_annotation = (annotation, self.last_annotation[1], self.last_annotation[2] + 1)
            else:
                # self.debug(f"writing: {self.last_annotation[1]},{self.last_annotation[2]},{self.last_annotation[0]}")
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
