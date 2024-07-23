import asyncio
import time
import numpy as np
import glob, random
import h5py
import pandas as pd
import os

from livenodes.producer_async import Producer_async


from livenodes_common_ports.ports import Port_Timeseries, Port_List_Str, Port_List_Str, Ports_empty
from typing import NamedTuple


class Ports_out(NamedTuple):
    ts: Port_Timeseries = Port_Timeseries("TimeSeries")
    channels: Port_List_Str = Port_List_Str("Channel Names")
    annot: Port_List_Str = Port_List_Str("Annotation")


class In_playback_h5_csv(Producer_async):
    """
    Playsback previously recorded data.

    Expects the following setup variables:
    - files (str): glob pattern for files
    - sample_rate (number): sample rate to simulate in frames per second
    - batch_size (int, default=5): number of frames that are sent at the same time -> not implemented yet
    """

    ports_in = Ports_empty()
    ports_out = Ports_out()

    category = "Data Source"
    description = ""

    example_init = {
        'name': 'Playback',
        'files': './data/*.h5',
        'meta': {'sample_rate': 1000, 'targets': ['stand'], 'channels': ['channel 1']},
        'annotation_holes': 'stand',
        'emit_at_once': 20,
    }

    # TODO: consider using a file for meta data instead of dictionary...
    def __init__(self, files, meta, loop=True, emit_at_once=1, annotation_holes="stand", name="Playback", compute_on="1", **kwargs):
        super().__init__(name=name, compute_on=compute_on, **kwargs)

        self.meta = meta
        self.files = files
        self.loop = loop
        self.emit_at_once = emit_at_once
        self.annotation_holes = annotation_holes

        self.sample_rate = meta.get('sample_rate')
        self.targets = meta.get('targets')
        self.channels = meta.get('channels')

    def _settings(self):
        return {
            "emit_at_once": self.emit_at_once,
            "files": self.files,
            "loop": self.loop,
            "meta": self.meta,
            "annotation_holes": self.annotation_holes,
        }

    async def _async_run(self):
        """
        Streams the data and calls frame callbacks for each frame.
        """
        fs = glob.glob(self.files)
        sleep_time = 1.0 / (self.sample_rate / self.emit_at_once)
        print(sleep_time, self.sample_rate, self.emit_at_once)
        last_time = time.time()

        # target_to_id = {key: key for i, key in enumerate(self.targets)}

        self.ret_accu(self.channels, port=self.ports_out.channels)
        ctr = -1

        # if self.annotation_holes not in target_to_id:
        #     raise Exception('annotation filler must be in known targets. got',
        #                     self.annotation_holes, target_to_id.keys())

        # TODO: add sigkill handler
        loop = True  # Should run at least once either way
        while loop:
            loop = self.loop
            f = random.choice(fs)
            ctr += 1
            self.info(ctr, f)

            # Read and send data from file
            with h5py.File(f, "r") as dataFile:
                dataSet = dataFile.get("data")
                start = 0
                end = len(dataSet)
                data = dataSet[start:end]  # load into mem

                # Prepare framewise annotation to be send
                targs = []
                if os.path.exists(f.replace('.h5', '.csv')):
                    ref = pd.read_csv(f.replace('.h5', '.csv'))

                    targs = []
                    last_end = 0
                    filler = self.annotation_holes  # use stand as filler for unknown. #Hack! TODO: remove
                    for _, row in ref.iterrows():
                        targs.append([filler] * (row['start'] - last_end))
                        # +1 as the numbers are samples, ie the last sample still has that label
                        targs.append(
                            [str(row['act'])] * (row['end'] - row['start'])
                        )  # +1 as the numbers are samples, ie the last sample still has that label
                        last_end = row['end']
                    targs.append([filler] * (len(data) - last_end))
                    targs = list(np.concatenate(targs))

                    print(f, end, len(targs))

                    # last_end = 0
                    # for _, row in ref.iterrows():
                    #     # This is hacky af, but hey... achieves that we cann playback annotaitons with holes (and fill those) and also playback annotations without holes
                    #     # if self.annotation_holes in target_to_id:
                    #     targs += [self.annotation_holes] * (
                    #         row['start'] - last_end
                    #         )  # use stand as filler for unknown. #Hack! TODO: remove
                    #     targs += [row['act'].strip()] * (row['end'] - row['start'])
                    #     last_end = row['end']
                    # # if self.annotation_holes in
                    # targs += [self.annotation_holes] * (len(data) - last_end)

                # TODO: for some reason i have no fucking clue about using read_data results in the annotation plot in draw recog to be wrong, although the targs are exactly the same (yes, if checked read_data()[1] == targs)...
                for i in range(start, end, self.emit_at_once):
                    # The data format is always: (time, channel)
                    # self.debug(data[i:i+self.emit_at_once][0])
                    result_data = np.array(data[i : i + self.emit_at_once])
                    # n_channels = len(self.channels)
                    # tmp_data = np.array([np.array([np.arange(i, i + self.emit_at_once) / 1000] * n_channels).T])
                    # print(tmp_data.shape)
                    self.ret_accu(result_data, port=self.ports_out.ts)

                    if len(targs[i : i + self.emit_at_once]) > 0:
                        # use reshape -1, as the data can also be shorter than emit_at_once and will be adjusted accordingly
                        self.ret_accu(targs[i : i + self.emit_at_once], port=self.ports_out.annot)

                    # self.debug(time.time(), last_time + sleep_time, time.time() < last_time + sleep_time)
                    while time.time() < last_time + sleep_time:
                        await asyncio.sleep(0.0001)

                    last_time = time.time()

                    yield self.ret_accumulated()
