import asyncio
import time
import numpy as np
import glob, random

from .abstract_in_h5_csv import Abstract_in_h5_csv


from livenodes_common_ports.ports import Port_Timeseries, Port_List_Str, Port_List_Str, Ports_empty
from typing import NamedTuple


class Ports_out(NamedTuple):
    ts: Port_Timeseries = Port_Timeseries("TimeSeries")
    channels: Port_List_Str = Port_List_Str("Channel Names")
    annot: Port_List_Str = Port_List_Str("Annotation")


class In_playback_h5_csv(Abstract_in_h5_csv):
    """Reads and plays back HDF5/.h5 data and corresponding .csv annotation.

    Batch size depends on the `emit_at_once` parameter, e.g. value 5 means that
    each batch will contain 5 samples. Higher values increase processing
    efficiency, but reduce the effective frame rate for following nodes (since
    playback "waits" until batch is filled before sending). Values around 10-20
    are recommended. For sending whole files as one batch use the `In_h5_csv`
    node instead.

    This node simulates real-time usage via the `sample_rate` meta parameter.
    The time between each `process` invocation depends on both `emit_at_once`
    and `sample_rate`.

    If a valid annotation CSV file with the same base name is found, its
    content is sent via the Annotation port. .h5 and .csv files created via
    the `Out_h5_csv` node automatically follow this format.

    Attributes
    ----------
    files : str
        glob pattern for files to include. Should end with ".h5" extension.
        Common examples are single files ("../data/data.h5") or all files in a
        directory ("../data/*.h5"). In the case of multiple files, a random
        file is selected.
    loop : bool
        Whether to loop playback or stop when a file's data is sent. If the
        `files` glob pattern contains multiple files and `loop` is `True`, the
        file selection is re-randomized on each loop.
    sample_rate : int
        Sample rate to simulate in frames per second.
    emit_at_once : int
        Batch size.
    compute_on : str
        Multiprocessing/-threading location to run node on. Advanced feature;
        see LiveNodes core docs for details.
    meta : dict
        Dict of meta parameters.

        * 'sample_rate' : int
            Sample rate to simulate.
        * 'channel_names' : list of unique str
            List of channel names for `channels` port.
        * 'targets' : list of unique str
            [DEPRECATED, UNUSED] List of possible target/annotation strings.

    annotation_holes : str
        [DEPRECATED] Fallback annotation string to use if missing for a sample.

    Ports Out
    ---------
    ts : Port_TimeSeries
        Data batch of size `emit_at_once` read from input HDF5/.h5 file.
    channels : Port_ListUnique_Str
        List of channel names defined with the `meta` attribute. Sent only once
        on the first batch.
    annot : Port_List_Str
        List of annotation strings corresponding to data batch, with one string
        per data sample. Only sent if valid .csv annotation file found.
        Otherwise empty list (if file does not exist) or list filled with
        `annotation_holes` backup string (if file exists, but empty).
    """

    ports_out = Ports_out()

    example_init = {
        'name': 'Playback',
        'files': './data/*.h5',
        'meta': {'sample_rate': 1000, 'channels': None},
        'emit_at_once': 10,
    }

    def __init__(self, files, meta, loop=True, emit_at_once=10, name="Playback", compute_on="1", **kwargs):
        super().__init__(name=name, files=files, meta=meta, compute_on=compute_on, **kwargs)
        self.loop = loop
        self.emit_at_once = emit_at_once
        self.sample_rate = meta.get('sample_rate')

    def _settings(self):
        return {
            "emit_at_once": self.emit_at_once,
            "files": self.files,
            "loop": self.loop,
            "meta": self.meta,
        }

    async def _async_run(self):
        """
        Streams the data and calls frame callbacks for each frame.
        """
        fs = glob.glob(self.files)
        sleep_time = 1.0 / (self.sample_rate / self.emit_at_once)
        last_time = time.time()

        ctr = -1

        # TODO: add sigkill handler
        loop = True  # Should run at least once either way
        while loop:
            loop = self.loop
            f = random.choice(fs)
            ctr += 1
            self.info(ctr, f)

            ts, annot, channels = self._read_data(f)

            channels = self._overwrite_channels(channels, ts.shape[1])

            if ctr == 0:
                self.ret_accu(channels, port=self.ports_out.channels)

            # TODO: for some reason i have no fucking clue about using read_data results in the annotation plot in draw recog to be wrong, although the targs are exactly the same (yes, if checked read_data()[1] == targs)...
            for i in range(0, len(ts), self.emit_at_once):
                result_data = np.array(ts[i : i + self.emit_at_once])
                self.ret_accu(result_data, port=self.ports_out.ts)

                if len(annot[i : i + self.emit_at_once]) > 0:
                    # use reshape -1, as the data can also be shorter than emit_at_once and will be adjusted accordingly
                    self.ret_accu(annot[i : i + self.emit_at_once], port=self.ports_out.annot)

                while time.time() < last_time + sleep_time:
                    await asyncio.sleep(0.0001)

                last_time = time.time()

                yield self.ret_accumulated()
