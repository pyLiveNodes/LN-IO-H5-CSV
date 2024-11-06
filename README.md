[![Format and Test](https://github.com/pyLiveNodes/LN-IO-H5-CSV/actions/workflows/format_test.yml/badge.svg)](https://github.com/pyLiveNodes/LN-IO-H5-CSV/actions/workflows/format_test.yml)
[![Publish](https://github.com/pyLiveNodes/LN-IO-H5-CSV/actions/workflows/publish.yml/badge.svg)](https://github.com/pyLiveNodes/LN-IO-H5-CSV/actions/workflows/publish.yml)

# LiveNodes IO H5 CSV

The LiveNodes IO H5 CSV package provides nodes for data input and output using HDF5/.h5 data files. Optionally, this also includes data annotations via .csv
files.

For HDF5/.h5 data files, the expected format is a dataset named "data" with samples in rows and channels in columns.

For .csv annotation files, each line contains a triple of the start sample number, the end sample number (exclusive), and the respective annotation string.
See the following example:

```
start,end,act
0,30,Stand
30,50,Walk
50,80,Run
80,100,Stand
```

Files created using the `Out_h5_csv` node automatically follow these formats.

## Nodes in this package
| Node                  | Purpose                                                                  |
| --------------------- | ------------------------------------------------------------------------ |
| `Annotate_channel`    | Creates annotation based on the specified channel and target names.      |
| `In_h5_csv`           | Reads and sends HDF5/.h5 data and corresponding .csv annotation.         |
| `In_playback_h5_csv`  | Reads and plays back HDF5/.h5 data and corresponding .csv annotation.    |
| `Out_h5_csv`          | Writes data to HDF5/.h5 files and (optionally) annotation to .csv files. |

## About LiveNodes
[LiveNodes](https://livenodes.pages.csl.uni-bremen.de/livenodes/index.html) are small units of computation for digital signal processing in python. They are connected multiple synced channels to create complex graphs for real-time applications. Each node may provide a GUI or Graph for live interaction and visualization.

Any contribution is welcome! These projects take more time, than I can muster, so feel free to create issues for everything that you think might work better and feel free to create a MR for them as well!

Have fun and good coding!

Yale

## Installation

`pip install ln_io_h5_csv `

## Docs

You can find the docs [here](https://livenodes.pages.csl.uni-bremen.de/packages/ln_io_h5_csv/readme.html).

## Restrictions

None, just pure python and numpy.
