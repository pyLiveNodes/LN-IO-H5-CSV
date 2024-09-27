import glob
from os.path import dirname, basename, isfile, join
from importlib.metadata import entry_points

import pytest


@pytest.fixture
def discovered_modules():
    exclude = ['__init__', 'utils', 'ports', 'abstract']
    modules = glob.glob(join(dirname(__file__), '../src/ln_io_h5_csv/', "*.py"))
    names = [basename(f)[:-3] for f in modules if isfile(f)]
    return [f for f in names if not any(f.startswith(ex) for ex in exclude)]


class TestProcessing:
    def test_modules_discoverable(self, discovered_modules):
        assert len(discovered_modules) > 0

    def test_all_declared(self, discovered_modules):
        livnodes_entrypoints = [x.name for x in entry_points(group='livenodes.nodes')]

        print(set(discovered_modules).difference(set(livnodes_entrypoints)))
        assert set(discovered_modules) <= set(livnodes_entrypoints)

    def test_loads_class(self):
        in_playback_h5_csv = [x.load() for x in entry_points(group='livenodes.nodes') if x.name == 'in_playback_h5_csv'][0]
        from ln_io_h5_csv.in_playback_h5_csv import In_playback_h5_csv

        assert in_playback_h5_csv == In_playback_h5_csv

    def test_all_loadable(self):
        for x in entry_points(group='livenodes.nodes'):
            x.load()
