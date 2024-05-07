import torchcompat.plugins
from torchcompat.core.load import discover_plugins


def test_plugins():
    plugins = discover_plugins(torchcompat.plugins)

    assert len(plugins) == 1
