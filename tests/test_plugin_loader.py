from engine.plugin_loader import PluginLoader


def test_plugin_loader_loads_plugins():

    loader = PluginLoader()

    plugins = loader.load_plugins()

    assert isinstance(plugins, list)
    assert len(plugins) >= 1
