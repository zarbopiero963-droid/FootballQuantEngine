from __future__ import annotations

import importlib
import pkgutil


class PluginLoader:

    def __init__(self, package_name="plugins.models"):

        self.package_name = package_name

    def load_plugins(self):

        package = importlib.import_module(self.package_name)

        plugins = []

        for _, module_name, _ in pkgutil.iter_modules(package.__path__):

            module = importlib.import_module(f"{self.package_name}.{module_name}")

            if hasattr(module, "ModelPlugin"):
                plugins.append(module.ModelPlugin())

        return plugins
