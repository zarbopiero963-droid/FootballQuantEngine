from pathlib import Path

from app.version_info import get_version_info


class ProjectSummary:

    def build(self):

        info = get_version_info()

        return {
            "name": info["name"],
            "version": info["version"],
            "author": info["author"],
            "description": info["description"],
            "has_readme": Path("README.md").exists(),
            "has_spec": Path("football_quant_engine.spec").exists(),
            "has_license": Path("LICENSE").exists(),
            "has_contributing": Path("CONTRIBUTING.md").exists(),
            "has_requirements": Path("requirements.txt").exists(),
            "has_outputs_dir": Path("outputs").exists(),
            "core_modules": {
                "analysis": Path("analysis").exists(),
                "app": Path("app").exists(),
                "config": Path("config").exists(),
                "data": Path("data").exists(),
                "database": Path("database").exists(),
                "engine": Path("engine").exists(),
                "export": Path("export").exists(),
                "features": Path("features").exists(),
                "models": Path("models").exists(),
                "notifications": Path("notifications").exists(),
                "plugins": Path("plugins").exists(),
                "simulation": Path("simulation").exists(),
                "strategies": Path("strategies").exists(),
                "training": Path("training").exists(),
                "ui": Path("ui").exists(),
                "tests": Path("tests").exists(),
            },
        }
