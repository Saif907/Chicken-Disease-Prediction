
from pathlib import Path
from src.utils.utils import read_yaml
from src.exception import CustomException

class ConfigManager:
    def __init__(self, config_path: Path = None, params_path: Path = None):
        try:
            # ------------------- Project root -------------------
            project_root = Path(__file__).resolve().parents[2]  # src/config -> project root

            # ------------------- YAML paths -------------------
            self.config_path = Path(config_path) if config_path else project_root / "config" / "config.yaml"
            self.params_path = Path(params_path) if params_path else project_root / "config" / "params.yaml"

            # Convert to absolute paths
            self.config_path = self.config_path.resolve()
            self.params_path = self.params_path.resolve()

            # ------------------- Load YAML files -------------------
            print(f"Loading config from: {self.config_path}")  # debug
            print(f"Loading params from: {self.params_path}")  # debug
            self.config = read_yaml(self.config_path)
            self.params = read_yaml(self.params_path)

        except Exception as e:
            raise CustomException(e)

    # ------------------- Config getters -------------------
    def get_data_ingestion_config(self):
        return self.config.get("data_ingestion", {})

    def get_model_config(self):
        return self.config.get("model", {})

    def get_log_config(self):
        return self.config.get("logs", {})

    def get_params(self):
        return self.params
