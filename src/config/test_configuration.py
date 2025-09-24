import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config.configuration import ConfigManager
from src.logger import logger
from src.exception import CustomException

def test_config():
    """
    Tests that configuration.yaml and params.yaml are loaded correctly.
    Prints all configs and hyperparameters.
    """
    try:
        config = ConfigManager(
            config_path=Path("../../config/config.yaml"),
            params_path=Path("../../config/params.yaml")
        )

        # Data ingestion config
        data_config = config.get_data_ingestion_config()
        print("=== Data Ingestion Config ===")
        for k, v in data_config.items():
            print(f"{k}: {v}")

        # Model config
        model_config = config.get_model_config()
        print("\n=== Model Config ===")
        for k, v in model_config.items():
            print(f"{k}: {v}")

        # Logging config
        log_config = config.get_log_config()
        print("\n=== Logging Config ===")
        for k, v in log_config.items():
            print(f"{k}: {v}")

        # Hyperparameters
        params = config.get_params()
        print("\n=== Hyperparameters ===")
        for k, v in params.items():
            print(f"{k}: {v}")

        print("\n✅ All configurations loaded successfully!")

    except CustomException as e:
        logger.error(f"Configuration test failed: {e}")

if __name__ == "__main__":
    test_config()
