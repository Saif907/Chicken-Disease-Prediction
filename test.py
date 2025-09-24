
from pathlib import Path

# __file__ = current file path (configuration.py)
project_root = Path(__file__).resolve().parents[2]  # 2 levels up from src/config
print(project_root)
