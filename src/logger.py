import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

# Create logs directory (always relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Daily rotating log file
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d')}.log"
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)

# Standard log format (industry style)
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"

# Configure logging with file + console handlers
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        TimedRotatingFileHandler(
            LOG_FILE_PATH, when="midnight", interval=1, backupCount=7, encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

# Export logger object
logger = logging.getLogger(__name__)
