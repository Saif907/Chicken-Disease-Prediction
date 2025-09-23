import sys
from src.logger import logger  # Full package path


def error_message_detail(error):
    """Return detailed error message with file and line number."""
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown"
    line_number = exc_tb.tb_lineno if exc_tb else "?"
    return f"[{file_name}:{line_number}] {str(error)}"

class CustomException(Exception):
    """Custom exception class with detailed logging."""
    def __init__(self, error):
        super().__init__(str(error))
        self.message = error_message_detail(error)
        logger.error(self.message)  # auto-log on creation

    def __str__(self):
        return self.message
