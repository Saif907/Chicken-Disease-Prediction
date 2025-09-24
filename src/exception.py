import sys
from src.logger import logger

def error_message_detail(error, exc_info=None):
    """
    Return detailed error message with file name and line number.
    
    Args:
        error: Original exception
        exc_info: Tuple from sys.exc_info() (optional)
    """
    if exc_info is None:
        exc_info = sys.exc_info()
    _, _, exc_tb = exc_info
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown"
    line_number = exc_tb.tb_lineno if exc_tb else "?"
    return f"[{file_name}:{line_number}] {str(error)}"

class CustomException(Exception):
    """Custom exception class with optional exc_info argument."""

    def __init__(self, error, exc_info=None):
        """
        Args:
            error: Original exception
            exc_info: Tuple from sys.exc_info() (optional)
        """
        super().__init__(str(error))
        if exc_info is None:
            exc_info = sys.exc_info()
        self.message = error_message_detail(error, exc_info)
        try:
            logger.error(self.message)  # Auto-log the error
        except Exception:
            pass

    def __str__(self):
        return self.message
