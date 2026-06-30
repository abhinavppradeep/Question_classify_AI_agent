import sys
import os

class TeeLogger:
    def __init__(self, filename="run.log"):
        # Open in "w" mode to overwrite the file on next run
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, data):
        try:
            is_closed = self.file.closed
        except (ValueError, AttributeError):
            is_closed = True
            
        if not is_closed:
            try:
                self.file.write(data)
                self.file.flush()
            except (ValueError, OSError):
                pass
        self.stdout.write(data)
        self.stdout.flush()

    def flush(self):
        try:
            is_closed = self.file.closed
        except (ValueError, AttributeError):
            is_closed = True
            
        if not is_closed:
            try:
                self.file.flush()
            except (ValueError, OSError):
                pass
        self.stdout.flush()
        
    def close(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

def setup_logger(filename="run.log"):
    """
    Redirects stdout and stderr to a log file (truncating on next run)
    while still printing to the console.
    """
    return TeeLogger(filename)
