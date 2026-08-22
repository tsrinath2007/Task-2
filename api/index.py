import sys
import os

# Include parent directory in sys.path so harness and server can be imported seamlessly
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from server import app
