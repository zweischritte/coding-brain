"""
Root conftest.py for pytest.

Sets up the Python path so imports work correctly.
The container mounts the api directory to /usr/src/openmemory.
"""
import sys
from pathlib import Path

# Get the working directory (where tests run from)
current_dir = Path(__file__).parent  # /usr/src/openmemory

# Add the working directory to sys.path so "from app.X import Y" works
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
