"""The A30 job's reader: tools/remote_frontier.py with --platform a30 (the default).
Kept under this name because remote/a30_frontier/ ships and calls it."""
import runpy
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('remote_frontier.py')), run_name='__main__')
