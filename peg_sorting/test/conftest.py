import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('PEG_SORTING_CONFIG_DIR', os.path.join(os.path.dirname(__file__), '..', 'config'))
