import os
import sys

# Tests import the plugin's helper package directly; WanGP itself is not needed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
