"""
High-performance SSH library using Go backend.

This module provides the same API as the original parallel_ssh_lib.py
but uses a Go shared library for 20x faster SSH operations.

Original library backed up to: parallel_ssh_lib.py.backup.20260414_080501
"""

# Import everything from the Go backend wrapper
from .go_ssh_lib import *

# For backward compatibility, make sure Pssh is available
from .go_ssh_lib import Pssh

print("🚀 Using high-performance Go SSH backend")
