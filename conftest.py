"""
Root-level conftest.py — automatically adds src/ to Python path.
This runs before any test, so all test files can import from src/
without needing sys.path.insert() in every file.
"""

import os
import sys

# Add src/ to path so tests can import project modules cleanly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
