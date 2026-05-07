"""Graphical user interface for raytracer.

add repo root directory to ``sys.path``
"""
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
# remove duplicate paths
sys.path = list(set(sys.path))
print("path=", sys.path)

