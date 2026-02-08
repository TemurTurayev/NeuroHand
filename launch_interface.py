#!/usr/bin/env python3
"""
NeuroHand Interface Launcher
============================

Simple script to launch the Gradio web interface for testing
the EEGNet motor imagery BCI model.

Usage:
    python launch_interface.py

    or

    ./launch_interface.py

Author: Temur Turayev
TashPMI, 2024
"""

from pathlib import Path

# Project root for file resolution (not added to sys.path)
PROJECT_ROOT = Path(__file__).parent

if __name__ == "__main__":
    from src.interface.gradio_app import main
    main()
