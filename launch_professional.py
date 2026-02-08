#!/usr/bin/env python3
"""
NeuroHand Professional Interface Launcher
==========================================

Launch the professional, startup-grade BCI platform interface.

This is the main entry point for the NeuroHand BCI Platform.
Use this for demos, investor presentations, and production deployment.

Usage:
    python launch_professional.py

    or

    ./launch_professional.py

Features:
    - Professional branding and design
    - Advanced 3D brain visualization
    - Real-time analytics dashboard
    - Comprehensive EEG analysis
    - Startup presentation-ready
    - Memory-safe operation

Author: Temur Turayev
Company: NeuroHand Technologies
Year: 2024
"""

from pathlib import Path

# Project root for file resolution (not added to sys.path)
PROJECT_ROOT = Path(__file__).parent

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║              🧠  NeuroHand Technologies  🧠                    ║
    ║                                                                ║
    ║         Advanced Brain-Computer Interface Platform            ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    from src.interface.professional_app import main
    main()
