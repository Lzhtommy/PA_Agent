"""Run Qt widget tests headless so they never pop windows on the desktop."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
