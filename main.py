import sys

from PyQt6.QtWidgets import QApplication
from gui.gui_lens_editor import LensEditor

app = QApplication(sys.argv)
w = LensEditor()
sys.exit(app.exec())
