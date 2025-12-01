from typing import List, Optional

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QComboBox, QLineEdit)

class ConfigOptionsEntry(QWidget):
    def __init__(self, parent, options_str: List[Optional[str]], placeholder_str: List[Optional[str]],
                 current_values: List[Optional[float]]) -> None:
        super().__init__(parent)

        assert len(options_str) == len(placeholder_str) == len(current_values)

        self.options_str = options_str
        self.placeholder_str = placeholder_str

        self.select_specify = QComboBox(self)
        self.select_specify.setFixedWidth(150)
        self.select_specify.addItems(self.options_str)
        self.select_specify.currentIndexChanged.connect(self.update_placeholder_text)

        self.line_edit = QLineEdit(self)
        _current_index = self.select_specify.currentIndex()
        self.line_edit.setPlaceholderText(self.placeholder_str[_current_index]
                                          +" ... "+"%s"%(current_values[_current_index])+" (current value)")

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.select_specify)
        self.layout.addWidget(self.line_edit)

        self.setLayout(self.layout)

    # Even handlers. This is the "slot" for the currentIndexChanged(index) "signal" of the QComboBox widget.
    # The "slot" has as many arguments as the "signal".
    def update_placeholder_text(self, index):
        self.line_edit.setPlaceholderText(self.placeholder_str[index])
