import sys

from PyQt6.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QItemDelegate, QStyledItemDelegate
from PyQt6.QtCore import Qt

class FloatDelegate(QItemDelegate):
    def __init__(self, decimals, parent=None):
        QItemDelegate.__init__(self, parent=parent)
        self.nDecimals = decimals

    def paint(self, painter, option, index):
        value = index.model().data(index, Qt.ItemDataRole.EditRole)
        try:
            number = float(value)
            painter.drawText(option.rect, Qt.AlignmentFlag.AlignLeft, "{:.{}f}".format(number, self.nDecimals))
        except :
            QItemDelegate.paint(self, painter, option, index)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = QTableWidget()

    w.setColumnCount(4)
    w.setRowCount(4)

    for i in range(w.rowCount()):
        for j in range(w.columnCount()):
            number = (i+1)/(j+1)
            w.setItem(i, j, QTableWidgetItem(str(number)))

    w.setItemDelegate(FloatDelegate(3,w))
    w.show()
    sys.exit(app.exec())