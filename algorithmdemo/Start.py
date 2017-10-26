'''Created on Jan 16, 2017 @author: zhengli'''

import MainWindow
import sys
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow.WolfMan()
    sys.exit(app.exec_())
