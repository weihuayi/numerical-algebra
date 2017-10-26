import time
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *


class LuAlgorithm:

    @staticmethod
    def lu_0(A, step):
        n = len(A)
        P, L, U = [np.eye(n), np.zeros((n, n)), np.zeros((n, n))]

        if step == n - 1:
            forward_step = step
        else:
            forward_step = step + 1

        for i in range(forward_step):

            m = np.argmax(np.abs(A[i:, i])) + i

            if A[m, i] == 0:
                raise ValueError("matrix is singular.")
            else:
                if m != i:
                    A[[i, m], :] = A[[m, i], :]
                    P[[i, m], :] = P[[m, i], :]
                    L[[i, m], :] = L[[m, i], :]
                for j in range(i + 1, n):
                    L[j, i] = A[j, i] / A[i, i]

                for j in range(i, n):
                    U[i, j] = A[i, j]

                for j in range(i + 1, n):
                    for k in range(i + 1, n):
                        A[j, k] -= L[j, i] * U[i, k]

        if step == n - 1:
            P = P.T
            L += np.eye(n)
            U[-1, -1] = A[-1, -1]
        return A, P, L, U

    @staticmethod
    def lu_1(A, step):
        n = len(A)
        P = np.eye(n)

        if step == n - 1:
            forward_step = step
        else:
            forward_step = step + 1

        for i in range(forward_step):

            m = np.argmax(np.abs(A[i:, i])) + i

            if A[m, i] == 0:
                raise ValueError("matrix is singular.")
            else:
                if m != i:
                    A[[i, m], :] = A[[m, i], :]
                    P[[i, m], :] = P[[m, i], :]

                for j in range(i + 1, n):
                    A[j, i] = A[j, i] / A[i, i]

                for j in range(i + 1, n):
                    for k in range(i + 1, n):
                        A[j, k] -= A[j, i] * A[i, k]

        if step == n - 1:
            P = P.T
            L = np.tril(A, -1) + np.eye(n)
            U = np.triu(A, 0)
        else:
            L = np.eye(n)
            U = np.zeros((n, n))
        return A, P, L, U

    @staticmethod
    def lu_2(A, step):
        n = len(A)
        P = np.eye(n)

        if step == n - 1:
            forward_step = step
        else:
            forward_step = step + 1

        for i in range(forward_step):

            m = np.argmax(np.abs(A[i:, i])) + i

            if A[m, i] == 0:
                raise ValueError("matrix is singular.")
            else:
                if m != i:
                    A[[i, m], :] = A[[m, i], :]
                    P[[i, m], :] = P[[m, i], :]

                A[(i + 1):, i] = A[(i + 1):, i] / A[i, i]
                A[(i + 1):, (i + 1):] -= A[(i + 1):, i][:, None] * A[i, (i + 1):]

        if step == n - 1:
            P = P.T
            L = np.tril(A, -1) + np.eye(n)
            U = np.triu(A, 0)
        else:
            L = np.eye(n)
            U = np.zeros((n, n))
        return A, P, L, U


class LuAlgorithmPage(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        # layout
        self.layout = QGridLayout()
        self.layout.setSpacing(10)

        self.widget, self.label = [], []
        self.combobox, self.radiobutton, self.pushbutton, self.tablewidget, self.tabwidget = [], [], [], [], []

        self.decimal_digits = '%.3f'

        self.widget_setting()
        self.widget_init()

    def widget_setting(self):

        # label 0 := matrix size
        self.label.append(QLabel('matrix size'))
        self.label[0].setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label[0], 0, 0, 1, 1)

        # combobox 0 := matrix size
        self.combobox.append(QComboBox())
        self.combobox[0].addItems(list(map(str, range(3, 10))))
        self.combobox[0].setCurrentIndex(1)
        self.layout.addWidget(self.combobox[0], 0, 1, 1, 1)

        # label 1 := step
        self.label.append(QLabel('step'))
        self.label[1].setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label[1], 0, 2, 1, 1)

        # combobox 1 := step
        self.combobox.append(QComboBox())
        self.combobox[1].addItems(list(map(str, range(int(self.combobox[0].currentText())))))
        self.combobox[1].setCurrentIndex(int(self.combobox[0].currentText()) - 1)
        self.layout.addWidget(self.combobox[1], 0, 3, 1, 1)

        # radiobutton 0 := matrix create
        self.radiobutton.append(QRadioButton('user-defined'))
        self.radiobutton[0].setChecked(True)
        self.layout.addWidget(self.radiobutton[0], 0, 4, 1, 1)

        # radiobutton 1 := matrix create
        self.radiobutton.append(QRadioButton('random matrix'))
        self.radiobutton[1].setChecked(False)
        self.layout.addWidget(self.radiobutton[1], 0, 5, 1, 1)

        # pushbutton 0 := algorithm info
        self.pushbutton.append(QPushButton('algorithm info'))
        self.layout.addWidget(self.pushbutton[0], 0, 6, 1, 1)

        # tablewidget 0 := analysis of lu algorithm
        self.tablewidget.append(QTableWidget(1, 1))
        self.tablewidget[0].setColumnCount(4)
        self.tablewidget[0].setHorizontalHeaderLabels(['', '2.2', '2.3', '2.4'])
        self.tablewidget[0].setRowCount(5)
        self.tablewidget[0].setItem(0, 0, QTableWidgetItem('time'))
        self.tablewidget[0].resizeColumnsToContents()
        self.layout.addWidget(self.tablewidget[0], 1, 0, 4, 2)

        class WidgetInner(QTabWidget):
            def __init__(self, dimation):
                QTabWidget.__init__(self)

                self.layout, self.widget, self.tablewidget = [], [], []

                # layout 0
                self.layout.append(QGridLayout())
                self.layout[0].setSpacing(10)
                # tablewidget 0 := origin matrix A_0 in lu_0
                self.tablewidget.append(QTableWidget(dimation, dimation))
                self.layout[0].addWidget(self.tablewidget[0], 0, 0, 1, 1)
                # tablewidget 1 := origin matrix P in lu_0
                self.tablewidget.append(QTableWidget(dimation, dimation))
                self.layout[0].addWidget(self.tablewidget[1], 0, 1, 1, 1)
                # tablewidget 2 := origin matrix L in lu_0
                self.tablewidget.append(QTableWidget(dimation, dimation))
                self.layout[0].addWidget(self.tablewidget[2], 1, 0, 1, 1)
                # tablewidget 3 := origin matrix U in lu_0
                self.tablewidget.append(QTableWidget(dimation, dimation))
                self.layout[0].addWidget(self.tablewidget[3], 1, 1, 1, 1)
                # widget 0
                self.widget.append(QWidget())
                self.widget[0].setLayout(self.layout[0])

                # layout 1
                self.layout.append(QGridLayout())
                self.layout[1].setSpacing(10)
                # tablewidget 5 := origin matrix A_0 in lu_1
                self.tablewidget.append(QTableWidget(dimation, dimation))
                self.layout[1].addWidget(self.tablewidget[4], 0, 0, 1, 1)
                # tablewidget 6 := origin matrix P in lu_1
                self.tablewidget.append(QTableWidget(dimation, dimation))
                self.layout[1].addWidget(self.tablewidget[5], 0, 1, 1, 1)
                # tablewidget 7 := origin matrix L in lu_1
                self.tablewidget.append(QTableWidget(dimation, dimation))
                self.layout[1].addWidget(self.tablewidget[6], 1, 0, 1, 1)
                # tablewidget 8 := origin matrix U in lu_1
                self.tablewidget.append(QTableWidget(dimation, dimation))
                self.layout[1].addWidget(self.tablewidget[7], 1, 1, 1, 1)
                # widget 1
                self.widget.append(QWidget())
                self.widget[1].setLayout(self.layout[1])

                # layout 2
                self.layout.append(QGridLayout())
                self.layout[2].setSpacing(10)
                # tablewidget 9 := origin matrix A_0 in lu_2
                self.tablewidget.append(QTableWidget(dimation, dimation))
                self.layout[2].addWidget(self.tablewidget[8], 0, 0, 1, 1)
                # tablewidget 10 := origin matrix P in lu_2
                self.tablewidget.append(QTableWidget(dimation, dimation))
                self.layout[2].addWidget(self.tablewidget[9], 0, 1, 1, 1)
                # tablewidget 11 := origin matrix L in lu_2
                self.tablewidget.append(QTableWidget(dimation, dimation))
                self.layout[2].addWidget(self.tablewidget[10], 1, 0, 1, 1)
                # tablewidget 12 := origin matrix U in lu_2
                self.tablewidget.append(QTableWidget(dimation, dimation))
                self.layout[2].addWidget(self.tablewidget[11], 1, 1, 1, 1)
                # widget 2
                self.widget.append(QWidget())
                self.widget[2].setLayout(self.layout[2])

                self.addTab(self.widget[0], '&Algorithm_2_2')
                self.addTab(self.widget[1], '&Algorithm_2_3')
                self.addTab(self.widget[2], '&Algorithm_2_4')

        # tabwidget 0
        self.tabwidget.append(WidgetInner(dimation=int(self.combobox[0].currentText())))
        self.layout.addWidget(self.tabwidget[0], 1, 2, 8, 5)

        # tablewidget 1 := analysis of lu algorithm
        self.tablewidget.append(QTableWidget(1, 1))
        self.tablewidget[1].setColumnCount(int(self.combobox[0].currentText()))
        self.tablewidget[1].setRowCount(int(self.combobox[0].currentText()))
        self.tablewidget[1].resizeColumnsToContents()
        self.layout.addWidget(self.tablewidget[1], 5, 0, 4, 2)

        self.setLayout(self.layout)

    def widget_init(self):

        matrix = np.array([[4, 2, 1, 5], [8, 7, 2, 10], [4, 8, 3, 6], [6, 8, 4, 9]], dtype=float)
        self.table_write(dimation=int(self.combobox[0].currentText()), table=self.tablewidget[1], matrix=matrix)
        self.matrix_calulate()

        self.widget_connet()

    def widget_connet(self):
        # change matrix size
        self.combobox[0].currentIndexChanged.connect(self.function_combobox_0)
        # change step
        self.combobox[1].currentIndexChanged.connect(self.function_combobox_1)
        # change origin matrix by user-defined
        self.radiobutton[0].clicked.connect(self.function_radiobutton_0)
        # change origin matrix by random it
        self.radiobutton[1].clicked.connect(self.function_radiobutton_0)
        # # show us the information of LU algorithm TODO: undefined action of pushbutton
        # self.pushbutton[0].clicked.connect(self.function_4)
        # change origin matrix
        self.tablewidget[1].itemChanged.connect(self.matrix_calulate)

    def widget_disconnet(self):
        # change matrix size
        self.combobox[0].currentIndexChanged.disconnect(self.function_combobox_0)
        # change step
        self.combobox[1].currentIndexChanged.disconnect(self.function_combobox_1)
        # change origin matrix by user-defined
        self.radiobutton[0].clicked.disconnect(self.function_radiobutton_0)
        # change origin matrix by random it
        self.radiobutton[1].clicked.disconnect(self.function_radiobutton_0)
        # # show us the information of LU algorithm TODO: undefined action of pushbutton
        # self.pushbutton[0].clicked.disconnect(self.function_4)
        # change origin matrix
        self.tablewidget[1].itemChanged.disconnect(self.matrix_calulate)

    @staticmethod
    def table_read(table):
        n = table.rowCount()
        matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                try:
                    matrix[i, j] = float(table.item(i, j).text())
                except ValueError:
                    matrix[i, j] = 0
        return matrix

    @staticmethod
    def table_write(dimation, table, matrix, decimal_digits=None):
        table.setColumnCount(dimation)
        table.setRowCount(dimation)
        for i in range(dimation):
            for j in range(dimation):
                try:
                    if decimal_digits is None:
                        new_item = QTableWidgetItem(str(matrix[i, j]))
                    else:
                        if abs(matrix[i, j]) < 0.001:
                            new_item = QTableWidgetItem('0')
                        else:
                            new_item = QTableWidgetItem(decimal_digits % matrix[i, j])
                except IndexError:
                    if decimal_digits is None:
                        new_item = QTableWidgetItem(str(1.0 * np.random.randint(0, 10)))
                    else:
                        new_item = QTableWidgetItem(decimal_digits % np.random.randint(0, 10))
                table.setItem(i, j, new_item)
        table.resizeColumnsToContents()

    def matrix_calulate(self):
        """
        calculate A P L U by using Gaussian elimination with partial pivoting.
        """
        dimation = int(self.combobox[0].currentText())
        matrix = self.table_read(self.tablewidget[1])
        step = int(self.combobox[1].currentText())

        start = time.clock()
        A, P, L, U = LuAlgorithm().lu_0(A=matrix.copy(), step=step)
        end = time.clock()
        self.tablewidget[0].setItem(0, 1, QTableWidgetItem('%.2es' % (end-start)))
        self.table_write(dimation, table=self.tabwidget[0].tablewidget[0], matrix=A, decimal_digits=self.decimal_digits)
        self.table_write(dimation, table=self.tabwidget[0].tablewidget[1], matrix=P, decimal_digits='%d')
        self.table_write(dimation, table=self.tabwidget[0].tablewidget[2], matrix=L, decimal_digits=self.decimal_digits)
        self.table_write(dimation, table=self.tabwidget[0].tablewidget[3], matrix=U, decimal_digits=self.decimal_digits)

        start = time.clock()
        A, P, L, U = LuAlgorithm().lu_1(A=matrix.copy(), step=step)
        end = time.clock()
        self.tablewidget[0].setItem(0, 2, QTableWidgetItem('%.2es' % (end - start)))
        self.table_write(dimation, table=self.tabwidget[0].tablewidget[4], matrix=A, decimal_digits=self.decimal_digits)
        self.table_write(dimation, table=self.tabwidget[0].tablewidget[5], matrix=P, decimal_digits='%.0f')
        self.table_write(dimation, table=self.tabwidget[0].tablewidget[6], matrix=L, decimal_digits=self.decimal_digits)
        self.table_write(dimation, table=self.tabwidget[0].tablewidget[7], matrix=U, decimal_digits=self.decimal_digits)

        start = time.clock()
        A, P, L, U = LuAlgorithm().lu_2(A=matrix.copy(), step=step)
        end = time.clock()
        self.tablewidget[0].setItem(0, 3, QTableWidgetItem('%.2es' % (end - start)))
        self.table_write(dimation, table=self.tabwidget[0].tablewidget[8], matrix=A, decimal_digits=self.decimal_digits)
        self.table_write(dimation, table=self.tabwidget[0].tablewidget[9], matrix=P, decimal_digits='%.0f')
        self.table_write(dimation, table=self.tabwidget[0].tablewidget[10], matrix=L, decimal_digits=self.decimal_digits)
        self.table_write(dimation, table=self.tabwidget[0].tablewidget[11], matrix=U, decimal_digits=self.decimal_digits)
        self.tablewidget[0].resizeColumnsToContents()

    def function_combobox_0(self):
        self.widget_disconnet()

        # change combobox 1
        dimation = int(self.combobox[0].currentText())
        self.combobox[1].clear()
        self.combobox[1].addItems(list(map(str, range(dimation))))

        self.function_combobox_1()

        self.widget_connet()

    def function_combobox_1(self):
        # resize all the tables
        matrix = self.table_read(self.tablewidget[1])
        self.table_write(int(self.combobox[0].currentText()), table=self.tablewidget[1], matrix=matrix)
        for i in range(12):
            self.table_write(int(self.combobox[0].currentText()), table=self.tabwidget[0].tablewidget[i], matrix=matrix)

        # change table 1~12
        self.matrix_calulate()

    def function_radiobutton_0(self):
        self.widget_disconnet()

        if self.radiobutton[1].isChecked():

            n = self.tablewidget[1].columnCount()
            for i in range(n):
                for j in range(n):
                    new_item = QTableWidgetItem(str(1.0 * np.random.randint(0, 10)))
                    self.tablewidget[1].setItem(i, j, new_item)
            self.tablewidget[1].resizeColumnsToContents()

        # change table 0~11 inside the tabwidget.
        self.matrix_calulate()

        self.widget_connet()
