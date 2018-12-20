from PyQt5 import QtCore, QtWidgets, Qt
from PyQt5.QtWidgets import *
from ctypes import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import *
import sys
sys.path.append("..")
from identify import identify
from scipy import misc
import os

class Example(QThread):
    signal = pyqtSignal(str)
    def __init__(self,image_dir):
        super().__init__()
        self.image_dir = image_dir


    def __del__(self):
        self.wait()

    def run(self):
        global result_name
        # 进行任务操作
        if os.path.exists(self.image_dir):
            pic = misc.imread(self.image_dir)
            result = identify(pic)
            self.signal.emit(result)
        else:
            QMessageBox.information(self,
                                            "Error",
                                            "FileNotFoundError",
                                            QMessageBox.Yes | QMessageBox.No)





class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(652, 500)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(23, 13, 601, 401))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.Detectline = QtWidgets.QLineEdit(self.layoutWidget)
        self.Detectline.setObjectName("Detectline")
        self.verticalLayout_2.addWidget(self.Detectline)
        self.DetectButton = QtWidgets.QPushButton(self.layoutWidget)
        self.DetectButton.setObjectName("DetectButton")
        self.verticalLayout_2.addWidget(self.DetectButton)
        self.openButton = QtWidgets.QPushButton(self.layoutWidget)
        self.openButton.setObjectName("openButton")
        self.verticalLayout_2.addWidget(self.openButton)
        self.startButton = QtWidgets.QPushButton(self.layoutWidget)
        self.startButton.setObjectName("startButton")
        self.verticalLayout_2.addWidget(self.startButton)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.closeSystem = QtWidgets.QPushButton(self.layoutWidget)
        self.closeSystem.setObjectName("closeSystem")
        self.verticalLayout_2.addWidget(self.closeSystem)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 1, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.Pic = QtWidgets.QLabel(self.layoutWidget)
        self.Pic.setMinimumSize(QtCore.QSize(409, 0))
        self.Pic.setObjectName("Pic")
        self.verticalLayout.addWidget(self.Pic)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.Resultlabel = QtWidgets.QLabel(self.layoutWidget)
        self.Resultlabel.setObjectName("Resultlabel")
        self.horizontalLayout.addWidget(self.Resultlabel)
        self.result = QtWidgets.QLineEdit(self.layoutWidget)
        self.result.setObjectName("result")
        self.horizontalLayout.addWidget(self.result)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.progressBar = QtWidgets.QProgressBar(self.layoutWidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout.addWidget(self.progressBar)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 652, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.DetectButton.setText(_translate("MainWindow", "检测设备"))
        self.openButton.setText(_translate("MainWindow", "打开图片"))
        self.startButton.setText(_translate("MainWindow", "开始判断"))
        self.closeSystem.setText(_translate("MainWindow", "关闭系统"))
        self.Pic.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:26pt; font-weight:600; color:#ff0000;\">请打开图片</span></p></body></html>"))
        self.Resultlabel.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:16pt; font-weight:600; color:#ff0000;\">判断结果：</span></p></body></html>"))

class Mywindow(Ui_MainWindow,QtWidgets.QMainWindow):

    def __init__(self,MainWindow):
        super(Mywindow, self).__init__()
        super().setupUi(MainWindow)
        self.image_dir = ""
        self.openButton.clicked.connect(self.open_pic)
        self.startButton.clicked.connect(self.predict)
        self.closeSystem.clicked.connect(self.close_system)
        self.DetectButton.clicked.connect(self.Detect)
        self.Detectline.setStyleSheet("color:red")
        self.Detectline.setText("未检测到设备")
        self.Detectline.setReadOnly(True)
        self.result.setReadOnly(True)
        self.startButton.setDisabled(True)
        self.openButton.setDisabled(True)

    def close_system(self):
        sys.exit()

    def open_pic(self, filename = r"Fingerprint/finger.bmp"):

        try:
            TzData = c_char_p(b'')
            ErrMsg = c_char_p(b'')
            TzLength = c_int()
            nResult = dll.FPIGetFeature(0, TzData, byref(TzLength), ErrMsg)
            if nResult == 0:
                print("func1:", nResult)
                ImgData = create_string_buffer(31478)
                ImgLength = c_int()
                r = dll.FPIGetImageData(1, ImgData, byref(ImgLength))
                if r == 0:
                    print("func3:", r)
                    with open(filename, 'wb') as f:
                        f.write(ImgData.raw)

            self.image_dir = filename  #这个将图片的地址给image_dir, image_dir 子线程调用模型时候要用
            pix = QPixmap(self.image_dir)
            pix.scaled(self.Pic.width(),self.Pic.height(),Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.Pic.setPixmap(pix)
            self.startButton.setDisabled(False)
        except:
            pass

    def setText(self, result_name):
        self.progressBar.setMaximum(100)
        self.progressBar.setProperty("value", 100)
        self.result.setText(result_name)

    def predict(self):
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(0)
        self.thread = Example(self.image_dir)
        self.thread.start()  # 启动线程
        self.thread.signal.connect(self.setText)
    def Detect(self):
        func1 = dll.FPIDevDetect(0)
        if func1==0:
            self.Detectline.setStyleSheet("color:green")
            self.Detectline.setText("已检测到设备")
            self.openButton.setDisabled(False)

if __name__ == "__main__":
    dll = WinDLL("libFPDev_zz.dll")
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Mywindow(MainWindow)
    MainWindow.setWindowTitle("指纹防伪系统")
    MainWindow.show()
    sys.exit(app.exec_())

