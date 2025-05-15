from PyQt5.QtWidgets import QMainWindow as MainWindow

from GarbageRecognition_UI import Ui_MainWindow


class QMainWindow(MainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs) -> None:
        super().__init__()
