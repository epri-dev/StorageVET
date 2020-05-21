"""
GUI.py

This Python script provides the basic User Interface set up for SVETapp.py to inherit from
"""

__author__ = 'Thien Nyguen, Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2019. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', "Thien Nguyen"]
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'



from PySide2.QtCore import *
from PySide2.QtWidgets import *
from PySide2.QtGui import *
import logging, inspect, os.path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys, os

gLogger = logging.getLogger('Gui')


# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
    images_rel_path = application_path + "\SVETapp\Images"
elif __file__:
    application_path = os.path.dirname(__file__)
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    storagevet__rel_path = os.path.dirname(os.path.abspath(filename))
    images_rel_path = storagevet__rel_path + "\Images"


class Ui_MainWindow(object):
    """
        Constructor for the QMainWindow GUI, creating and customizing the basic features of the GUI
        setupUi will eventually include setupButtons, setupTools, setupMenu, setupDWidget, and setupStatus
        QMainWindow has 5 main components for its layout:
        Menu Bar
        Toolbars
        Dock Widgets
        Status Bar
        Central Widget
    """

    def setupUi(self, MainWindow):
        self.setWindowTitle("StorageVET 2.0 Beta")
        self.setGeometry(50, 50, 700, 400)
        self.setWindowIcon(QIcon(images_rel_path + "/favicon.ico"))

        self.label = QPlainTextEdit("                                            Welcome to StorageVET \n \n" +
            "        The Electric Power Research Institute's energy storage system \n" +
            "        analysis, dispatch, modelling, optimization, and valuation tool. \n" +
            "        Should be used with Python 3.6, Pandas 0.24, and CVXPY 1.0.21 \n" +
            "        Copyright 2019. Electric Power Research Institute (EPRI). \n        All Rights Reserved.")
        self.text = QTextEdit()
        self.inputFileName = ""
        self.init = ""

        self.stackedWidgets = QStackedWidget()
        figure = Figure()
        self.canvas = FigureCanvas(figure)
        self.threadpool = QThreadPool()
        self.stackedWidgets.addWidget(self.label)
        self.stackedWidgets.addWidget(self.text)
        # self.text.setAlignment(Qt.AlignCenter)
        self.scrollArea = QScrollArea(widgetResizable=True)
        self.scrollArea.setWidget(self.stackedWidgets)
        self.setCentralWidget(self.scrollArea)

        self.setupmenu()
        self.setupstatusbar()
        self.setuptools()
        # self.setupbuttons()

        self.checkBox = QCheckBox('Enlarge', self)
        self.checkBox.move(150, 200)
        self.checkBox.stateChanged.connect(self.enlargeWindow)
        self.dockWidget = QDockWidget("Window")
        self.dockWidget.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.dockWidget.setWidget(self.checkBox)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dockWidget)

        self.msgBox = QMessageBox()
        self.msgBox.setWindowTitle('EPRI Message')
        self.msgBox.setWindowIcon(QIcon(images_rel_path + "/favicon.ico"))

    # status bar is set up, but the stat progress is not fully set up due to missing QFutureWatcher library in PySide2
    def setupstatusbar(self):
        self.statusBar = QStatusBar(self)
        self.statusBar.showMessage("Ready for input")
        self.statLabel = QLabel(self)
        self.statProgress = QProgressBar(self)
        self.statusBar.addPermanentWidget(self.statLabel)
        self.statusBar.addPermanentWidget(self.statProgress, 1)
        self.statusBar.setGeometry(100, 30, 500, 20)
        self.statLabel.setText('Ready')
        self.statProgress.setTextVisible(False)
        self.statProgress.setRange(0, 100)

    # def setupbuttons(self):
    #     """
    #         Set up the buttons needed for the main GUI Window
    #     """
    #
    #     btn = QPushButton("Run", self)
    #     btn.resize(btn.minimumSizeHint())
    #     btn.move(15, 200)
    #     btn.clicked.connect(self.run)
    #     # self.showButton.setText("Run")
    #     # self.connect(self.showButton, SIGNAL("clicked()"), self.run)

    def setuptools(self):
        """
            Set up all the features on the Tool Bar for the main GUI window
        """

        extractAction = QAction(QIcon(images_rel_path + "/quit.png"), 'Sudden Quit', self)
        extractAction.triggered.connect(self.close)
        extractAction2 = QAction(QIcon(images_rel_path + "/run.png"), 'Quick Run', self)
        extractAction2.triggered.connect(self.run)
        self.toolbar = self.addToolBar("Sudden Quit")
        self.toolbar.addAction(extractAction)
        self.toolbar = self.addToolBar("Quick Run")
        self.toolbar.addAction(extractAction2)

    def setupmenu(self):
        """
            Set up all the subMenus and functions on the Menu Bar
        """

        menubar = self.menuBar()
        filemenu = menubar.addMenu("File")
        initmenu = menubar.addMenu("Initialization")
        premenu = menubar.addMenu("Pre-visualization")
        runmenu = menubar.addMenu("Run")

        open = QAction("Upload File", self)
        open.setShortcut("Ctrl+U")
        open.setStatusTip('Upload the Input File')
        open.triggered.connect(self.openfile)
        filemenu.addAction(open)
        reset = QAction("Reset", self)
        reset.setShortcut("Ctrl+E")
        reset.setStatusTip('Reset the Input File')
        reset.triggered.connect(self.reset)
        filemenu.addAction(reset)
        quit = QAction("Quit", self)
        quit.setShortcut("Ctrl+Q")
        quit.setStatusTip('Leave the App')
        quit.triggered.connect(self.close)
        filemenu.addAction(quit)

        init = QAction("Initialize Inputs", self)
        init.triggered.connect(self.initialize)
        initmenu.addAction(init)
        validate = QAction("Validate Inputs", self)
        validate.triggered.connect(self.validate)
        initmenu.addAction(validate)

        battery_summary = QAction("Battery Cycle Life Summary", self)
        battery_summary.triggered.connect(self.battery_summary)
        premenu.addAction(battery_summary)
        input_summary = QAction("Input Summary", self)
        input_summary.triggered.connect(self.summary)
        premenu.addAction(input_summary)
        monthly_summary = QAction("Monthly Data Summary", self)
        monthly_summary.triggered.connect(self.monthly_data_summary)
        premenu.addAction(monthly_summary)
        sens_summary = QAction("Sensitivity Summary", self)
        sens_summary.triggered.connect(self.sens_summary)
        premenu.addAction(sens_summary)
        tariff_summary = QAction("Tariff Summary", self)
        tariff_summary.triggered.connect(self.tariff_summary)
        premenu.addAction(tariff_summary)
        series_summary = QAction("Time Series Summary", self)
        series_summary.triggered.connect(self.series_summary)
        premenu.addAction(series_summary)

        run = QAction("Run StorageVET", self)
        run.setShortcut("Ctrl+R")
        run.setStatusTip('Run the App')
        run.triggered.connect(self.run)
        runmenu.addAction(run)




