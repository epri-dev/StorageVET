"""
SVETapp.py

This Python script serves as the GUI launch point executing the Python-based version of StorageVET
(AKA StorageVET 2.0 or SVETpy).
"""

__author__ = 'Thien Nguyen'
__copyright__ = 'Copyright 2019. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani',
               'Micah Botkin-Levy', "Thien Nguyen", 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.0.2'

import os, sys, time, logging, matplotlib
import inspect, os.path
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from datetime import datetime
from pathlib import Path
from Params import Params
from run_StorageVET import StorageVET
from GUI import Ui_MainWindow
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

# PySide2 supports Python 3.5 and above
# PySide2 (QT for Python) can be used for open-source commercial software, while PyQT is more restricted by GNU license

config_name = 'SVETapp.cfg'

# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
    config_path = os.path.join(application_path, config_name)
    schema_rel_path = application_path + "\SVETapp\Schema.xml"
    images_rel_path = application_path + "\SVETapp\Images"
    results_rel_path = application_path + "\SVETapp\Results"
elif __file__:
    application_path = os.path.dirname(__file__)
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    storagevet__rel_path = os.path.dirname(os.path.abspath(filename))
    schema_rel_path = storagevet__rel_path + "\Schema.xml"
    images_rel_path = storagevet__rel_path + "\Images"
    results_rel_path = storagevet__rel_path.replace("\dervet\storagevet", "") + "\Results"


gui_path = '.\logs'
try:
    os.mkdir(gui_path)
except OSError:
    print("Creation of the gui_log directory %s failed. Possibly already created." % gui_path)
else:
    print("Successfully created the directory %s " % gui_path)

LOG_FILENAME = gui_path + '\\gui_log_' + datetime.now().strftime('%H_%M_%S_%m_%d_%Y.log')
handler = logging.FileHandler(Path(LOG_FILENAME))
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
gLogger = logging.getLogger('Gui')
gLogger.setLevel(logging.DEBUG)
gLogger.addHandler(handler)
gLogger.info('Started logging...')



# TODO: Multi-Threading is slow for test cases that involve Sensitivity Analysis, we will consider Multi-Processes - TN

class Worker(QRunnable):
    """
        Worker thread

        Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

        :param callback: The function callback to run on this worker thread. Supplied args and
                         kwargs will be passed through to the runner.
        :type callback: function
        :param args: Arguments to pass to the callback function
        :param kwargs: Keywords to pass to the callback function
    """

    def __init__(self, params):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.input = params
        self.run_time = 0
        # self.dialog = QProgressDialog()
        # self.dialog.setLabelText("Simulation progress status:")
        # self.dialog.setRange(0, 100)

    def run(self):
        """
            Initialise the runner function with passed args, kwargs.
        """
        start = time.time()
        StorageVET('', '', self.input)
        end = time.time()
        statement = "Simulation thread completed within " + str(end - start) + " sec. \nIt successfully ran."
        print(statement)
        gLogger.info(statement)
        self.run_time = end - start

class SVETapp(QMainWindow, Ui_MainWindow):
    """
        Provide the GUI for StorageVET to run as a Python-based application, instead of via Terminal console.
        QMainWindow is chosen because it has its own window management, and its own layout that includes QToolBars,
        QDockWidgets, a QMenuBar, and a QStatusBar. It has a separate Central Widget which contains the main application
        content and can have any kind of Widget.

        Args:
            QMainWindow (QWidget object): main application window for StorageVET
            Ui_MainWindow (Ui object): the window Ui already set up and inherited from GUI class
    """

    gLogger.info('Started configuring SVETapp...')

    def __init__(self, parent=None):
        super(SVETapp, self).__init__(parent)
        self.setupUi(self)
        self.inputFileName = ""
        self.init = False
        self.valid = False

        # just in case we run SVETapp from a Python standalone environment, make sure to locate its new directory
        self.current_dir = QDir.currentPath()
        if "storagevet_pkg" not in self.current_dir:
            self.current_dir = self.current_dir + "/Lib/site-packages/storagevet_pkg"

        gLogger.info("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        gLogger.info('Finished configuration for SVETapp GUI window...')

    def enlargeWindow(self, state):
        """
            Optional feature to enlarge the main GUI window
            Part of the Dock Widgets
        """

        if state == Qt.Checked:
            self.setGeometry(50, 50, 1000, 600)
        else:
            self.setGeometry(50, 50, 500, 300)

    def openfile(self):
        """
            Function to open the folder directory to look and select the input files
            Part of the MenuBar
        """

        self.statProgress.reset()
        self.stackedWidgets.setCurrentIndex(0)
        try:
            self.inputFileName, _ = QFileDialog.getOpenFileName(self, "Upload File", self.current_dir,
                                "CSV files (*.csv);;XML files (*.xml)")
            if self.inputFileName != "":
                self.msgBox.setText('The provided file is accepted')
                self.msgBox.show()
                gLogger.info('User configured data with the provided file: %s', self.inputFileName)
        except ValueError:
            self.msgBox.setText('Invalid file was provided')
            self.msgBox.exec_()
        if self.inputFileName == "":
            self.msgBox.setText('User did not provide any file.')
            self.msgBox.exec_()
            gLogger.info('User did not provide any file, or invalid file was provided.')

    def initialize(self):
        """
            Function to call the initialize function on the Params object of the StorageVET application
            Part of the MenuBar
        """

        self.statProgress.reset()
        self.stackedWidgets.setCurrentIndex(0)
        if self.inputFileName:
            if self.inputFileName.endswith(".csv"):
                self.inputFileName = Params.csv_to_xml(self.inputFileName)

            # Initialize the Input Object from Model Parameters and Simulation Cases
            Params.initialize(self.inputFileName, schema_rel_path)
            self.init = True
            self.msgBox.setText('Successfully initialized the Params class with the XML file.')
            self.msgBox.exec_()
            gLogger.info('User successfully initialized the Params class with the XML file.')
        else:
            # Look for proper ways so that the Params initialize errors from StorageVET side can be reported to
            # SVETapp main window text; currently, SVETapp window just exited without any messages...
            self.msgBox.setText('Params has not been initialized to validate.')
            self.msgBox.exec_()
            gLogger.info('User has not given an input file to initialize.')

    def validate(self):
        """
            Function to call the validate function on the Params object after Initialization
            Part of the MenuBar
        """

        self.stackedWidgets.setCurrentIndex(0)
        if self.init:
            Params.validate()
            self.msgBox.setText('Successfully validated the Params class with the Schema file.')
            self.msgBox.exec_()
            gLogger.info('User successfully validated the Params class with the Schema file.')
            self.valid = True
        else:
            # Look for proper ways so that the Params validation errors from StorageVET side can be reported to
            # SVETapp main window text; currently, SVETapp window just exited without any messages...
            self.msgBox.setText('Input has not been initialized to validate.')
            self.msgBox.exec_()
            gLogger.info('User has not initialized the input to validate.')

    def summary(self):
        """
            Function to call the class_summary function on the Params object
            Part of the MenuBar
        """

        if self.init:
            table = Params.class_summary()
            table.align['Sensitivity'] = "l"
            table.align['Value'] = "l"
            self.text.setText(table.get_html_string())
            self.stackedWidgets.setCurrentIndex(1)
            gLogger.info('User successfully finished class summary.')
        else:
            self.msgBox.setText('Input has not been initialized to summarize.')
            self.msgBox.exec_()
            gLogger.info('User has not initialized the input to summarize.')

    def series_summary(self):
        """
            Function to call the series_summary function on the Params object
            Part of the MenuBar
        """

        if self.init:
            no_series = len(Params.referenced_data["time_series"])
            if no_series == 1:
                if Params.series_summary():
                    self.canvas = FigureCanvas(Params.series_summary())
                    self.canvas.draw()
                    self.stackedWidgets.addWidget(self.canvas)
                    self.stackedWidgets.setCurrentIndex(2)
                    self.msgBox.setText('Successfully plotted all the provided time series.')
                    self.msgBox.exec_()
                    gLogger.info('User successfully plotted all the provided time series.')
                else:
                    self.msgBox.setText('The plot is not optimized for time series data longer than 1 year and '
                                        'with timestep dt < 1 hour.')
                    self.msgBox.exec_()
                    gLogger.info('The plot is not optimized for time series data longer than 1 year and '
                                        'with timestep dt < 1 hour.')
            elif no_series > 1:
                self.stackedWidgets.setCurrentIndex(0)
                self.msgBox.setText('There is more than 1 time series data to plot. '
                                    'Current GUI version has not supported this yet.')
                self.msgBox.exec_()
            else:
                self.stackedWidgets.setCurrentIndex(0)
                self.msgBox.setText('There is 0 time series data to plot or '
                                    'GUI cannot read the input name for time series data.')
                self.msgBox.exec_()
        else:
            self.msgBox.setText('Input has not been initialized to plot time series.')
            self.msgBox.exec_()
            gLogger.info('User has not initialized the input to plot time series..')

    def battery_summary(self):
        """
            Function to call the battery_cycle_life_summary function on the Params object
            Part of the MenuBar
        """

        if self.init:
            table = Params.battery_cycle_life_summary()
            table.align['Cycle Depth Upper Limit'] = "r"
            table.align['Cycle Life Value'] = "r"
            self.text.setText(table.get_html_string())
            self.stackedWidgets.setCurrentIndex(1)
            self.msgBox.setText('Successfully printed the battery cycle life table.')
            self.msgBox.exec_()
            gLogger.info('User successfully printed the battery cycle life table.')
        else:
            self.msgBox.setText('Input has not been initialized to summarize the cycle life table.')
            self.msgBox.exec_()
            gLogger.info('User has not initialized the input to summarize the cycle life table.')

    def monthly_data_summary(self):
        """
            Function to call the monthly_data_summary function on the Params object
            Part of the MenuBar
        """

        if self.init:
            table = Params.monthly_data_summary()
            self.text.setText(table.get_html_string())
            self.stackedWidgets.setCurrentIndex(1)
            self.msgBox.setText('Successfully printed the monthly data table.')
            self.msgBox.exec_()
            gLogger.info('User successfully printed the monthly data table.')
        else:
            self.msgBox.setText('Input has not been initialized to summarize the monthly data table.')
            self.msgBox.exec_()
            gLogger.info('User has not initialized the input to summarize the monthly data table.')

    def tariff_summary(self):
        """
            Function to call the table_summary function on the Params object
            Part of the MenuBar
        """

        if self.init:
            table = Params.verify_tariff()
            table.align['Billing Period'] = "r"
            table.align['Start Month'] = "r"
            table.align['End Month'] = "r"
            table.align['Start Time'] = "r"
            table.align['End Time'] = "r"
            table.align['Excluding Start Time'] = "r"
            table.align['Excluding End Time'] = "r"
            table.align['Weekday?'] = "r"
            table.align['Value'] = "r"
            table.align['Charge'] = "r"
            table.align['Name_optional'] = "r"
            self.text.setText(table.get_html_string())
            self.stackedWidgets.setCurrentIndex(1)
            self.msgBox.setText('Successfully printed the user tariff data table.')
            self.msgBox.exec_()
            gLogger.info('User successfully printed the user tariff data table.')
        else:
            self.msgBox.setText('Input has not been initialized to summarize the user tariff data table.')
            self.msgBox.exec_()
            gLogger.info('User has not initialized the input to summarize the user tariff data table.')

    def sens_summary(self):
        """
            Function to call the sens_summary function on the Params object
            Part of the MenuBar
        """

        if self.init:
            table = Params.sens_summary()
            if table == 0:
                self.stackedWidgets.setCurrentIndex(0)
                self.msgBox.setText('There is no sensitivity data.')
                self.msgBox.exec_()
                gLogger.info('File has no sensitivity data to summarize.')
            else:
                self.text.setText(table.get_html_string())
                self.stackedWidgets.setCurrentIndex(1)
                self.msgBox.setText('Successfully printed the sensitivity data.')
                self.msgBox.exec_()
                gLogger.info('User successfully printed the sensitivity data.')
        else:
            self.msgBox.setText('Input has not been initialized to summarize.')
            self.msgBox.exec_()
            gLogger.info('User has not initialized the input to summarize.')

    def reset(self):
        """
            Function to reset the input file and its initialization
            Part of the MenuBar
        """
        self.init = False
        self.inputFileName = ""
        self.statProgress.reset()
        self.msgBox.setText('Successfully reset the data.')
        self.msgBox.exec_()
        gLogger.info('Successfully reset the data.')

    def run(self):
        """
            Function to call the class and run the simulation run_StorageVET
            Part of the MenuBar
        """
        self.msgBox.setText("Simulation started, using this file: \n" + self.inputFileName)
        self.msgBox.exec_()
        if self.init and self.valid:
            gLogger.info("User clicked Run the Application")
            print("Simulation thread started.")
            # run_StorageVET(Params)
            worker = Worker(Params)
            self.threadpool.start(worker)
            if self.threadpool.waitForDone(-1):
                # a hackey estimate that an error in optimization problem usually produces within less than 3 sec - TN
                if worker.run_time > 3:
                    statement = "Simulation thread completed within " + str(worker.run_time) + " sec. \n"
                else:
                    statement = "Simulation thread produced error within " + str(worker.run_time) + " sec. \n"
                gLogger.info(statement)
                self.text.setText("Simulation is done, using this file: \n" + self.inputFileName + "\n\n" + statement)
                self.stackedWidgets.setCurrentIndex(1)
        elif not self.init:
            self.text.setText('Unfortunately, input has not been initialized to run.')
            self.stackedWidgets.setCurrentIndex(1)
            # self.msgBox.setText('Unfortunately, input has not been initialized to run.')
            # self.msgBox.exec_()
            gLogger.info('User has not initialized the input to run the simulation.')
        elif not self.valid:
            self.text.setText('Warning: input has not been validated to run. Simulation might produce inaccurate results.')
            self.stackedWidgets.setCurrentIndex(1)
            gLogger.info('User has not validated the input to run the simulation.')

if __name__ == "__main__":
    """
        the Main section for the GUI Window to start
    """

    app = QApplication(sys.argv)
    mainWindow = SVETapp()

    pixmap = QPixmap(images_rel_path + "/splashscreen.png")
    splash = QSplashScreen(pixmap)
    splash.show()
    acknowledgement = QMessageBox.question(mainWindow, 'Pre-authorization', "You must read and accept the "
                                        "acknowledgement to run this application.",  QMessageBox.Yes | QMessageBox.No)

    # need this for event handlers and threading
    app.processEvents()

    if acknowledgement == QMessageBox.Yes:
        gLogger.info("SVETapp GUI window started...")
        mainWindow.show()
        splash.finish(mainWindow)
        status = app.exec_()
    else:
        mainWindow.close()
        splash.close()
        status = app.exit()

    gLogger.info("SVETapp GUI window exited...")
    sys.exit(status)




