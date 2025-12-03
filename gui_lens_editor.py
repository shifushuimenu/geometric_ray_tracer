# TODO: Use TableModel to format entries
# https://www.pythonguis.com/tutorials/pyqt6-qtableview-modelviews-numpy-pandas/

# TODO: surface count should start at zero (Object surface), like in Zemax

import os
import sys
import numpy as np # IMPROVE: don't use numpy
from datetime import datetime
from typing import List, Iterable

from interface import DisplayInterface, DisplayInterfaceRayspot
from lens import LensSequence
from trace_ray import RayTracer, MeridionalRayData, NonmeridionalRayData

from config import Config
from gui_config_options import ConfigOptionsEntry

from gui_table_item_delegate import FloatDelegate

os.environ["QT_API"] = "PyQt6"

import matplotlib.pyplot as plt 

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenu,
    QToolBar,    
    QTabWidget,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QFileDialog, 
    QHeaderView,
    QMessageBox,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
)

from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import QSize, Qt

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)

plt.ion()

# Formatting routines 
float_as_str = lambda x: str("%20.4f"%(x))
str_as_float = lambda x: float(x) if x != "---" else 0

CA_COLUMN = 6

class LensEditor(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.setWindowTitle("Lens Editor")
        self.left = 0
        self.top = 0
        self.width = 800
        self.height = 400
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.create_menus()
        self.create_toolbar()

        self.configTabs = LensConfigTabs(self)
        self.setCentralWidget(self.configTabs)

        # Global configuration object: entrance pupil diameter, fields, wavelengths, ...
        self.config = Config(max_obj_height=1.0, entrance_pupil_diameter=2.0)

        self.show()

    def create_menus(self):
        """Create the main menu bar"""
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu("&Lens File")
        open_file_action = QAction(QIcon(), "&Open lens file", self)
        save_file_action = QAction(QIcon(), "&Save config", self)
        open_file_action.triggered.connect(self.open_file_handler)
        save_file_action.triggered.connect(self.save_file_handler)
        fileMenu.addAction(open_file_action)
        fileMenu.addAction(save_file_action)
        settingsMenu = menuBar.addMenu("&Settings")
        set_pupils_action = QAction(QIcon(), "&Pupils", self)
        set_fields_action = QAction(QIcon(), "&Fields", self)
        set_pupils_action.triggered.connect(self.set_pupils_handler)
        set_fields_action.triggered.connect(self.set_fields_handler)
        set_wavelengths_action = QAction(QIcon(), "&Wavelengths", self)        
        settingsMenu.addAction(set_pupils_action)        
        settingsMenu.addAction(set_fields_action)
        settingsMenu.addAction(set_wavelengths_action)
        configsMenu = menuBar.addMenu("&Configurations")
        add_config_action = QAction(QIcon(), "&Add configuration", self)
        add_config_action.triggered.connect(self.add_lens_configuration)
        configsMenu.addAction(add_config_action)
        toolsMenu = menuBar.addMenu("Handy &Tools")
        autofocus_action = QAction(QIcon(), "Paraxial &Autofocus", self)
        defocus_action = QAction(QIcon(), "&Defocus to CLC", self)
        make_symmetric_action = QAction(QIcon(), "&Make symmetric", self)
        toolsMenu.addAction(autofocus_action)
        toolsMenu.addAction(defocus_action)
        toolsMenu.addAction(make_symmetric_action)
        insertMenu = toolsMenu.addMenu("&Insert")
        insert_field_flattener_action = QAction(QIcon(), "&Paraxial field flattener", self)   
        insert_field_lens_action = QAction(QIcon(), "&Field lens", self)
        insertMenu.addAction(insert_field_flattener_action)
        insertMenu.addAction(insert_field_lens_action)
        helpMenu = menuBar.addMenu("&Help")
        open_docs_action = QAction(QIcon(), "Open &Documentation", self)
        helpMenu.addAction(open_docs_action)

        # insert field lens 
        # show pupils and stops
    
    def create_toolbar(self):
        """Create the main toolbar"""
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setStyleSheet("QToolBar { background-color: white; border: 2px solid black; }")
        toolbar.setIconSize(QSize(60, 60))
        raytrace_action = QAction(QIcon("icons/raytrace_icon.png"), "&Raytrace", self)        
        spot_action = QAction(QIcon("icons/ray_spot_diagram_icon.png"), "&Ray Spot Diagram", self)
        rayfan_action = QAction(QIcon("icons/rayfan_icon.png"), "&Ray Fan Plot", self)
        Seidel_coefficients_action = QAction(QIcon(), "&Seidel Diagram", self)
        MTF_action = QAction(QIcon(), "&MTF", self)        
        Gaussian_beam_action = QAction(QIcon(), "&Gaussian Beam", self)        

        self.raytraceWindow = None
        raytrace_action.triggered.connect(self.showRaytraceDiagram)
        self.raySpotDiagramWindow = None
        spot_action.triggered.connect(self.showRaySpotDiagram)
        self.rayFanDiagramWindow = None
        rayfan_action.triggered.connect(self.showRayFanDiagram)
        self.SeidelWindow = None
        Seidel_coefficients_action.triggered.connect(self.showSeidelAberrationCoefficientsDiagram)
        self.MTFWindow = None
        MTF_action.triggered.connect(self.showModulationTransferFunctionDiagram)
        self.GaussianBeamWindow = None
        Gaussian_beam_action.triggered.connect(self.showGaussianBeamDiagram)

        toolbar.addAction(raytrace_action)
        toolbar.addAction(spot_action)
        toolbar.addAction(rayfan_action)
        toolbar.addAction(Seidel_coefficients_action)
        toolbar.addAction(MTF_action)
        toolbar.addAction(Gaussian_beam_action)

        self.addToolBar(toolbar)


    # ================
    # Event handlers
    # ================
    def open_file_handler(self):
        filename, _ = QFileDialog.getOpenFileName(self, filter="*.txt")
        if filename:
            print("opening file %s"%(filename))
            self.configTabs.read_from_file(filename)

    def save_file_handler(self):
        filename, _ = QFileDialog.getSaveFileName(self, filter="*.txt")
        if filename:
            print("Saving to file %s"%(filename))
            try:
                self.configTabs.save_to_file(filename)
            except AttributeError as e:
                msgBox = QMessageBox()
                msgBox.setText("Cannot save lens file with invalid (empty) entries.")
                msgBox.exec()
                # raise Warning("Cannot save lens file with invalid (empty) entries.")

    # =========================================
    # Event handlers for menubar
    # =========================================
    def add_lens_configuration(self):
        self.configTabs.addNewLensConfig()

    def set_pupils_handler(self):
        self.popup_window = QWidget()
        self.popup_window.setWindowTitle("Pupils and Fields")
        self.popup_window.setFixedSize(600,200)

        self.layout = QVBoxLayout()

        self.selectSpecifyPupil = ConfigOptionsEntry(self.popup_window, options_str=["EPD", "XPD", "Image Space NA", "Object Space NA"],
            placeholder_str=["Enter entrance pupil diameter", "Enter exit pupil diameter", "Enter image space numerical aperture", 
             "Enter object space numerical aperture"],
            current_values=[self.config.EPD, None, None, None])
        self.layout.addWidget(self.selectSpecifyPupil)
        self.selectSpecifyField = ConfigOptionsEntry(self.popup_window, options_str=["Field height", "Field angle"], 
            placeholder_str=["Enter max field height", "Enter max field angle"],
            current_values=[self.config.max_obj_height, None])
        self.layout.addWidget(self.selectSpecifyField)

        self.savePupilFieldButton = QPushButton("Save", self.popup_window)
        self.savePupilFieldButton.setFixedSize(100, 60)
        self.savePupilFieldButton.clicked.connect(self.update_pupil_and_field_height)
        self.layout.addWidget(self.savePupilFieldButton)
        self.popup_window.setLayout(self.layout)
        self.popup_window.show()

    def update_pupil_and_field_height(self):
        # IMPROVE: Handle invalid input.
        if self.selectSpecifyPupil.select_specify.currentIndex() != 0:
            raise NotImplementedError
        if self.selectSpecifyField.select_specify.currentIndex() != 0:
            raise NotImplementedError
        _text = self.selectSpecifyPupil.line_edit.text()
        if _text not in ['']:
            self.config.EPD = float(_text)
        _text = self.selectSpecifyField.line_edit.text()
        if _text not in ['']:
            self.config.max_obj_height = float(_text)
        print("EPD=", self.config.EPD)
        print("max obj height=", self.config.max_obj_height)
        self.popup_window.close()

    def set_fields_handler(self):
        pass

    # =================================
    # Event handlers for toolbar 
    # =================================    
    def showRaytraceDiagram(self):

        print("self.config.max_obj_height=", self.config.max_obj_height)

        if self.raytraceWindow is None:
            self.current_tab = self.configTabs.tabs_widget.currentWidget()
            self.lens_sequence = self.current_tab.get_lens_sequence_from_table(self.current_tab.table)        
                        
            self.ray_tracer = RayTracer(self.lens_sequence)
            self.ray_data = self.ray_tracer.calculate_meridional_ray_data(self.lens_sequence, self.config)

            # Using the ray data, update the clear apertures in the lens editor.
            self.current_tab.update_clear_apertures(self.current_tab.table, self.ray_data.clear_apertures)            

            self.display_interface = DisplayInterface(self.lens_sequence, self.config, self.ray_data)            
            self.raytraceWindow = RaytraceDiagram(self.display_interface, self.ray_data)

            self.raytraceWindow.show()
        else:
            self.raytraceWindow.close()
            self.raytraceWindow = None

    def showRaySpotDiagram(self):
        if self.raySpotDiagramWindow is None:
            self.current_tab = self.configTabs.tabs_widget.currentWidget()
            self.lens_sequence = self.current_tab.get_lens_sequence_from_table(self.current_tab.table)        

            if not hasattr(self, "ray_tracer"):
                self.ray_tracer = RayTracer(self.lens_sequence)
                self.ray_data = self.ray_tracer.calculate_meridional_ray_data(self.lens_sequence, self.config)

            self.nonmeridional_ray_data = self.ray_tracer.calculate_nonmeridional_ray_data(self.lens_sequence, self.config)

            self.display_interface = DisplayInterfaceRayspot(self.lens_sequence, self.config, self.nonmeridional_ray_data)

            self.raySpotDiagramWindow = RaySpotDiagram(self.display_interface, self.nonmeridional_ray_data)
            self.raySpotDiagramWindow.show()
        else:
            self.raySpotDiagramWindow.close()
            self.raySpotDiagramWindow = None

    def showRayFanDiagram(self):
        if self.rayFanDiagramWindow is None:
            self.rayFanDiagramWindow = RayFanDiagram()
            self.rayFanDiagramWindow.show()
        else:
            self.rayFanDiagramWindow.close()
            self.rayFanDiagramWindow = None

    def showModulationTransferFunctionDiagram(self):
        if self.MTFWindow is None:
            self.MTFWindow = ModulationTransferFunctionDiagram()
            self.MTFWindow.show()
        else:
            self.MTFWindow.close()
            self.MTFWindow = None

    def showSeidelAberrationCoefficientsDiagram(self):
        if self.SeidelWindow is None:
            self.SeidelWindow = SeidelAberrationCoefficientsDiagram()
            self.SeidelWindow.show()
        else:
            self.SeidelWindow.close()
            self.SeidelWindow = None

    def showGaussianBeamDiagram(self):
        if self.GaussianBeamWindow is None:
            self.GaussianBeamWindow = GaussianBeamDiagram()
            self.GaussianBeamWindow.show()
        else:
            self.GaussianBeamWindow.close()
            self.GaussianBeamWindow = None



class LensConfigTabs(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)

        self.parent = parent

        # Initialize tabs
        self.tabs_widget = QTabWidget()
        self.tabs_widget.setTabsClosable(True)
        self.tabs = []        
        self.tabs_widget.resize(800, 400)

        tab1 = LensEditorConfig(self.parent)
        self.tabs.append(tab1)        
        self.tabs_widget.addTab(tab1, "Config 1")
        self.counter = len(self.tabs)

        # Make tabs closable
        self.tabs_widget.tabCloseRequested.connect(self.removeLensConfig)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.tabs_widget)
        self.setLayout(self.layout)

    def read_from_file(self, filename):
        # clear all tabs
        self.tabs = []
        self.tabs_widget.clear()
        DATA = np.loadtxt(filename)
        # with open(filename, "r") as fh:
        #     for line in fh:
        #         if not line.startswith("#"):
        #             line = line.split("#", 1)[0] # remove comments at the end of a line
        #             try:
        #                 surf, stop_flag, radius, thickness, n_d, V_d, clear_diameter = line.split()
        #                 print(surf, stop_flag, radius, thickness, n_d, V_d, clear_diameter)
        #             except ValueError as e:
        #                 print("non-conforming row detected -> just skip it")

        tab1 = LensEditorConfig(self.parent)
        self.tabs.append(tab1)
        self.tabs_widget.addTab(tab1, "Config 1")
        tab1.read_config(DATA)

    def save_to_file(self, filename):
        "save current tab to file"
        with open(filename, "w") as fh:
            current_tab = self.tabs_widget.currentWidget()
            fh.write("# Created on: " + str(datetime.today())[0:19] + "\n")
            header_str = "# "+" | ".join([str(current_tab.table.horizontalHeaderItem(i).data(0)) for i in range(current_tab.table.columnCount())])+"\n"
            fh.write(str("#"+"="*(len(header_str)-1))+"\n")
            fh.write(header_str)
            fh.write(str("#"+"="*(len(header_str)-1))+"\n")
            for row in range(current_tab.table.rowCount()):
                if current_tab.table.item(row, 0).data(0) in ['STOP']:
                    stop_flag = 1
                else:
                    stop_flag = 0
                # IMPROVE with complete format string 
                entries = [str_as_float(current_tab.table.item(row, i).data(0)) for i in range(2, current_tab.table.columnCount())]
                print("entries=", entries)
                fh.write(str("%5d %10d       " + "%18.8e     "*(current_tab.table.columnCount()-2) + "\n")
                         %((row, stop_flag)+tuple(entries)))


    def addNewLensConfig(self):
        print("adding new config")
        nc = self.tabs_widget.count()
        assert nc == len(self.tabs)
        self.counter += 1
        current_tab = self.tabs_widget.currentWidget()
        try:
            new_tab = LensEditorConfig(self, current_tab)
            self.tabs.append(new_tab)
            self.tabs_widget.addTab(new_tab, "Config %d"%(self.counter))
        except TypeError as e:
            print(e)

    def removeLensConfig(self, index):
        print("removing configuration with index %d"%(index+1))
        self.tabs.pop(index)
        self.tabs_widget.removeTab(index)


class LensEditorConfigBase(QWidget):
    """
    Base class for lens editor containing possibly several configurations of lens prescriptions.
    Implements basic functionality such as
        - minimum of three rows: OBJ, STOP, IMAG
        - a context menu for adding and removing rows and changing the location of the aperture STOP
    """
    
    def __init__(self, parent):
        super().__init__()

        self.parent = parent

        self.table = QTableWidget(None)
        self.header_labels = ["Surface", "Stop flag", "Radius of curvature (mm)", "Thickness (mm)", "index of refraction n_d", "Abbe number V_d", "clear semidiameter (mm)"]
        self.table.setColumnCount(len(self.header_labels))        
        self.table.setHorizontalHeaderLabels(self.header_labels)
        self.table.setRowCount(3)        
        self.table.setItem(0, 0, QTableWidgetItem('OBJ'))        
        for i, entry in enumerate(["0", np.inf, 10.0, 1.0, np.inf, 0.0]):
            self.table.setItem(0, i+1, QTableWidgetItem(float_as_str(entry) if type(entry)==float else str(entry)))
        self.table.setItem(1, 0, QTableWidgetItem('STOP'))                            
        for i, entry in enumerate(["1", np.inf, 10.0, 1.0, np.inf, 0.0]):
            self.table.setItem(1, i+1, QTableWidgetItem(float_as_str(entry) if type(entry)==float else str(entry)))
        self.table.setItem(2, 0, QTableWidgetItem('IMAG'))            
        for i, entry in enumerate(["0", np.inf, "---", "---", "---", 0.0]):
            self.table.setItem(2, i+1, QTableWidgetItem(float_as_str(entry) if type(entry) == float else str(entry)))

        self.stop_surface = 1

        # context menu for inserting and removing rows
        self.contextMenu = QMenu(self.table)
        insert_row_action = self.contextMenu.addAction("Insert row below")
        remove_row_action = self.contextMenu.addAction("Remove this row")
        change_stop_action = self.contextMenu.addAction("Make current surface the STOP")

        # connect to event handlers
        insert_row_action.triggered.connect(self.insert_row_action_handler)
        remove_row_action.triggered.connect(self.remove_row_action_handler)
        change_stop_action.triggered.connect(self.change_stop_action_handler)
        self.table.cellActivated.connect(self.on_cell_activated)
        self.show()
    
        # ==========
        # Layout 
        # ==========
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)        

        # adjust column width to match header entries and allow user to change the column width interactively
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)

        self._set_table_entry_settings()

    def _set_table_entry_settings(self):
        self.table.setItemDelegate(FloatDelegate(3))
        self.table.setItemDelegateForColumn(1, FloatDelegate(0))
        self._make_column_readonly(self.table, 0) # column with surface type
        self._make_column_readonly(self.table, 1) # column with stop flags
        self._make_column_readonly(self.table, 6) # column with clear diameters

        # self.table.itemChanged.connect(self.on_item_changed)


    def _make_column_readonly(self, table: QTableWidget, column: int) -> None:
        for row in range(table.rowCount()):
            item = table.item(row, column)
            if item:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

    def _get_column_data(self, table: QTableWidget, column: int) -> List:
        data = []
        for row in range(table.rowCount()):
            item = table.item(row, column)
            if item is not None:
                if item.text() == '---': 
                    item_ = None
                else:
                    item_ = float(item.text())
            else:
                item_ = None
            data.append(item_)
        return data
    
    def _set_column_data(self, table: QTableWidget, column: int, column_data: Iterable) -> None:
        for row in range(table.rowCount()):
            entry = column_data[row]
            self.table.setItem(row, column, QTableWidgetItem(float_as_str(entry) if type(entry)==float else str(entry)))

    # Read current lens sequence 
    def get_lens_sequence_from_table(self, table: QTableWidget) -> LensSequence:
        stop_flag = np.array(list(map(int, self._get_column_data(table, 1))))
        R = np.array(self._get_column_data(table, 2))
        t = np.array(self._get_column_data(table, 3))
        n = np.array(self._get_column_data(table, 4))
        Vd = np.array(self._get_column_data(table, 5))
        clear_diameter = np.array(self._get_column_data(table, CA_COLUMN))
        num_surfs = len(R)
        AS_surf = self.stop_surface
        print("AS_surf=", AS_surf)
        print("stop_flag=", stop_flag)

        return LensSequence(num_surfs, AS_surf, stop_flag, True, "mm", R, t, n, Vd, 
                     clear_diameter, True)
    
    def update_clear_apertures(self, table: QTableWidget, clear_apertures: Iterable) -> None:
        self._set_column_data(table, CA_COLUMN, clear_apertures)
        self._make_column_readonly(table, CA_COLUMN)

    # ===================
    # Event handlers
    # ===================
    def contextMenuEvent(self, event):
        # Show the context menu
        self.contextMenu.exec(event.globalPos())

    def insert_row_action_handler(self):
        row = self.table.currentRow()
        if row < self.table.rowCount()-1:
            self.table.insertRow(row+1)
            self.table.setItem(row+1, 0, QTableWidgetItem('Spherical'))
            for i, entry in enumerate(["0", np.inf, 0.0, 1.0, np.inf, 0.0]):
                cell = QTableWidgetItem()
                cell.setData(Qt.ItemDataRole.EditRole, float_as_str(entry) if type(entry)==float else str(entry))
                self.table.setItem(row+1, i+1, cell)
            # If the row was inserted above the stop, the index of the stop surface 
            # needs to be incremented by one.
            if row < self.stop_surface:
                self.stop_surface += 1
        else:
            raise Warning("Cannot insert surface after IMAG")
    
    def remove_row_action_handler(self):
        row = self.table.currentRow()
        if self.table.item(row, 0).data(0) in ['OBJ', 'STOP', 'IMAG']:
            raise Warning("Cannot remove row since at least three surfaces (OBJ, STOP, IMAG) are required.")
        else:
            self.table.removeRow(row)

    def change_stop_action_handler(self):
        row = self.table.currentRow()
        old = self.stop_surface
        print("old=", old)
        if (row != old):
            # unset surface label and stop flag 
            self.table.setItem(old, 0, QTableWidgetItem('Spherical'))
            cell = QTableWidgetItem()
            cell.setData(Qt.ItemDataRole.EditRole, float(0))
            self.table.setItem(old, 1, cell) # unset stop flag 
            # update: set surface label and stop flag  
            self.table.setItem(row, 0, QTableWidgetItem('STOP'))
            cell = QTableWidgetItem()
            cell.setData(Qt.ItemDataRole.EditRole, float(1))
            self.table.setItem(row, 1, cell) # set stop flag 
            self.stop_surface = row
            
    def on_cell_activated(self, row, column):
        "Highlight current surface in the lens drawing"
        print("column=", column, "row=", row)
        print("current row=", self.table.currentRow())

        # if self.parent.raytraceWindow is not None:
            # self.parent.raytraceWindow.highlight_surface(row)

        if hasattr(self.parent,'raytraceWindow'):
            if self.parent.raytraceWindow is not None:
                print("self.parent.raytraceWindow=", self.parent.raytraceWindow)
                self.parent.raytraceWindow.highlight_surface(row)

    # def on_item_changed(self, item):
    #     row = item.row()
    #     column = item.column()
    #     print("Item changed ")
    #     item.setText("blablabla")


class LensEditorConfig(LensEditorConfigBase):

    def __init__(self, parent, other=None):
        super().__init__(parent)

        if other is None:
            # init default table (done by base class constructor)
            print("Calling default constructor")
            pass                    
        else:
            # copy table items from `other` 
            self.table.setRowCount(other.table.rowCount())
            for i in range(other.table.rowCount()):
                for j in range(other.table.columnCount()):
                    item_ij = other.table.item(i,j)                    
                    if item_ij  is not None:
                        cell = QTableWidgetItem()
                        cell.setData(Qt.ItemDataRole.EditRole, item_ij.data(0))
                        self.table.setItem(i, j, cell)
            self.stop_surface = other.stop_surface
            self._set_table_entry_settings()            

    def read_config(self, DATA):
        DATA = np.asarray(DATA)
        nrows, ncols = DATA.shape
        self.table.setRowCount(nrows)
        self.STOP_FOUND = False
        for i in range(nrows):            
            for j in range(ncols):
                cell = QTableWidgetItem()
                cell.setData(Qt.ItemDataRole.EditRole, float(DATA[i,j]))
                self.table.setItem(i, j, cell)
            if i==0:
                self.table.setItem(i, 0, QTableWidgetItem('OBJ'))
            elif DATA[i,1] == 1:
                if not self.STOP_FOUND:
                    self.table.setItem(i, 0, QTableWidgetItem('STOP'))
                    self.STOP_FOUND = True
                    self.stop_surface = i
                else:
                    raise Warning("There must be one and only one aperture stop.")
            elif i==self.table.rowCount()-1:
                self.table.setItem(i, 0, QTableWidgetItem('IMAG'))
            else:
                self.table.setItem(i, 0, QTableWidgetItem('Spherical'))
                
        if not self.STOP_FOUND:
            raise Warning("There must be one and only one aperture stop.")

        self._set_table_entry_settings()


class RaytraceDiagram(QWidget):
    "This widget has no parent and will appear as a free floating window."

    def __init__(self, display_interface: DisplayInterface, ray_data: MeridionalRayData):
        super().__init__()

        self.setWindowTitle("Layout")
        self.display_interface = display_interface
        # Create the Figure and Canvas
        self.fig = self.display_interface.init_figure()
        self.canvas = FigureCanvas(self.fig)

        # Optional toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)        

        self.fig = self.display_interface.plot_spherical_surfaces(self.fig)
        self.fig = self.display_interface.plot_ray_bundles(ray_data, self.fig)

        self.canvas.draw()        
        self.canvas.flush_events()

    def highlight_surface(self, surf_nr):
        self.fig = self.display_interface.highlight_surface(surf_nr)
        self.canvas.draw()
        self.canvas.flush_events()


class RaySpotDiagram(QWidget):
    "This widget has no parent and will appear as a free floating window."

    def __init__(self, display_interface: DisplayInterfaceRayspot, ray_data: NonmeridionalRayData):
        super().__init__()
        self.setWindowTitle("Ray Spot Diagram")
        self.display_interface = display_interface
        self.ray_data = ray_data

        self.fig = self.display_interface.init_figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.surf_input = QDoubleSpinBox()
        self.surf_input.setPrefix("surface: ")
        self.surf_input.setDecimals(0)
        self.surf_input.setMinimum(0)
        self.surf_input.setMaximum(self.ray_data.num_surfs-1)
        self.surf_input.setValue(self.ray_data.num_surfs-1)

        layout = QVBoxLayout()
        layout.addWidget(self.surf_input)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # even handler
        self.surf_input.valueChanged.connect(self.update_rayspot_diagram)

        self.update_rayspot_diagram()

    def update_rayspot_diagram(self):
        if self.fig is not None:
            for ax in self.fig.axes:
                ax.clear()                
        surf = int(self.surf_input.value())
        self.fig = self.display_interface.plot_ray_spots(self.ray_data, surf=surf, fig=self.fig)
        self.canvas.draw()
        self.canvas.flush_events()


class RayFanDiagram(QWidget):
    "This widget has no parent and will appear as a free floating window."

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ray Fan Plot")
        self.label = QLabel("Ray Fan Plot")
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

class ModulationTransferFunctionDiagram(QWidget):
    "This widget has no parent and will appear as a free floating window."

    def __init__(self):
        super().__init__()
        self.label = QLabel("MTF")
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

class SeidelAberrationCoefficientsDiagram(QWidget):
    "This widget has no parent and will appear as a free floating window."

    def __init__(self):
        super().__init__()
        self.label = QLabel("Seidel Aberration Coefficients")
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

class GaussianBeamDiagram(QWidget):
    "This widget has no parent and will appear as a free floating window."

    def __init__(self):
        super().__init__()
        self.label = QLabel("Propagation of Gaussian Beams")
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)


app = QApplication(sys.argv)
w = LensEditor()
app.exec()
