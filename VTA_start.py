# pyinstaller requirements
import struct

# main imports
import numpy as npy
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.interpolate import CubicSpline
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QDialog
import pyqtgraph as pg
import os
import re
from datetime import datetime
from pyqtgraph.exporters import ImageExporter

# ossdbs prerequisites:
import ngsolve
import dipy
import dipy.tracking
import dipy.tracking.metrics
import fileinput
import neuron
from abc import ABC, abstractmethod
from typing import Optional


# subporcess and custom libraries
from runpy import run_path
import sys
from modules.VTA_optimisation_Qt_extended import Ui_Dialog
import modules.contacts_mt as contacts_mt
import modules.contact_selection_module as contact_selection_module
import modules.current_estimation as current_estimation

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class ui_window(qtw.QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_Dialog()
        print("UI object created")
        self.ui.setupUi(self)
        print("UI setup complete, tabWidget:", self.ui.tabWidget)

        # Create aliases for radio buttons
        self.ui.radioButton_left = self.ui.radioButton
        self.ui.radioButton_right = self.ui.radioButton_2
        
        self.plot1 = self.ui.graphicsView
        self.plot2 = self.ui.graphicsView_2

        self.ui.tabWidget.setVisible(True)
        self.ui.tabWidget.setCurrentIndex(0)
        self.ui.groupBox.setVisible(True)
        self.ui.mono.setVisible(True)
        self.ui.bi.setVisible(True)
        self.ui.progressBar.setVisible(True)
        self.ui.current_groupbox.setVisible(True)
        self.ui.max_current.setVisible(True)
        self.ui.min_current.setVisible(True)
        self.ui.label_2.setVisible(True)
        self.ui.qt_spinbox_lineedit2.setVisible(True)
        self.ui.initial_VTA_current.setVisible(True)
        self.ui.label_3.setVisible(True)
        self.ui.multiple_current_calc.setVisible(True)
        self.ui.n_contacts.setVisible(True)
        self.ui.label_4.setVisible(True)
        self.ui.label_5.setVisible(True)
        self.ui.run_button.setVisible(True)
        self.ui.reset_button.setVisible(True)
        self.ui.listWidget.setVisible(True)
        self.ui.listWidget.setEnabled(True)
        self.ui.qt_scrollarea_vcontainer.setVisible(False)
        self.ui.qt_scrollarea_viewport.setVisible(False)
        self.ui.listWidget.setStyleSheet("background-color: white; color: black;")
        # Connect existing first tab signals
        self.ui.browse_folder.clicked.connect(self.browse_folder_clicked)
        self.ui.listWidget.itemSelectionChanged.connect(self.item_selection_changed)
        self.ui.mono.clicked.connect(self.mono_clicked)
        self.ui.bi.clicked.connect(self.bi_clicked)
        self.ui.reset_button.clicked.connect(self.reset_clicked)
        self.ui.run_button.clicked.connect(self.run_clicked)
        self.ui.load_patient.clicked.connect(self.load_patient_data)
        self.ui.save_patient.clicked.connect(self.save_patient_data)

        self.ui.radioButton_3.setChecked(True)

        # Initialize variables
        self.basedir = ''
        self.patients_list = []
        self.contacts_mode = ''
        self.ipn_subset = []
        self.min_current = 0.0
        self.max_current = 0.0
        self.initial_current = 0.0
        self.n_contacts = 0
        self.multiple_estimates = False
        
        
        # Second tab setup
        self.result_df = None  # To store interpolated dataframe
        self.setup_second_tab()


        # Connect radio buttons to update the slider when hemisphere changes
        self.ui.radioButton.toggled.connect(self.update_slider_for_hemisphere)
        self.ui.radioButton_2.toggled.connect(self.update_slider_for_hemisphere)

        # Connect slider changes to update the current and plots
        self.ui.horizontalSlider.valueChanged.connect(self.slider_value_changed)

        # Set the initial slider value based on the default hemisphere
        self.update_slider_for_hemisphere()

        
        self.left_current = 1.5  # Default to 1.5 mA
        self.right_current = 1.5  # Default to 1.5 mA


        # Sample data (replace with actual data)
        self.data = {
            "Folder1": {"left": (npy.linspace(0.5, 4, 36), npy.random.rand(36)),
                        "right": (npy.linspace(0.5, 4, 36), npy.random.rand(36))},
            "Folder2": {"left": (npy.linspace(0.5, 4, 36), npy.random.rand(36)),
                        "right": (npy.linspace(0.5, 4, 36), npy.random.rand(36))},
            "Folder3": {"left": (npy.linspace(0.5, 4, 36), npy.random.rand(36)),
                        "right": (npy.linspace(0.5, 4, 36), npy.random.rand(36))}
        }

        self.update_plots()

        

    def setup_second_tab(self):
        # Connect slider and radio buttons
        self.ui.horizontalSlider.valueChanged.connect(self.update_plots)
        self.ui.radioButton_3.toggled.connect(self.update_harmonic_mean)
        self.ui.radioButton_4.toggled.connect(self.update_harmonic_mean)
        self.ui.radioButton_5.toggled.connect(self.update_harmonic_mean)
        self.ui.horizontalSlider_2.valueChanged.connect(self.update_checkbox_label)
        self.ui.checkBox.toggled.connect(self.update_plots)

        # Set slider range to match 0.5-4.0 mA
        self.ui.horizontalSlider.setMinimum(5)
        self.ui.horizontalSlider.setMaximum(40)
        self.ui.horizontalSlider.setValue(15)  # Default to 1.5 mA
        self.update_checkbox_label(self.ui.horizontalSlider_2.value())
        self.update_plots()

    def browse_folder_clicked(self):
        self.basedir = qtw.QFileDialog.getExistingDirectory(self, "Select Folder")
        print("Selected folder:", self.basedir)

        if not self.basedir:
            return

        # check if the selected folder or any parent is the target folder
        leaddbs_path = self.find_leaddbs_in_path(self.basedir)
        if leaddbs_path:
            self.basedir = leaddbs_path
        else:
            # search downwards in subdirectories
            self.basedir = self.find_leaddbs_path(self.basedir)

        print("Found leaddbs path:", self.basedir)

        if self.basedir:
            self.ui.listWidget.setEnabled(True)
            self.ui.listWidget.setVisible(True)
            self.patients_list = [f for f in os.listdir(self.basedir) if f.startswith('sub-')]
            self.ui.listWidget.clear()
            self.ui.listWidget.addItems(self.patients_list)
            self.ui.listWidget.sortItems(qtc.Qt.AscendingOrder)
            self.ui.listWidget.update()
            self.ui.listWidget.repaint()
            print("Patient folders:", len(self.patients_list))
        else:
            print("Target folder '/derivatives/leaddbs' not found.")

    def find_leaddbs_in_path(self, path):
        """Check if the path contains '/derivatives/leaddbs' by looking upward."""
        components = path.split(os.sep)
        for i in range(len(components) - 1, -1, -1):
            if components[i] == 'leaddbs' and i > 0 and components[i - 1] == 'derivatives':
                return os.sep.join(components[:i + 1])
        return None

    def find_leaddbs_path(self, base_path, level=0):
        """Search downward for '/derivatives/leaddbs' in subdirectories."""
        if level > 3:
            return None
        # check if the current folder is the target
        if os.path.basename(base_path) == 'leaddbs' and os.path.basename(os.path.dirname(base_path)) == 'derivatives':
            return base_path
        # search subdirectories
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                found_path = self.find_leaddbs_path(item_path, level + 1)
                if found_path:
                    return found_path
        return None

    def item_selection_changed(self):
        self.ipn_subset = [item.text() for item in self.ui.listWidget.selectedItems()]
        if self.ipn_subset and self.multiple_estimates:
            self.load_and_interpolate_data()
        
    def load_patient_data(self):
        selected_items = self.ui.listWidget.selectedItems()
        if not selected_items:
            print("No patient selected.")
            return
        # take only the first selected item
        self.ipn_subset = [selected_items[0].text()]
        print(f"Loading data for patient: {self.ipn_subset[0]}.")
        self.load_and_interpolate_data()
        self.update_plots()

    def format_contact_pairs(self, contact_str):
        """Format the contact_pairs string into a readable format with (-) and (+) labels."""
        if not contact_str or contact_str == "N/A":
            return "N/A"
        contacts = re.findall(r'-1|\d', contact_str)
        formatted_contacts = ["case" if c == "-1" else c for c in contacts]
        
        if formatted_contacts[-1] == "case":
            negative_contacts = formatted_contacts[:-1]
            positive_contacts = [formatted_contacts[-1]]
        else:
            negative_contacts = [formatted_contacts[0]]
            positive_contacts = formatted_contacts[1:]
        
        negative_str = ", ".join(negative_contacts) if negative_contacts else ""
        positive_str = ", ".join(positive_contacts) if positive_contacts else ""
        
        result = ""
        if negative_str:
            result += f"(-) : {negative_str}"
        if positive_str:
            if result:
                result += ", "
            result += f"(+) : {positive_str}"
        
        return result

    def mono_clicked(self):
        self.contacts_mode = 'monopolar'

    def bi_clicked(self):
        self.contacts_mode = 'bipolar'

    def get_basedir(self):
        return self.basedir

    def get_ipns(self):
        return self.ipn_subset if hasattr(self, 'ipn_subset') else []

    def get_contacts_mode(self):
        return self.contacts_mode

    def get_min_current(self):
        return self.ui.min_current.value()

    def get_max_current(self):
        return self.ui.max_current.value()

    def get_initial_current(self):
        return self.ui.initial_VTA_current.value()

    def get_n_contacts(self):
        return self.ui.n_contacts.value()

    def get_multiple_estimates(self):
        return self.ui.multiple_current_calc.isChecked()


    def reset_clicked(self):
        self.basedir = ''
        self.patients_list = []
        self.contacts_mode = ''
        self.ipn_subset = []
        self.multiple_estimates = False
        self.ui.listWidget.clear()
        self.ui.max_current.setValue(3.5)
        self.ui.initial_VTA_current.setValue(4.0)
        self.ui.n_contacts.setValue(3)
        self.ui.min_current.setValue(1.0)
        self.ui.multiple_current_calc.setChecked(False)
        self.ui.progressBar.setValue(0)
        self.ui.mono.setChecked(True)
        self.ui.bi.setChecked(False)

    def run_clicked(self):
        if self.ui.listWidget.selectedItems():
            self.settings_thread = settings()
            self.settings_thread.thingy_done.connect(self.on_thingy_done)
            self.settings_thread.start()

    def on_thingy_done(self):
        # in the main thread
        self.load_and_interpolate_data()

    def progressbar_update(self, value):
        self.ui.progressBar.setValue(value)

    def slider_to_mA(self, slider_val):
        return slider_val * 0.1

    def mA_to_slider(self, mA):
        return int(mA / 0.1)

    def update_slider_for_hemisphere(self):
        self.ui.horizontalSlider.blockSignals(True)  # prevent triggering valueChanged
        if self.ui.radioButton.isChecked():  # left hemisphere
            self.ui.horizontalSlider.setValue(self.mA_to_slider(self.left_current))
        elif self.ui.radioButton_2.isChecked():  # right hemisphere
            self.ui.horizontalSlider.setValue(self.mA_to_slider(self.right_current))
        self.ui.horizontalSlider.blockSignals(False)
        self.update_plots()

    def load_and_interpolate_data(self):
        overlaps_file = os.path.join(self.basedir, 'overlaps.csv')
        print("Loading overlaps.csv...")
        if not os.path.exists(overlaps_file):
            print("overlaps.csv not found!")
            return
        print("Interpolating data...")
        df = pd.read_csv(overlaps_file)
        
        new_currents = npy.arange(0.5, 4.1, 0.1)
        results = {'motor': [], 'assoc': [], 'limbic': [], 'motor_VTA': []}

        
        for (ipn, side), group in df.groupby(['ipn', 'side']):
            if ipn not in self.ipn_subset:
                continue
            group = group.sort_values('current (mA)')
            currents = group['current (mA)'].values
            
            # create a df with a single row of zeros
            zero_row = pd.DataFrame({
                'ipn': [ipn],
                'side': [side],
                'current (mA)': [0],
                'motor': [0],
                'assoc': [0],
                'limbic': [0],
                'motor_VTA': [0]
            })

            # concatenate the zero row with the group and sort
            group = pd.concat([zero_row, group], ignore_index=True)
            group = group.sort_values('current (mA)').reset_index(drop=True)

            # update currents to include the new zero point
            currents = group['current (mA)'].values

            print(f"Interpolating {ipn} {side}...")

            for output in results.keys():
                if group[output].isnull().any():
                    continue
                if len(currents) > 2:
                    spline = CubicSpline(currents, group[output].values)
                    predicted_values = npy.clip(spline(new_currents), 0, 1)
                # use linear interpolation if <3 points:
                if len(currents) < 3:
                    predicted_values = npy.interp(new_currents, currents, group[output].values)

                temp_df = pd.DataFrame({
                    'ipn': ipn,
                    'side': side,
                    'current (mA)': new_currents,
                    output: predicted_values
                })

                

                results[output].append(temp_df)

        for output in results:
            results[output] = pd.concat(results[output], ignore_index=True)

        self.result_df = results['motor']
        for output in ['assoc', 'limbic', 'motor_VTA']:
            self.result_df = self.result_df.merge(
                results[output], on=['ipn', 'side', 'current (mA)'], how='outer'
            )
        print(f"Loaded data with shape: {self.result_df.shape}, columns: {self.result_df.columns}.")
        contact_pairs = df[['ipn', 'side', 'contact_pairs']].drop_duplicates()
        self.result_df = self.result_df.merge(contact_pairs, on=['ipn', 'side'], how='left')
        self.update_harmonic_mean()

    def update_harmonic_mean(self):
        if self.result_df is None:
            return

        if self.ui.radioButton_3.isChecked():
            self.result_df['harmonic_mean'] = 2 * (self.result_df['motor'] * self.result_df['motor_VTA']) / (self.result_df['motor'] + self.result_df['motor_VTA'])
        elif self.ui.radioButton_4.isChecked():
            self.result_df['harmonic_mean'] = 2 * (self.result_df['motor'] * (1 - self.result_df['limbic'])) / (self.result_df['motor'] + (1 - self.result_df['limbic']))
        elif self.ui.radioButton_5.isChecked():
            self.result_df['harmonic_mean'] = 2 * (self.result_df['motor'] * (1 - self.result_df['assoc'])) / (self.result_df['motor'] + (1 - self.result_df['assoc']))

        self.result_df['harmonic_mean'] = self.result_df['harmonic_mean'].fillna(0)
        self.update_plots()

    def update_checkbox_label(self, value):
        self.ui.checkBox.setText(f"Highlight top {value} %")
        self.h_mean_threshold = value / 100.0
        self.update_plots()


    def slider_value_changed(self):
        if self.ui.radioButton.isChecked():  # Left hemisphere
            self.left_current = self.slider_to_mA(self.ui.horizontalSlider.value())
        elif self.ui.radioButton_2.isChecked():  # Right hemisphere
            self.right_current = self.slider_to_mA(self.ui.horizontalSlider.value())
        self.update_plots()

    def percentile_slider_changed(self):
        self.update_plots()

    def update_plots(self):
        if self.result_df is None or not self.ipn_subset:
            print("No data to plot.")
            return

        ipn = self.ipn_subset[0]
        side = 'left' if self.ui.radioButton.isChecked() else 'right'

        # Use the current for the selected hemisphere
        slider_val = self.left_current if side == 'left' else self.right_current

        df_subset = self.result_df[(self.result_df['ipn'] == ipn) & (self.result_df['side'] == side)]
        if df_subset.empty:
            print(f"No data for {ipn} on {side} side.")
            return

        df_subset = df_subset.sort_values('current (mA)')
        currents = df_subset['current (mA)'].values
        motor_values = df_subset['motor'].values

        # plot 1: Motor STN coverage vs current
        self.plot1.clear()
        self.plot1.plot(currents, motor_values, pen=pg.mkPen(color='#a080d0', width=3), name='Motor Coverage')
        self.plot1.addLine(x=slider_val, pen='#4080a0')  # vertical line at selected current
        self.plot1.setLabel('bottom', 'Current (mA)')
        self.plot1.setLabel('left', 'Motor STN Coverage')

        if self.ui.checkBox.isChecked():
            harmonic_mean = df_subset['harmonic_mean'].values
            max_hm = harmonic_mean.max()
            min_hm = harmonic_mean.min()
            range_hm = max_hm - min_hm
            threshold = max_hm - (self.h_mean_threshold * range_hm)
            
            
            mask = harmonic_mean >= threshold
            highlight_currents = currents[mask]
            highlight_motor = motor_values[mask]
            self.plot1.plot(highlight_currents, highlight_motor, pen=None, symbol='o', symbolBrush='#40c040', symbolSize=10)


        # plot 2: Bar plot at selected current
        idx = npy.abs(currents - slider_val).argmin()
        selected_data = df_subset.iloc[idx]
        categories = ['motor', 'limbic', 'assoc']
        values = [selected_data[cat] for cat in categories]

        self.plot2.clear()
        bar_graph = pg.BarGraphItem(x=npy.arange(len(categories)), height=values, width=0.6, brush='#40a0c0')
        self.plot2.addItem(bar_graph)
        self.plot2.setYRange(0, 1)  # fix y-axis to [0,1]
        self.plot2.setLabel('bottom', 'STN subregions')
        self.plot2.setLabel('left', 'Overlap')
        self.plot2.getAxis('bottom').setTicks([[(i, cat) for i, cat in enumerate(categories)]])

        left_rows = self.result_df[(self.result_df['ipn'] == ipn) & (self.result_df['side'] == 'left')]
        if not left_rows.empty:
            left_contact_str = self.format_contact_pairs(left_rows['contact_pairs'].iloc[0])
            left_data = left_rows.iloc[npy.abs(left_rows['current (mA)'] - self.left_current).argmin()]
            left_text = f"Motor: {left_data['motor']:.2f}, Limbic: {left_data['limbic']:.2f}, Associative: {left_data['assoc']:.2f}"
        else:
            left_contact_str = "N/A"
            left_text = "No data available"

        
        right_rows = self.result_df[(self.result_df['ipn'] == ipn) & (self.result_df['side'] == 'right')]
        if not right_rows.empty:
            right_contact_str = self.format_contact_pairs(right_rows['contact_pairs'].iloc[0])
            right_data = right_rows.iloc[npy.abs(right_rows['current (mA)'] - self.right_current).argmin()]
            right_text = f"Motor: {right_data['motor']:.2f}, Limbic: {right_data['limbic']:.2f}, Associative: {right_data['assoc']:.2f}"
        else:
            right_contact_str = "N/A"
            right_text = "No data available"

        
        self.ui.textBrowser.clear()
        if left_contact_str != "N/A":
            self.ui.textBrowser.append(f"Left contacts: {left_contact_str}")
            self.ui.textBrowser.append(f"Left Hemisphere (at {self.left_current:.1f} mA): {left_text}")
        else:
            self.ui.textBrowser.append("Left Hemisphere: No data available")

        if right_contact_str != "N/A":
            self.ui.textBrowser.append(f"Right contacts: {right_contact_str}")
            self.ui.textBrowser.append(f"Right Hemisphere (at {self.right_current:.1f} mA): {right_text}")
        else:
            self.ui.textBrowser.append("Right Hemisphere: No data available")
    def save_patient_data(self):
        """Save text browser content, contact pairs, and plots for both hemispheres to an HTML file."""
        if not hasattr(self, 'ipn_subset') or not self.ipn_subset:
            print("No patient selected.")
            return

        ipn = self.ipn_subset[0]
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = ipn

        suggested_filename = f"{folder_name}_both_{current_datetime}.html"

        # save dialog
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Patient Data", suggested_filename, "HTML Files (*.html)")
        if not file_path:  # exit on cancel
            print("Save operation cancelled.")
            return

        # get the directory to save images (same as HTML file)
        save_dir = os.path.dirname(file_path)

        # get data for left hemisphere
        left_df = self.result_df[(self.result_df['ipn'] == ipn) & (self.result_df['side'] == 'left')]
        if not left_df.empty:
            left_contact_pairs = left_df['contact_pairs'].iloc[0]
            left_current = self.left_current
            left_data = left_df.iloc[npy.abs(left_df['current (mA)'] - left_current).argmin()]
            left_text = f"Motor STN: {100*left_data['motor']:.0f}%, Limbic STN: {100*left_data['limbic']:.0f}%, Associative STN: {100*left_data['assoc']:.0f}%"
        else:
            left_contact_pairs = "N/A"
            left_text = "No data available"

        # get data for right hemisphere
        right_df = self.result_df[(self.result_df['ipn'] == ipn) & (self.result_df['side'] == 'right')]
        if not right_df.empty:
            right_contact_pairs = right_df['contact_pairs'].iloc[0]
            right_current = self.right_current
            right_data = right_df.iloc[npy.abs(right_df['current (mA)'] - right_current).argmin()]
            right_text = f"Motor STN: {100*right_data['motor']:.0f}%, Limbic STN: {100*right_data['limbic']:.0f}%, Associative STN: {100*right_data['assoc']:.0f}%"
        else:
            right_contact_pairs = "N/A"
            right_text = "No data available"

        # store current state
        original_side = 'left' if self.ui.radioButton.isChecked() else 'right'

        # define image filenames and full paths
        left_plot1_filename = f"{folder_name}_left_plot1_{current_datetime}.png"
        left_plot2_filename = f"{folder_name}_left_plot2_{current_datetime}.png"
        right_plot1_filename = f"{folder_name}_right_plot1_{current_datetime}.png"
        right_plot2_filename = f"{folder_name}_right_plot2_{current_datetime}.png"

        left_plot1_path = os.path.join(save_dir, left_plot1_filename)
        left_plot2_path = os.path.join(save_dir, left_plot2_filename)
        right_plot1_path = os.path.join(save_dir, right_plot1_filename)
        right_plot2_path = os.path.join(save_dir, right_plot2_filename)

        # set to left hemisphere and capture plots
        self.ui.radioButton.setChecked(True)
        self.update_plots()
        exporter1 = ImageExporter(self.plot1.plotItem)
        exporter1.export(left_plot1_path)
        exporter2 = ImageExporter(self.plot2.plotItem)
        exporter2.export(left_plot2_path)

        # set to right hemisphere and capture plots
        self.ui.radioButton_2.setChecked(True)
        self.update_plots()
        exporter1.export(right_plot1_path)
        exporter2.export(right_plot2_path)

        # restore original state
        if original_side == 'left':
            self.ui.radioButton.setChecked(True)
        else:
            self.ui.radioButton_2.setChecked(True)
        self.update_plots()

        # create HTML content with relative paths (just filenames)
        html_content = f"""
        <html>
        <head><title>Folder Data - {folder_name} - {current_datetime}</title></head>
        <body>
        <h1>Folder: {folder_name}</h1>
        <h2>Date and Time: {current_datetime}</h2>

        <h2>Left Hemisphere</h2>
        <h3>Contacts: {self.format_contact_pairs(left_contact_pairs)}</h3>
        <h3>Current: {left_current:.1f} mA</h3>
        <p>{left_text}</p>
        <img src="{left_plot1_filename}" alt="Left Motor STN Coverage Plot">
        <img src="{left_plot2_filename}" alt="Left VTA Spillover Plot">

        <h2>Right Hemisphere</h2>
        <h3>Contacts: {self.format_contact_pairs(right_contact_pairs)}</h3>
        <h3>Current: {right_current:.1f} mA</h3>
        <p>{right_text}</p>
        <img src="{right_plot1_filename}" alt="Right Motor STN Coverage Plot">
        <img src="{right_plot2_filename}" alt="Right VTA Spillover Plot">
        </body>
        </html>
        """

        # write HTML to file
        with open(file_path, 'w') as file:
            file.write(html_content)
        print(f"Data saved to {file_path}.")


class settings(QThread):
    thingy_done = pyqtSignal()
    def __init__(self, *args, **kwargs):
        super(settings, self).__init__(*args, **kwargs)
        self.basedir = ''
        self.patients_list = []
        self.selected_ipns = []
        self.contacts_mode = ''
        self.min_current = 0.0
        self.max_current = 0.0
        self.initial_current = 0.0
        self.n_contacts = 0
        self.multiple_estimates = False

    def run(self):
        self.basedir = window.get_basedir()
        self.patients_list = window.get_ipns()
        self.selected_ipns = self.patients_list
        self.contacts_mode = window.get_contacts_mode()
        self.min_current = window.get_min_current()
        self.max_current = window.get_max_current()
        self.initial_current = window.get_initial_current()
        self.n_contacts = window.get_n_contacts()
        self.multiple_estimates = window.get_multiple_estimates()

        # disable UI during processing
        for widget in [window.ui.browse_folder, window.ui.bi, window.ui.initial_VTA_current, 
                       window.ui.listWidget, window.ui.max_current, window.ui.min_current, 
                       window.ui.multiple_current_calc, window.ui.reset_button, window.ui.run_button, 
                       window.ui.mono, window.ui.n_contacts]:
            widget.setDisabled(True)

        check_status = contacts_mt.external_call(
            self.basedir, self.selected_ipns, self.contacts_mode, self.min_current,
            self.max_current, self.initial_current, self.multiple_estimates, self.n_contacts,
            'distal_native', window.ui.progressBar
        )

        if check_status == 0:
            print("Contacts calculation completed successfully.")
            self.thingy_done.emit()
        else:
            print("Error in contacts calculation.")
        window.accept()

    
if __name__ == "__main__":
    
    # check if bundled and '--run-script'
    if getattr(sys, 'frozen', False) and len(sys.argv) > 1 and sys.argv[1] == '--run-script':
        script_path = sys.argv[2]  # path to ossdbs main.py
        script_args = sys.argv[3:]  # arguments (json)
        # adjust sys.argv for the script
        original_argv = sys.argv[:]
        sys.argv = [script_path] + script_args
        try:
            run_path(script_path, run_name='__main__')
        finally:
            sys.argv = original_argv  # restore original sys.argv
        sys.exit(0)
    else:
        import sys
        app = qtw.QApplication([])
        window = ui_window()
        window.show()
        sys.exit(app.exec_())