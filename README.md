# DBS-VTA-optimisation

## Requirements
To successfully run the DBS-VTA-optimisation project, ensure you have the following software and libraries installed:

Python: Ensure you have a compatible version of Python installed (preferably Python 3.6 or later).

OSS-DBSv2: Install the OSS-DBSv2 package from https://github.com/SFB-ELAINE/OSS-DBSv2. Follow the installation instructions provided in the repository, and update the subprocess source paths in contacts_mt.py if necessary. An older snapshot of this software is placed in the resources directory.

Python Libraries: Install the following Python libraries using pip:
pip install numpy pandas matplotlib scipy pyqt5 pyqtgraph ngsolve dipy

OSS-DBSv2 prerequisites:

NEURON: Download and install NEURON from https://neuron.yale.edu/.

ngsolve (handled above)

dipy (handled above)

## Operation

### Inputs
This software relies on Lead-DBS/OSS-DBS native space reconstructions with the DISTAL atlas. Only one reconstruction is needed, and only once, per patient folder. This can be done with either the 3D rendering or Lead Group stimulation setup: Create a new stimulation, give it a name including "ossdbs", and use OSS-DBS instead of the default (Fieldtrip/Simbio). Lead-DBS fails when rendering both hemispheres with OSS-DBS, so only select the Right hemisphere and add a dummy contact selection with a non-zero current in the first box.

### Operation
Start the app by clicking VTA_start, or calling it from the terminal. For example:
python3 VTA_start.py

A window “VTA optimisation” should appear, with a button on the top-left, to select a leaddbs folder. Click it and navigate to a “leaddbs” folder (it will automatically locate any “leaddbs” folders up to 3 folder levels down).
A list of the available patient folders will appear in the list box on the left. You can select one or more folders to run the calculation (1st tab).

#### Calculate
The contact selection uses a contact ranking system. The best 1-2 contacts are always selected (1 contact if using an old electrode type with non-directional contacts). A 3rd contact can be added if its rank score is close to the top selection. 

The contact ranking is based on the geometry of the configuration or the geometry of the configuration and clinical review information. An example of a clinical review record is located in the "clinical review input" folder (clinical_review.xlsx). The data can be filled either directly or through the helper python script (which ensure no inappropriate data formats can be entered). The column headers must match the patient folder names (without the "sub-" prefix by Lead-DBS). The column and row format must be maintained (number of rows, no gaps between columns). If a "clinical_review.xlsx" file is placed in the "leaddbs" folder it will be used by the main app as additional input for the contact suggestions (a "nudged" version of the contact scores will appear in the output in the terminal).

Selecting “bipolar” will use the same exact contacts as the monopolar calculation, with 2 differences: 
1.	in Bipolar, the case contact is disabled, and both (+) and (-) poles are assigned to numbered lead contacts,
2.	the negative pole is assigned to the contact with the top rank score.

The polarity selection is arbitrary, with the sole intention to concentrate the electric field closer to the best contact.
The number of selected contacts can be limited using the box.

The current selection boxes set a lower and upper boundary for the current suggestion, and the initial VTA estimation. 
Check the “multiple currents” option to calculate four additional VTAs for currents at 0.8, 1.6, 2.4, 3.2 mA; this increases the fitting accuracy later.
Click Run to start the calculations, or Reset to re-initialise the window.

The calculation works best in machines with at least 4 cores (or 2, with hyperthreading), because all calculations are supposed to run in parallel (on multiple threads). More available cores further speed up certain parts of the calculation. Expect a runtime of 15-20 minutes per patient folder, with minimal recommended specs (4 cores), possibly down to 5-10 minutes with bigger/faster systems.

The output is written to a csv file (“overlaps.csv”), in “leaddbs” folder.
It contains the folder id, hemisphere, selected contacts, current (the suggested current is calculated right after the initial (4 mA) current), and coverage of STN subregions (fraction between 0 and 1). “VTA_motor” is the portion of the VTA that stays within the motor STN.

The contacts are numbered in a manufacturer-independent way, starting from the tip of the lead. For example, on a lead with contacts "9", "10A-C", "11A-C", "12", the contacts will be indicated as 0 (for "9"), 1-3 (for "10A-C", respectively), 4-6 (for "11A-C", respectively), 7 (for "12").


#### Single patient
The second tab allows fine-tuning of the current selection. Select a leaddbs folder (top left button) where some processing has been done, as described above. Then select a processed patient folder from the list, and click "load". Cycle through the Left and Right hemispheres to ensure both are loaded.

A curve will be displayed of the left panel showing the fraction of the Motor STN subregion estimated to be covered by the VTA (field intensity of 200 mV/mm at least) depending on the current. The curve is a cubic spline fit and relies on the "multiple currents" option in the calculation process - it will revert to a simple linear interpolation if not enough current-VTA pairs were calculated previously.

Detailed contact and current information is updated in the box with the “current settings” under the panel.
Moving the slider under the left panel adjusts the current (moving vertical blue line on the plot).

The bar plot on the right shows the fractions of the motor, associative and limbic subregions of the STN that are expected to be covered using the selected current.

The box on the bottom-right can display an indicator on the curve using the harmonic means of motor STN overlap with :
1.	VTA fraction within the motor STN (conservative – aims to limit VTA leakage)
2.	Fraction of not-covered limbic STN (tries to maximise motor STN coverage given the limbic STN coverage)
3.	Fraction of not-covered associative STN (tries to maximise motor STN coverage given the associative STN coverage).

A small slider adjusts how much of the harmonic mean range (from the maximum harmonic mean) will be highlighted on the curve. Note: this is not percentile-based; it uses the range from the minimum harmonic mean (0%) to the maximum (100%) - the distribution is not considered. For example setting the slider to 50% will highlight the <i>top 50% of the range</i> from 50% of the harmonic mean range (not the median, but average of minimum and maximum) to 100% (the maximum).

Clicking the “save” button saves the current settings and plots for both hemispheres to a printable html file, with a timestamp (creation date and time).


## Licensing and Copyright notice
This software is released under the terms of the GNU General Public License v3.0 (GPL-3.0). You can redistribute it and/or modify it under the terms of the GPL-3.0 as published by the Free Software Foundation.

### Copyright Holders
DBS-VTA-optimisation: © Apostolos Mikroulis, Department of Cybernetics, Faculty of Electrical Engineering, Czech Technical University, 2025.

OSS-DBSv2: Portions of this software are derived from the OSS-DBS package, © 2017 Christian Schmidt; © 2023-2024 Jan Philipp Payonk, Johannes Reding, Julius Zimmermann, Konstantin Butenko, Shruthi Chakravarthy, Tom Reincke; © 2024 Julius Zimmermann, Konstantin Butenko. The OSS-DBS package is licensed under the GPL-3.0.

### Acknowledgments
We acknowledge the authors and contributors of the OSS-DBSv2 package for their work, which forms a crucial part of this project. For more information on OSS-DBSv2, please visit OSS-DBS GitHub Repository at https://github.com/SFB-ELAINE/OSS-DBSv2.

### Disclaimer
This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

