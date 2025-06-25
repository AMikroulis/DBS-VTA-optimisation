import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import itertools

# Fixed parameters from the Excel structure
sides = ['R', 'L']
contacts = [0, 1, 2, 3]
effects = ['rigidity', 'akinesia', 'tremor', 'sideeffects']
currents = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]

ui_dict = {'R': 'Right', 'L': 'Left', 0: '0 (bottom)', 1: '1', 2: '2', 3: '3 (top)', 'rigidity': 'Rigidity', 'akinesia': 'Akinesia', 'tremor': 'Tremor', 'sideeffects': 'Side effects'}

class DataEntryApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Patient Data Entry")
        self.patient_id = None
        self.current_combination = 0
        self.combinations = list(itertools.product(sides, contacts, effects))
        self.patient_data = [np.nan] * 288  # 288 rows total
        self.entries = []

        # Start screen: patient ID
        self.id_label = tk.Label(master, text="Enter patient ID (e.g., patient_1):")
        self.id_entry = tk.Entry(master)
        self.start_button = tk.Button(master, text="Start", command=self.start_entry)

        self.id_label.grid(row=0, column=0, padx=10, pady=10)
        self.id_entry.grid(row=0, column=1, padx=10, pady=10)
        self.start_button.grid(row=1, column=0, columnspan=2, pady=10)

    def start_entry(self):
        self.patient_id = self.id_entry.get().strip()
        if not self.patient_id:
            messagebox.showerror("Error", "Please enter a patient ID!")
            return

        # Check if ID exists in Excel
        try:
            df = pd.read_excel('clinical_review.xlsx', header=None)
            if self.patient_id in df.iloc[0].values:
                messagebox.showerror("Error", f"ID '{self.patient_id}' already exists. Pick a new one!")
                return
        except FileNotFoundError:
            messagebox.showerror("Error", "Excel file 'clinical_review.xlsx' not found!")
            return

        # Clear start screen, begin data entry
        self.id_label.destroy()
        self.id_entry.destroy()
        self.start_button.destroy()
        self.show_combination()

    def is_valid_sequence(self, values):
        found_blank = False
        for val in values:
            if val == '':  # Blank entry
                found_blank = True
            elif found_blank:  # Non-blank after a blank
                return False
        return True

    def show_combination(self):
        if self.current_combination >= 32:
            self.save_data()
            return
        # Set a fixed size for the window
        self.master.geometry("500x250")

        side, contact, effect = self.combinations[self.current_combination]

        # Clear old widget
        for widget in self.master.winfo_children():
            widget.destroy()

        # Show current combination
        label_text = f"Side: {ui_dict[side]}, Contact: {ui_dict[contact]}, Effect: {ui_dict[effect]}"
        desc_label = tk.Label(self.master, text=label_text, font=("Arial", 14), width=40)
        desc_label.grid(row=0, column=0, columnspan=10, pady=10)

        # Input fields for currents
        self.entries = []
        for i, current in enumerate(currents):
            tk.Label(self.master, text=f"{current}:").grid(row=1, column=i, padx=5)
            entry = tk.Entry(self.master, width=5)
            entry.grid(row=2, column=i, padx=5)
            self.entries.append(entry)

        # Progress indicator
        progress = f"Step {self.current_combination + 1} of 32"
        tk.Label(self.master, text=progress).grid(row=3, column=0, columnspan=10, pady=10)

        # Navigation buttons
        prev_button = tk.Button(self.master, text="Previous", command=self.previous)
        if self.current_combination == 31:
            # Last step: "Save" button in red
            next_button = tk.Button(self.master, text="Save", command=self.next, fg="red")
        else:
            # All other steps: "Next" button
            next_button = tk.Button(self.master, text="Next", command=self.next)

        prev_button.grid(row=4, column=0, columnspan=5, padx=10, pady=10)
        next_button.grid(row=4, column=5, columnspan=5, padx=10, pady=10)

        if self.current_combination == 0:
            prev_button.config(state=tk.DISABLED)

    def next(self):
        values = [entry.get().strip() for entry in self.entries]
        _, _, effect = self.combinations[self.current_combination]

        # Validate for rigidity, akinesia, tremor
        if effect in ['rigidity', 'akinesia', 'tremor']:
            if not self.is_valid_sequence(values):
                messagebox.showerror("Error", "You cannot have lower current blanks before entering scores for higher currents.")
                return

        # Process values
        processed_values = []
        for val in values:
            if val:
                try:
                    processed_values.append(np.abs(float(val)))
                except ValueError:
                    messagebox.showerror("Error", f"'{val}' isn’t a number! Use numbers or leave blank.")
                    return
            else:
                processed_values.append(np.nan)

        # Store and proceed
        start_idx = self.current_combination * 9
        self.patient_data[start_idx:start_idx + 9] = processed_values
        self.current_combination += 1
        self.show_combination()

    def previous(self):
        if self.current_combination > 0:
            self.current_combination -= 1
            self.show_combination()

    def save_data(self):
        # Read the Excel file without header assumption
        df = pd.read_excel('clinical_review.xlsx', header=None)

        # Validate structure: should have 290 rows (header + blank + 288 data)
        if len(df) != 290:
            raise ValueError(f"Expected 290 rows (header + blank + 288 data), but got {len(df)}")

        # Get current patient codes from the first row
        patient_codes = df.iloc[0].tolist()

        # Add new patient code to the first row
        patient_codes.append(self.patient_id)
        df.columns = range(len(df.columns))  # Temporarily set integer columns
        df = df.reindex(columns=range(len(patient_codes)))  # Expand columns if needed
        df.iloc[0] = patient_codes

        # Ensure second row is blank
        df.iloc[1] = [np.nan] * len(df.columns)

        # Add patient data to rows 2 to 289 (indices 2 to 289, corresponding to rows 3–290 in Excel)
        df.iloc[2:290, -1] = self.patient_data

        # Save to Excel without index or header manipulation
        df.to_excel('clinical_review.xlsx', index=False, header=False)

        messagebox.showinfo("Done", f"Data for {self.patient_id} saved. You’re good to go!")
        print(f"App closed. Data for '{self.patient_id}' saved.")
        self.master.quit()

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = DataEntryApp(root)
    root.mainloop()
    try:
        app.master.destroy()
    except:
        print("App closed.")