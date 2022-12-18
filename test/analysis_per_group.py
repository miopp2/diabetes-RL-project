"""
Changes that have been made:
- our implementation
"""

import os
import glob
import pandas as pd
from test.analysis.report import report

folder_to_analyse = "2022-12-17_14-58-23"

groups = ["/adolescent*.csv", "/adult*.csv", "/child*.csv", "/*#*.csv"]
# groups = ["/adolescent*.csv"]
part_path = "./results/" + folder_to_analyse

paths_all = [os.path.join(part_path, controller) for controller in os.listdir(part_path)]
paths_controller = [controller for controller in paths_all if os.path.isdir(controller)]

group_paths = [path + group for group in groups for path in paths_controller]

for i, group in enumerate(group_paths):
    path_ctrl = (os.path.splitext(group)[0]).replace('*', '_')
    os.makedirs(path_ctrl, exist_ok=True)
    files = glob.glob(group)
    patient_names = [os.path.splitext(os.path.basename(file))[0] for file in files]
    results = [pd.read_csv(file, index_col='Time') for file in files]
    df = pd.concat(results, keys=patient_names)
    sim_days = int(len(df) / len(files) / 480)
    if df['insulin'].dtypes == 'object':
        df['insulin'] = df['insulin'].str.replace(r'[\[\]]', '', regex=True).astype(float)
    results, ri_per_hour, figs, axes = report(df, path_ctrl, sim_days=sim_days)
