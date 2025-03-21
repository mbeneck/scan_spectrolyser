import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import os

def plot_scan_fp(df, measurement, **kwargs):
    df.T.drop('Date/Time').plot(y=measurement, xlabel = 'Wavelength (nm)', ylabel= 'Absorbance', **kwargs)

def import_scan_fp(path):
    df = pd.read_csv(path, sep='\t', header=1).drop('Status_0', axis=1)
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    df.set_index('Date/Time', inplace=True)
    return df

def get_fp_files(directory):
    # List all files in the directory ending with .fp
    return [directory + f for f in os.listdir(directory) if f.endswith('.fp')]


def import_all_scan_fp(directory):
    paths = get_fp_files(directory)
    files = [import_scan_fp(path) for path in paths]
    return pd.concat(files).drop_duplicates()

def apply_calibrations(df, calibrations, calibrations_kwargs):
    for calibration in calibrations:
        df = calibration(df, **calibrations_kwargs[calibration])
    return df