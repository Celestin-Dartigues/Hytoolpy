# -*- coding: utf-8 -*-
"""
Data loading utilities for pumping test analysis.

This module provides a function to load time–drawdown data from
different file formats (.txt, .csv, .xlsx, .xls). It automatically
detects delimiters, cleans column names, and extracts the two
main variables:
    - t : time (s)
    - s : drawdown (m)
"""

import pandas as pd
from detect_delimiter import detect
import csv


def load(file):
    """Load pumping test data (time, drawdown) from various file formats.

    Supported formats:
        - TXT  (delimiter auto-detected)
        - CSV  (delimiter auto-detected)
        - XLSX, XLS (Excel files)

    Parameters
    ----------
    file : str
        Path to the input file.

    Returns
    -------
    t : ndarray
        Array of time values (float, >0).
    s : ndarray
        Array of drawdown values (float).
    """
    possible_t_names = ['t', 'temps', 'time']
    possible_s_names = ['s', 'drawdown', 'rabattement']

    # --- Load file depending on extension ---
    if file.lower().endswith('.txt'):
        # Detect delimiter from the first line
        with open(file, 'r') as f:
            first_line = f.readline()
            sep = detect(first_line)
        df = pd.read_csv(file, sep=sep, engine="python")

    elif file.lower().endswith('.csv'):
        # Detect delimiter using Python's CSV sniffer
        with open(file, 'r', newline='') as f:
            sample = f.read(1024)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample)
            df = pd.read_csv(f, delimiter=dialect.delimiter)

    elif file.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file)

    else:
        raise ValueError("Unsupported file format (only txt, csv, xlsx, xls are allowed).")

    # --- Minimal validation ---
    if df.shape[1] < 2:
        raise ValueError("The file must contain at least two columns.")

    # Normalize column names (lowercase, no spaces)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Try to identify time and drawdown columns
    t_col = next((col for col in df.columns if col in possible_t_names), None)
    s_col = next((col for col in df.columns if col in possible_s_names), None)

    # If not found → take the first two columns
    if t_col is None or s_col is None:
        t_col, s_col = df.columns[:2]

    # Select and rename columns
    df = df[[t_col, s_col]].rename(columns={t_col: 't', s_col: 's'})

    # Convert to numeric values
    df['t'] = pd.to_numeric(df['t'], errors='coerce')
    df['s'] = pd.to_numeric(df['s'], errors='coerce')

    # Keep only valid rows (t > 0 and no NaN)
    df = df[df['t'] > 0].dropna()

    # Extract numpy arrays
    t = df['t'].values
    s = df['s'].values

    return t, s
