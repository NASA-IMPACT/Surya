#!/usr/bin/env python3
"""
python create_ar_csv.py

This script generates CSV index files for Active Region (AR) segmentation datasets.
It scans a directory tree for .pth maskß files (PyTorch tensors) named with a specific timestamp
pattern, and creates a CSV file for each year, listing all possible 12-minute intervals
in that year, the expected file path, the timestamp, and whether the file is present.

Example usage:
    python create_ar_csv.py

Author: [Rohit Lal]
Date: [2025-08-20]
"""

import os
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

def fetch_pth_files(directory, start_year, start_month, end_year, end_month):
    """
    Recursively find all .pth files in a directory tree that match a specific
    timestamp pattern and fall within the specified year/month range.

    Args:
        directory (str or Path): Root directory to search.
        start_year (int): Start year (inclusive).
        start_month (int): Start month (1-12, inclusive).
        end_year (int): End year (inclusive).
        end_month (int): End month (1-12, inclusive).

    Returns:
        list of str: List of file paths (as strings) matching the criteria.
    """
    pattern = re.compile(r"(\d{8})_(\d{4})\.pth")
    matching_files = []

    for filepath in sorted(Path(directory).rglob("*.pth")):
        filename = filepath.name
        match = pattern.match(filename)
        if not match:
            continue
        date_str = match.group(1)
        year = int(date_str[:4])
        month = int(date_str[4:6])

        # Check if file is within the specified year/month range
        if (
            (start_year < year < end_year)
            or (year == start_year and start_month <= month <= end_month)
            or (year == end_year and month <= end_month)
        ):
            matching_files.append(str(filepath))

    return matching_files

def create_csv_index(
    dirpath,
    start_year,
    start_month,
    end_year,
    end_month,
    csv_output,
    all_possible_intervals,
):
    """
    Create a CSV index for AR segmentation .pth files.

    Args:
        dirpath (Path): Directory containing AR .pth files.
        start_year (int): Start year (inclusive).
        start_month (int): Start month (1-12, inclusive).
        end_year (int): End year (inclusive).
        end_month (int): End month (1-12, inclusive).
        csv_output (Path): Output CSV file path.
        all_possible_intervals (list): List of [filepath, timestamp] pairs for all intervals.

    Returns:
        pd.DataFrame: DataFrame containing the index.
    """
    pth_files = fetch_pth_files(dirpath, start_year, start_month, end_year, end_month)
    pth_files_set = set(pth_files)
    records = []

    for filepath, time_val in tqdm(all_possible_intervals, desc="Processing files"):
        filepath_str = str(filepath)
        present = 1 if filepath_str in pth_files_set else 0
        records.append(
            {
                "path": filepath_str,
                "timestep": time_val,  # string, e.g. "2013-01-01 00:00:00"
                "present": present,
            }
        )
    df = pd.DataFrame(records)
    df.to_csv(csv_output, index=False)
    print(f"Index file created at {csv_output}")
    return df

def generate_time_intervals(dirpath, start_year, start_month, end_year, end_month):
    """
    Generate all possible 12-minute intervals between the start and end year/month,
    and construct the expected .pth file path for each interval.

    Args:
        dirpath (Path): Root directory for AR .pth files.
        start_year (int): Start year (inclusive).
        start_month (int): Start month (1-12, inclusive).
        end_year (int): End year (inclusive).
        end_month (int): End month (1-12, inclusive).

    Returns:
        list: List of [Path, str] pairs, where Path is the expected .pth file path,
              and str is the formatted timestamp.
    """
    # Start at the first day of the start month
    start_time = np.datetime64(f"{start_year}-{start_month:02d}-01 00:00:00")
    # End at the last second of the end month
    end_time = (
        pd.to_datetime(f"{end_year}-{end_month:02d}-01")
        + pd.DateOffset(months=1)
        - pd.Timedelta(seconds=1)
    )
    end_time = np.datetime64(end_time)

    # 12-minute intervals
    time_intervals = pd.date_range(start=start_time, end=end_time, freq="12T")
    result = []

    for time in time_intervals:
        date_str = time.strftime("%Y%m%d")
        time_str = time.strftime("%H%M")
        # Path: {dirpath}/{year}_extracted/{year}/{month:02d}/{YYYYMMDD}_{HHMM}.pth
        filename = (
            dirpath
            / f"{time.year}_extracted"
            / f"{time.year}"
            / f"{time.month:02d}"
            / f"{date_str}_{time_str}.pth"
        )
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S")
        result.append([filename, formatted_time])

    return result

def main(start_year, start_month, end_month):
    """
    Main entry point for AR CSV index generation.
    Generates a CSV file for each year in the range 2011-2014 (inclusive).
    """
    end_year = start_year

    cwd = Path(__file__).parent.resolve()
    valid_extracted_path = cwd / "assets" / "surya-bench-ar-segmentation"
    csv_output = cwd / "assets" / "ar_csv_files" / f"ar_{start_year}.csv"
    csv_output.parent.mkdir(parents=True, exist_ok=True)

    all_possible_intervals = generate_time_intervals(
        valid_extracted_path, start_year, start_month, end_year, end_month
    )

    _ = create_csv_index(
        valid_extracted_path,
        start_year,
        start_month,
        end_year,
        end_month,
        csv_output,
        all_possible_intervals,
    )

if __name__ == "__main__":
    main(start_year=2011, start_month=1, end_month=12)
