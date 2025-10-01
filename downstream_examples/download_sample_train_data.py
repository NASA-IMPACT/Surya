import os
import re
from pathlib import Path
from huggingface_hub import snapshot_download, list_repo_files
from tqdm import tqdm
import logging
import pandas as pd
import numpy as np

import tarfile

REPO_ID = "nasa-ibm-ai4science/SDO_training"
VALID_SPLITTED_TAR_DIR_NAME = "validate_splitted_tars"
TEST_SPLITTED_TAR_DIR_NAME = "test_splitted_tars"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def combine_tar_parts(path: Path, prefix: str) -> Path:
    """
    Combine the different tar parts into a single tar file
    Args:
        path: Path to the tar parts
        prefix: name for the output tar file
    Returns:
        Path to the output tar file
    """
    parts = sorted(Path(path).glob("*.tar.part_*"))
    output_file = f"{prefix}.tar"
    output_file = path / output_file

    logger.info(f"Combining tars at path {path}")
    logger.info(f"Saving tars at path {output_file}")

    with open(output_file, "wb") as out_f:
        for part in parts:
            logger.info(f"Reading part file {part.name}")
            with open(part, "rb") as part_f:
                while True:
                    chunk = part_f.read(1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)

    logger.info("Done reading parts")
    return output_file


def verify_tar_file(tar_file: Path) -> bool:
    """
    Verify if it is a tar file
    Args:
        tar_file: Path to the tar file
    Returns:
        True if the tar file is valid, False otherwise
    """
    try:
        with tarfile.open(tar_file, "r") as tar:
            members = tar.getnames()
            logger.info(f"Tar file {tar_file} verification success")
            logger.info(f"Contains {len(members)} files/directories")

            if members:
                logger.info("5 samples")
                for name in members[:5]:
                    logger.info(f"{name}")

        return True

    except Exception as e:
        logger.error(f"Tar file verification failed: {e}")
        return False


def list_files(repo_id: str) -> list:
    """
    List the files in the huggingface repository
    Args:
        repo_id: ID of the repository
    Returns:
        List of files in the repository
    """
    try:
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        return files
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return []


def download_dataset(
    download_dir: Path,
    repo_id: str,
    allow_patterns: list = [],
    ignore_patterns: list = [],
    resume_download: bool = True,
) -> Path:
    """
    Download the dataset from the huggingface repository
    Args:
        download_dir: Path to the download directory
        repo_id: ID of the repository
        allow_patterns: List of patterns to allow
        ignore_patterns: List of patterns to ignore
        resume_download: Whether to resume the download
    Returns:
        Path to the downloaded dataset
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using path: {download_dir}")

    downloaded_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=download_dir,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        resume_download=resume_download,
        token=None,
    )

    logger.info(f"Dataset downloaded successfully at: {downloaded_path}")

    return Path(downloaded_path)


def extract_tar(
    tar_file_path: Path,
    extract_path: Path,
) -> Path:
    """
    Extract the tar file
    Args:
        tar_file_path: Path to the tar file
        extract_path: Path to the parent directory of the extracted file
    Returns:
        Path to the extracted directory
    """
    extract_path.mkdir(parents=True, exist_ok=True)
    tar_ = tar_file_path
    logger.info(f"Extracting {tar_}")
    is_valid_tar = verify_tar_file(tar_)
    if is_valid_tar:
        with tarfile.open(tar_, "r") as tar:
            tar.extractall(path=str(extract_path))
    else:
        logger.error(f"Cannot extract {tar_}")

    return extract_path / tar_file_path.stem


def fetch_nc_files(directory, start_year, start_month, end_year, end_month):
    # Define the regex pattern for matching the filenames
    pattern = re.compile(r"(\d{8})_(\d{4})\.nc")

    # List to store matching files
    matching_files = []

    # Iterate over files in the specified directory
    for filepath in sorted(Path(directory).rglob("*.nc")):
        filename = filepath.name
        match = pattern.match(filename)
        # if match:
        if int(match.group(1)[4:6]) >= start_month:
            # Extract the date part from the filename (YYYYMMDD)
            date_str = match.group(1)
            if start_month and end_month:
                year = int(date_str[:4])  # Get the year from the date string
                month = int(date_str[4:6])  # Get the month from the date string

                # Check if the file is within the given year and month range
                if (
                    (start_year < year < end_year)
                    or (year == start_year and month >= start_month and month <= end_month)
                    or (year == end_year and month <= end_month)
                ):
                    matching_files.append(str(filepath))

            else:
                year = int(date_str[:4])  # Get the year from the date string
                # Check if the year is within the given range
                if start_year <= year <= end_year:
                    matching_files.append(filepath)

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
    Create a csv index for the nc files
    Args:
        dirpath: Path to the directory
        start_year: Start year
        start_month: Start month
        end_year: End year
        end_month: End month
        csv_output: Path to the csv output file
        all_possible_intervals: List of all possible intervals
    Returns:
        None
    """
    #    nc_files = sorted(Path(dirpath).rglob("*.nc"))
    nc_files = fetch_nc_files(dirpath, start_year, start_month, end_year, end_month)

    nc_files = set(nc_files)
    records = []

    # Wrap the loop with tqdm for the progress bar
    for filepath, time_val in tqdm(all_possible_intervals, desc="Processing files"):
        if filepath in nc_files:
            present = 1
        else:
            present = 0

        records.append(
            {
                "path": filepath,
                "timestep": time_val,  # numpy.datetime64[ns]
                "present": present,
            }
        )
    # Create a DataFrame from the records and save it as a CSV file
    df = pd.DataFrame(records)
    df.reset_index()
    # df.to_csv(csv_output)
    # logger.info(f"Index file created at {csv_output}")
    return df


def generate_time_intervals(dirpath, start_year, start_month, end_year, end_month):
    """
    Generate the time intervals
    Args:
        dirpath: Path to the directory
        start_year: Start year
        start_month: Start month
        end_year: End year
        end_month: End month
    Returns:
        List of time intervals
    """
    # Define the start time using numpy.datetime64
    start_time = np.datetime64(f"{start_year}-{start_month:02d}-01 00:00:00")

    # Use pandas to get the last day of the end month
    end_time = (
        pd.to_datetime(f"{end_year}-{end_month:02d}-01")
        + pd.DateOffset(months=1)
        - pd.Timedelta(days=1)
    )
    end_time = np.datetime64(end_time)  # Convert it back to numpy.datetime64

    # Create time intervals (12-minute increments)
    time_intervals = pd.date_range(start=start_time, end=end_time, freq="12T")

    # Initialize a list to store the file paths and corresponding datetime
    result = []

    # Iterate over the generated time intervals
    for time in time_intervals:
        # Generate the filename based on the time
        date_str = time.strftime("%Y%m%d")  # Extract date as YYYYMMDD
        time_str = time.strftime("%H%M")  # Extract time as HHMM

        # filename = os.path.join(
        #     dirpath, f"{time.year}", f"{time.month:02d}", f"{date_str}_{time_str}.nc"
        # )
        filename = os.path.join(dirpath, f"{date_str}_{time_str}.nc")
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S")

        result.append([filename, formatted_time])

    return result


def download_and_process(
    tar_download_dir: Path,
    extract_path: Path,
    splitted_tar_dir_name: str,
    prefix: str,
    repo_id: str = REPO_ID,
    allow_patterns: list = [],
):
    start_year = 2011
    start_month = 1
    end_year = 2011
    end_month = 2

    # Download tars
    valid_downloaded_data_path = download_dataset(
        download_dir=tar_download_dir,
        repo_id=repo_id,
        allow_patterns=allow_patterns,
    )

    # combine multiple tars into one
    valid_tar_file_path = combine_tar_parts(
        path=valid_downloaded_data_path / splitted_tar_dir_name, prefix=prefix
    )

    # Extract files from tars
    valid_extracted_path = extract_tar(tar_file_path=valid_tar_file_path, extract_path=extract_path)

    # create csv
    logger.info(valid_extracted_path)

    csv_output = Path().cwd() / "assets" / f"sdo_{prefix}.csv"
    all_possible_intervals = generate_time_intervals(
        valid_extracted_path, start_year, start_month, end_year, end_month
    )

    df = create_csv_index(
        valid_extracted_path,
        start_year,
        start_month,
        end_year,
        end_month,
        csv_output,
        all_possible_intervals,
    )

    downstream_tasks = [
        "ar_segmentation",
        "euv_spectra_prediction",
        "solar_flare_forcasting",
        "solar_wind_forcasting",
    ]

    for task in downstream_tasks:
        csv_save_path = Path(__file__).parent / task / "assets"
        csv_save_path.mkdir(parents=True, exist_ok=True)
        csv_file_name = f"sdo_{prefix}.csv"

        df.to_csv(csv_save_path / csv_file_name)
        logger.info(f"Saved csv at {csv_save_path / csv_file_name}")


def main():
    files = list_files(REPO_ID)
    logger.info(files)

    valid_allow_patterns = ["**/val.tar.part_*"]
    test_allow_patterns = ["**/test.tar.part_*"]

    data_path = Path(__file__).parent / "common_data"
    data_path.mkdir(parents=True, exist_ok=True)

    tar_download_dir = data_path / "tars"

    download_and_process(
        tar_download_dir=tar_download_dir,
        extract_path=data_path,
        splitted_tar_dir_name=VALID_SPLITTED_TAR_DIR_NAME,
        prefix="validate",
        repo_id=REPO_ID,
        allow_patterns=valid_allow_patterns,
    )

    download_and_process(
        tar_download_dir=tar_download_dir,
        extract_path=data_path,
        splitted_tar_dir_name=TEST_SPLITTED_TAR_DIR_NAME,
        prefix="test",
        repo_id=REPO_ID,
        allow_patterns=test_allow_patterns,
    )


if __name__ == "__main__":
    main()
