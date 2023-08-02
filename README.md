# Bichromatic Rabi control of semiconductor qubits

This repository contains all the scripts, data, and figures associated with our research project. The organization of this repository is described below.

## Directory Structure

- `/`: Contains all the Python scripts used in this project.
- `/data/`: Contains the measurement data. The data are organized in subfolders by date.
    - `2022-06-29/`: Data from measurements taken on July 29, 2022.
    - `2022-07-01/`: Data from measurements taken on August 1, 2022.
    ... (other dates)
    - `/attenuation_lovelace_fridge/`: Attenuation data of the lines of an equivalent fridge setup
    - `/config_freq_rabi.py`: Callibrated Rabi frequencies
- `/utils/`: Python files containing methods, defenitions, and settings
    - `/settings.py`: General plotting settings that apply to all scripts
    - `/delft_tools.py`: Methods, and definitions for the experimental analysis
    - `/settings.py`: Methods, and definitions for the theoretical analysis
- `/Figures/`: Contains the figures generated from the data.
    - `/Figure1/`: Figure 1 of the main text
    - `/Figure2/`: Figure 2 of the main text
    ... (other main Figures)
    - `/FigureS1/`: Figure S1 of the supplementary
    ... (other supplementary Figures)

## Usage

1. **Setting Up Directory**: Ensure that the directory is set correctly in the Python files to properly load the data from the specific date folders.

2. **Running Scripts**: Execute the Python scripts in the main directory to analyze the data and generate figures.

3. **Data Retrieval**: All measurement data are stored in the `data/` folder, organized by the date of the measurement.

4. **Figures**: The resulting figures are stored in the `Figures/` folder.

## Dependencies

Make sure to install the required dependencies listed in the `requirements.txt` file.

## Contact

For any questions or additional information, please contact [Your Name and Email Address].

---

_Last updated: [Date of Last Update]_
