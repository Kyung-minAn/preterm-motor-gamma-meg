"""Example configuration. Copy this file to config.py (or point the environment variable PRETERM_MEG_CONFIG to
a copy kept elsewhere) and edit the paths. The data themselves are not part of this repository."""
from pathlib import Path

# Root folder of the Brainstorm database; the 'file' column of TFR_FILES is relative to this folder
BRAINSTORM_ROOT = Path("/path/to/brainstorm_db")

# CSV with one row per participant and hemisphere: ID, Group (FT or PT), side (contra or ipsi), file
# ('file' = Brainstorm time-frequency file, timefreq_morlet_*_ersd_time.mat, containing both precentral scouts)
TFR_FILES = Path("/path/to/tfr_files.csv")

# Participant-level spreadsheets (see README.md for the expected columns)
BEHAVIOUR_XLSX = Path("/path/to/behaviour.xlsx")
BACKGROUND_XLSX = Path("/path/to/background.xlsx")
CLINICAL_XLSX = Path("/path/to/clinical.xlsx")      # optional: set to None if not available

# FreeSurfer subjects directory and folder-name pattern of each participant ({id} = participant ID)
FREESURFER_DIR = Path("/path/to/freesurfer_subjects")
FREESURFER_SUBJECT_PATTERN = {"FT": "sub-{id}*", "PT": "sub-{id}*"}

# Where tables, intermediate files and figures are written (keep it outside the repository)
OUTPUT_DIR = Path("/path/to/results")
