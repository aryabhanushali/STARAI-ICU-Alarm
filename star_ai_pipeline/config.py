# config.py

from pathlib import Path
import os

DATA_DIR = Path(os.environ.get("STAR_DATA_DIR", str(Path(__file__).parent.parent)))

TRAIN_CSV     = DATA_DIR / "merged_stage_train.csv.gz"
TEST_CSV      = DATA_DIR / "merged_stage_test.csv.gz"
TRAIN_LABELS  = DATA_DIR / "train_labels.json"
TEST_IDS      = DATA_DIR / "test.json"
GENERAL_TABLE = DATA_DIR / "general_table.csv"

MODELS_DIR    = Path(__file__).parent / "models"
RESULTS_DIR   = Path(__file__).parent / "results"
LOGS_DIR      = Path(__file__).parent / "logs"
SUBMISSION_PATH = Path(__file__).parent / "submission.csv"

FEATURES_TRAIN = RESULTS_DIR / "features_train.parquet"
FEATURES_TEST  = RESULTS_DIR / "features_test.parquet"
LOG_FILE       = LOGS_DIR / "pipeline.log"

PID_COL   = "patientid"
LABEL_COL = "mortality"

TRAIN_TIME = "datetime"
TEST_TIME  = "datetime"

# 18 clinical variables in the CSVs
FEATURE_COLS = [
    "vm1",   # heart rate
    "vm3",   # systolic BP
    "vm4",   # diastolic BP
    "vm5",   # MAP
    "vm13",  # temperature
    "vm20",  # SpO2
    "vm28",  # respiratory rate
    "vm62",  # mechanical ventilation flag
    "vm136", # GCS
    "vm146", # FiO2
    "vm172", # lactate
    "vm174", # creatinine
    "vm176", # bilirubin
    "pm41",  # norepinephrine
    "pm42",  # epinephrine
    "pm43",  # vasopressin
    "pm44",  # dobutamine
    "pm87",  # milrinone
]

MAP_COL          = "vm5"
SPO2_COL         = "vm20"
MV_COL           = "vm62"
VASOPRESSOR_COLS = ["pm41", "pm42", "pm43"]
DELTA_COLS       = ["vm5", "vm20"]

MAP_LOW  = 65.0   # shock threshold (mmHg)
SPO2_LOW = 90.0   # hypoxemia threshold (%)

AGG_STATS = ["mean", "std", "min", "max", "median", "p25", "p75", "first", "last", "trend"]

MIN_TREND_OBS = 3
CHUNK_SIZE    = 500_000  # read in chunks to avoid memory issues

SEED       = 42
N_FOLDS    = 5
LR         = 0.05
N_TREES    = 1000
EARLY_STOP = 50
N_LEAVES   = 63  # do NOT change: validated at this value

SUBSAMPLE      = 0.8
COLSAMPLE_TREE = 0.8
BAGGING_FREQ   = 5

N_TEST = 10_173  # sanity check on submission size
