import os
import pathlib


### File paths ###
ROOT_DIR_PATH = pathlib.Path(__file__).parent.parent.absolute()

DATA_DIR_PATH = os.path.join(ROOT_DIR_PATH, "00_data")
TRIPS_TARFILE_PATH = os.path.join(DATA_DIR_PATH, "leipzig.tar")
UNPACKED_TRIPS_DIR_PATH = os.path.join(DATA_DIR_PATH, "leipzig")

PROCESSED_DATA_DIR_PATH = os.path.join(DATA_DIR_PATH, "processed")
ORIGINAL_DATA_MERGED_PATH = os.path.join(
    PROCESSED_DATA_DIR_PATH, "original_data_merged.pkl"
)
TRIPS_PATH = os.path.join(PROCESSED_DATA_DIR_PATH, "trips.pkl")
RELOCATIONS_PATH = os.path.join(PROCESSED_DATA_DIR_PATH, "relocations.pkl")
MOVEMENTS_PATH = os.path.join(PROCESSED_DATA_DIR_PATH, "movements.pkl")

TRIPS_GROUPED_SPATIAL_PATH = os.path.join(
    PROCESSED_DATA_DIR_PATH, "trips_grouped_spatial.pkl"
)
TRIPS_GROUPED_SPATIO_TEMPORAL_PATH = os.path.join(
    PROCESSED_DATA_DIR_PATH, "trips_grouped_spatio_temporal.pkl"
)
RELOCATIONS_GROUPED_SPATIO_TEMPORAL_PATH = os.path.join(
    PROCESSED_DATA_DIR_PATH, "relocations_grouped_spatio_temporal.pkl"
)
MOVEMENTS_GROUPED_SPATIO_TEMPORAL_PATH = os.path.join(
    PROCESSED_DATA_DIR_PATH, "movements_grouped_spatio_temporal.pkl"
)

AVAILABILITY_PATH = os.path.join(PROCESSED_DATA_DIR_PATH, "availability.pkl")

POIS_PATH = os.path.join(PROCESSED_DATA_DIR_PATH, "pois.pkl")

REPO_DATA_DIR_PATH = os.path.join(DATA_DIR_PATH, "repo_data")
FLEXZONE_GEOJSON_PATH = os.path.join(REPO_DATA_DIR_PATH, "leipzig.geojson")


### Configurations ###
H3_RESOLUTION = 8

N_TIME_INTERVALS = 4
TIME_INTERVAL_LENGTH = 24 / N_TIME_INTERVALS
