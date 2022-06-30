import os
import pathlib


### File paths ###
ROOT_DIR_PATH = pathlib.Path(__file__).parent.parent.absolute()


DATA_DIR_PATH = os.path.join(ROOT_DIR_PATH, "00_data")
REPO_DATA_DIR_PATH = os.path.join(DATA_DIR_PATH, "repo_data")
TRIPS_TARFILE_PATH = os.path.join(DATA_DIR_PATH, "leipzig.tar")
UNPACKED_TRIPS_DIR_PATH = os.path.join(DATA_DIR_PATH, "leipzig")

PROCESSED_DATA_DIR_PATH = os.path.join(DATA_DIR_PATH, "processed")
ORIGINAL_DATA_MERGED_PATH = os.path.join(
    PROCESSED_DATA_DIR_PATH, "original_data_merged.parquet"
)
TRIPS_PATH = os.path.join(PROCESSED_DATA_DIR_PATH, "trips.parquet")
RELOCATIONS_PATH = os.path.join(PROCESSED_DATA_DIR_PATH, "relocations.parquet")
MOVEMENTS_PATH = os.path.join(PROCESSED_DATA_DIR_PATH, "movements.parquet")

TRIPS_GROUPED_SPATIO_TEMPORAL_PATH = os.path.join(
    PROCESSED_DATA_DIR_PATH, "trips_grouped_spatio_temporal.parquet"
)
RELOCATIONS_GROUPED_SPATIO_TEMPORAL_PATH = os.path.join(
    PROCESSED_DATA_DIR_PATH, "relocations_grouped_spatio_temporal.parquet"
)
MOVEMENTS_GROUPED_SPATIO_TEMPORAL_PATH = os.path.join(
    PROCESSED_DATA_DIR_PATH, "movements_grouped_spatio_temporal.parquet"
)

WEATHER_AGGR_TEMPORAL_PATH = os.path.join(REPO_DATA_DIR_PATH, "weather_aggr.parquet")

HEXAGONS_WITH_LAND_USE_PATH = os.path.join(
    REPO_DATA_DIR_PATH, "hexagons_land_use_columns.parquet"
)

AVAILABILITY_PATH = os.path.join(PROCESSED_DATA_DIR_PATH, "availability.parquet")

POIS_PATH = os.path.join(PROCESSED_DATA_DIR_PATH, "pois.parquet")
HEXAGON_WITH_POIS_PATH = os.path.join(
    PROCESSED_DATA_DIR_PATH, "hexagons_with_pois.parquet"
)

MODEL_DATA_PATH = os.path.join(PROCESSED_DATA_DIR_PATH, "model_data.feather")
REDUCED_MODEL_DATA_PATH = os.path.join(
    PROCESSED_DATA_DIR_PATH, "reduced_model_data.feather"
)
MODEL_DATA_DIR_PATH = os.path.join(PROCESSED_DATA_DIR_PATH, "model_data")

SVM_FIRST_STAGE_RESULTS_PATH = os.path.join(
    REPO_DATA_DIR_PATH, "svm_first_stage_results.parquet"
)
SVM_SECOND_STAGE_RESULTS_PATH = os.path.join(
    REPO_DATA_DIR_PATH, "svm_second_stage_results.parquet"
)

NN_FIRST_STAGE_DEMAND_RESULTS_PATH = os.path.join(
    REPO_DATA_DIR_PATH, "nn_first_stage_demand_results.parquet"
)
NN_SECOND_STAGE_DEMAND_RESULTS_PATH = os.path.join(
    REPO_DATA_DIR_PATH, "nn_second_stage_demand_results.parquet"
)
NN_THIRD_STAGE_DEMAND_RESULTS_PATH = os.path.join(
    REPO_DATA_DIR_PATH, "nn_third_stage_demand_results.parquet"
)
NN_FOURTH_STAGE_DEMAND_RESULTS_PATH = os.path.join(
    REPO_DATA_DIR_PATH, "nn_fourth_stage_demand_results.parquet"
)

NN_FIRST_STAGE_AVAILABILITY_RESULTS_PATH = os.path.join(
    REPO_DATA_DIR_PATH, "nn_first_stage_availability_results.parquet"
)
NN_SECOND_STAGE_AVAILABILITY_RESULTS_PATH = os.path.join(
    REPO_DATA_DIR_PATH, "nn_second_stage_availability_results.parquet"
)
NN_THIRD_STAGE_AVAILABILITY_RESULTS_PATH = os.path.join(
    REPO_DATA_DIR_PATH, "nn_third_stage_availability_results.parquet"
)
NN_FOURTH_STAGE_AVAILABILITY_RESULTS_PATH = os.path.join(
    REPO_DATA_DIR_PATH, "nn_fourth_stage_availability_results.parquet"
)

FINAL_MODEL_RESULTS_PATH = os.path.join(
    REPO_DATA_DIR_PATH, "final_model_results.parquet"
)

FLEXZONE_GEOJSON_PATH = os.path.join(REPO_DATA_DIR_PATH, "leipzig.geojson")


### Configurations ###
H3_RESOLUTION = 8
TUNE_H3_RESOLUTION = 8
ORIGIN_DESTINATION_H3_RESOLUTION = 7
CALC_H3_RESOLUTIONS = [7, 8, 9]
PREDICTIVE_H3_RESOLUTIONS = [7, 8]

N_TIME_INTERVALS = 4
TIME_INTERVAL_LENGTH = 24 / N_TIME_INTERVALS
TUNE_TIME_INTERVAL_LENGTH = 6
ORIGIN_DESTINATION_TIME_INTERVAL_LENGTH = 24
CALC_TIME_INTERVAL_LENGTHS = [1, 2, 6, 24]

# defined as (h3, time)
ADDITIONAL_PREDICTIVE_RESOLUTIONS = [(9, 24)]
