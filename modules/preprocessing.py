from time import time
from pandas import DataFrame, Series
import pandas as pd
import h3


def aggregate(
    trips: DataFrame, h3_resolution: int, timeinterval_length: int, start_and_end=False
) -> Series:
    """
    Aggregates trips by H3 resolution and time interval.
    """
    trips_c = trips.copy()
    trips_c["datetime_start_floored"] = trips_c["datetime_start"].dt.floor(
        f"{timeinterval_length}H"
    )
    if start_and_end:
        trips_c["datetime_end_floored"] = trips_c["datetime_end"].dt.floor(
            f"{timeinterval_length}H"
        )

    trips_c["end_hex_id"] = trips.apply(
        lambda row: h3.geo_to_h3(row["lat_end"], row["lng_end"], h3_resolution), axis=1
    )
    trips_c["start_hex_id"] = trips.apply(
        lambda row: h3.geo_to_h3(row["lat_start"], row["lng_start"], h3_resolution),
        axis=1,
    )

    groupby_cols = (
        ["datetime_start_floored", "datetime_end_floored", "start_hex_id", "end_hex_id"]
        if start_and_end
        else ["datetime_start_floored", "start_hex_id", "end_hex_id"]
    )

    return trips_c.groupby(groupby_cols).size().to_frame("demand")


def calculate_availability(
    movements_grouped: DataFrame,
    first_locations: DataFrame,
    last_locations: DataFrame,
    h3_resolution: int,
    time_interval_length: int,
):
    starting_movements, ending_movements = _prepare_movements(movements_grouped)
    first_locations, last_locations = _prepare_locations(
        first_locations, last_locations, h3_resolution, time_interval_length
    )
    full_index = _get_full_index(
        starting_movements,
        ending_movements,
        first_locations,
        last_locations,
        time_interval_length,
    )

    starting_movements = starting_movements.reindex(full_index, fill_value=0)
    ending_movements = ending_movements.reindex(full_index, fill_value=0)
    first_locations = first_locations.reindex(full_index, fill_value=0)
    last_locations = last_locations.reindex(full_index, fill_value=0)

    net = (
        (ending_movements - starting_movements + first_locations - last_locations)
        .to_frame("n_bikes")
        .reset_index()
        .rename(columns={"level_0": "hex_id"})
        .sort_values("datetime")
        .groupby(["datetime", "hex_id"])
        .sum()
    )
    net.index.names = ["datetime", "hex_id"]
    availability = net.groupby(level=-1).cumsum()

    return availability


def _prepare_movements(movements_grouped: DataFrame):
    starting_movements = movements_grouped.groupby(
        ["datetime_start_floored", "start_hex_id"]
    )["n_bikes"].sum()
    ending_movements = movements_grouped.groupby(
        ["datetime_end_floored", "end_hex_id"]
    )["n_bikes"].sum()
    starting_movements.index.names = ["datetime", "hex_id"]
    ending_movements.index.names = ["datetime", "hex_id"]
    return starting_movements, ending_movements


def _prepare_locations(
    first_locations: DataFrame,
    last_locations: DataFrame,
    h3_resolution: int,
    time_interval_length: int,
):
    first_locations = first_locations.copy()
    last_locations = last_locations.copy()
    get_hex_id = lambda row: h3.geo_to_h3(row["lat"], row["lng"], h3_resolution)
    first_locations["hex_id"] = first_locations.apply(get_hex_id, axis=1)
    last_locations["hex_id"] = last_locations.apply(get_hex_id, axis=1)

    first_locations["datetime"] = first_locations["datetime"].dt.floor(
        f"{time_interval_length}H"
    )
    last_locations["datetime"] = last_locations["datetime"].dt.floor(
        f"{time_interval_length}H"
    )

    first_locations = first_locations.groupby(["datetime", "hex_id"])[
        "b_number"
    ].count()
    last_locations = last_locations.groupby(["datetime", "hex_id"])["b_number"].count()

    return first_locations, last_locations


def _get_full_index(
    starting_movements: DataFrame,
    ending_movements: DataFrame,
    first_locations: DataFrame,
    last_locations: DataFrame,
    time_interval_length: int,
):
    all_hex_ids = (
        starting_movements.index.get_level_values("hex_id")
        .unique()
        .union(ending_movements.index.get_level_values("hex_id").unique())
        .union(first_locations.index.get_level_values("hex_id").unique())
        .union(last_locations.index.get_level_values("hex_id").unique())
    )
    daterange = pd.date_range(
        start="2019-01-01", end="2019-12-31", freq=f"{time_interval_length}H"
    )

    full_index = pd.MultiIndex.from_product(
        [daterange, all_hex_ids], names=["datetime", "hex_id"]
    )

    return full_index
