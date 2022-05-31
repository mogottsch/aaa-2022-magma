import plotly.express as px
import json
from geojson import Feature, FeatureCollection
import h3
import geopandas as gpd
from shapely.geometry import Polygon

# documentation with explanation of kwargs
# https://plotly.github.io/plotly.py-docs/generated/plotly.express.choropleth_mapbox.html
def plot_choropleth(geo_data, hex_col, color_by_col, **kwargs):
    geojson = h3_to_geojson(geo_data, hex_col)

    if not "mapbox_style" in kwargs:
        kwargs["mapbox_style"] = "carto-positron"

    if not "center" in kwargs:
        raise Exception("Please provide a center point for the map.")

    if not "color_continuous_scale" in kwargs:
        kwargs["color_continuous_scale"] = "Viridis"

    # return geojson
    fig = px.choropleth_mapbox(
        geo_data,
        geojson=geojson,
        locations=hex_col,
        color=color_by_col,
        **kwargs,
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


def h3_to_geojson(df_hex, hex_field):
    list_features = []

    for i, row in df_hex.iterrows():
        # list_features.append(Polygon(h3.h3_to_geo_boundary(row[hex_field])))

        feature = Feature(
            geometry=Polygon(h3.h3_to_geo_boundary(row[hex_field], geo_json=True)),
            id=row[hex_field],
        )
        list_features.append(feature)

    feat_collection = FeatureCollection(list_features)

    return feat_collection
