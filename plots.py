import dataExtract as de
import plotly.express as px
import pandas as pd


def map_track():
    """
    Maps the dataset track and signal type (NLOS and LOS).
    :return:None
    """
    longitude = de.extract_coordinates('proccessed_data/PotsdamerPlatz-RAWX.csv', 2, float)
    latitude = de.extract_coordinates('proccessed_data/PotsdamerPlatz-RAWX.csv', 3, float)
    state = de.extract_coordinates('proccessed_data/PotsdamerPlatz-RAWX.csv', 5, int)
    signal = de.extract_NLOS('proccessed_data/PotsdamerPlatz-RAWX.csv', 5)
    assert len(longitude) == len(latitude) == len(signal) == len(state)

    df = pd.DataFrame({
        'Longitude': longitude,
        'Latitude': latitude,
        'State': state,
        'Signal': signal,
    })

    # print(df.head())

    fig = px.scatter_mapbox(df, lat='Latitude', lon='Longitude',
                            hover_data='Signal', color='State', color_continuous_scale=['dodgerblue', 'orangered'],
                            title="PotsdamerPlatz, Berlin Track with Labeled NLOS",
                            zoom=15, height=850, mapbox_style="open-street-map")

    fig.update_traces(marker=dict(size=8))
    fig.show()


