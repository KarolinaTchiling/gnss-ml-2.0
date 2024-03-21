import dataExtract as de
import plotly.express as px
import pandas as pd


def map_track():
    """
    Maps the dataset track and signal type (NLOS and LOS).
    :return:None
    """
    de.extract_raw('smartLoc_data/Berlin_PotsdamerPlatz/RXM-RAWX.csv',
                   'proccessed_data/PotsdamerPlatz-coordinates.csv',
                   [2, 4, 33])  # long, lat, NLOS label

    df = pd.read_csv('proccessed_data/PotsdamerPlatz-coordinates.csv')
    df.columns = ['Longitude', 'Latitude', 'State']
    df['Longitude'] = df['Longitude'].astype(float)
    df['Latitude'] = df['Latitude'].astype(float)
    df['State'] = df['State'].astype(int)
    df['Signal'] = de.get_nlos_label_list('proccessed_data/PotsdamerPlatz-coordinates.csv', 2)

    # print(df.head())

    fig = px.scatter_mapbox(df, lat='Latitude', lon='Longitude',
                            hover_data='Signal', color='State', color_continuous_scale=['dodgerblue', 'orangered'],
                            title="PotsdamerPlatz, Berlin Track with Labeled NLOS",
                            zoom=15, height=850, mapbox_style="open-street-map")

    fig.update_traces(marker=dict(size=8))
    fig.show()


