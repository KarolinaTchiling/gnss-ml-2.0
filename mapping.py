import plotly.express as px
import pandas as pd
import csv


def get_nlos_label_list(filepath, col):
    """
    Extracts the NLOS labels from a csv and map them to the label names, store in a list
    :param filepath: path of CSV
    :param col: The location of NLOS labels within the csv
    :return: List of signal labels
    """
    alist = []
    with open(filepath, "r", newline="") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            if row[col] == '0':
                alist.append('LOS')
            elif row[col] == '1':
                alist.append('NLOS')
            elif row[col] == '#':
                alist.append('N/A')
    return alist


def get_state_label_list(filepath, col):
    """
    Extracts the NLOS labels from a csv and map them to the label names, store in a list
    :param filepath: path of CSV
    :param col: The location of NLOS labels within the csv
    :return: List of signal labels
    """
    alist = []
    with open(filepath, "r", newline="") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            if row[col] == '0':
                alist.append(1)
            elif row[col] == '1':
                alist.append(3)
            elif row[col] == '#':
                alist.append(2)
    return alist


def map_track(file, title):
    """
    Maps the dataset track and signal type (NLOS and LOS).
    :return:None
    """

    df = pd.read_csv(file)
    df.columns = ['Longitude', 'Latitude', 'State']
    df['Longitude'] = df['Longitude'].astype(float)
    df['Latitude'] = df['Latitude'].astype(float)
    df['State'] = get_state_label_list(file, 2)
    df['Signal'] = get_nlos_label_list(file, 2)

    print(df.head(20))

    fig = px.scatter_mapbox(df, lat='Latitude', lon='Longitude',
                            hover_data='Signal', color='State', color_continuous_scale=['dodgerblue', 'white', 'orangered'],
                            title=title,
                            zoom=14, height=870, mapbox_style="carto-positron")

    fig.show()


if __name__ == '__main__':
    map_track('mapping_data/berlin.csv',
              "PotsdamerPlatz and Gendarmenmarkt, Berlin Tracks with Labeled NLOS")
    map_track('mapping_data/frankfurt.csv',
              "Main and West Tower, Frankfurt Tracks with Labeled NLOS")

