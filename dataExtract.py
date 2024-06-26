import csv
import pandas as pd
import numpy as np


def extract_raw(in_filepath, out_filepath, col_list, delimiter):
    """
     Extracts columns from the raw CSV and places them into a new CSV
    :param in_filepath: input csv filepath
    :param out_filepath: output csv filepath
    :param col_list:  list of col indexes
    :return: None
    """

    with open(in_filepath, "r", newline="") as infile:
        reader = csv.reader(infile, delimiter=delimiter)

        with open(out_filepath, "w", newline="") as outfile:
            writer = csv.writer(outfile, delimiter=",")

            for row in reader:
                extracted_row = []
                for col in col_list:
                    extracted_row.append(row[col])
                writer.writerow(extracted_row)

    print("\nExtraction completed. The selected columns have been saved to " + out_filepath)


def add_header_to_csv(file_path, header):
    with open(file_path, 'r') as infile:
        reader = csv.reader(infile, delimiter=",")
        rows = list(reader)

    rows.insert(0, header)

    with open(file_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=",")
        writer.writerows(rows)


def merge_elevation_date(file1, file2, out_fp):
    # Read the CSV files into DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Check if the number of rows are the same
    if df1.shape[0] != df2.shape[0]:
        raise ValueError("The two CSV files do not have the same number of rows.")

    # Combine the DataFrames
    combined_df = pd.concat([df1, df2], axis=1)

    # Write the combined DataFrame to a new CSV file
    combined_df.to_csv(out_fp, index=False)
    print("The CSV files have been combined successfully.")


def merge_csv(file_list, out_filepath):
    dataframes = []
    for file in file_list:
        df = pd.read_csv(file)
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv(out_filepath, index=False)

    print("\nMerge completed. The dataset has been saved to " + out_filepath)


def balance_data(file, outfile):
    df = pd.read_csv(file)
    df = df.dropna()    # Drop rows with any missing data (elevation, azimuth)
    nlos_column = 'NLOS (0 == no, 1 == yes, # == No Information)'

    # Ensure there are enough rows to sample from
    if df[df[nlos_column] == '1'].shape[0] < 50000 or df[df[nlos_column] == '0'].shape[0] < 50000:
        raise ValueError("Not enough data to sample 50,000 rows for each category.")

    # Randomly sample 50,000 rows for NLOS and LOS
    nlos_rows = df[df[nlos_column] == '1'].sample(n=50000, random_state=np.random.RandomState())
    los_rows = df[df[nlos_column] == '0'].sample(n=50000, random_state=np.random.RandomState())

    # Combine the sampled rows into a new DataFrame
    sampled_df = pd.concat([nlos_rows, los_rows])
    sampled_df.to_csv(outfile, index=False)

    print(f"\nBalancing completed. 50,000 NLOS and 50,000 LOS measurements have been randomly selected"
          f" and saved in {outfile}")



def get_coordinate_list(filepath, col, type):
    """
    Extracts a column from csv and places into a list
    :param filepath: path of csv file
    :param col: Columns of interest
    :param type: The data type of stored data
    :return:  list
    """

    alist = []
    with open(filepath, "r", newline="") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            alist.append(row[col])

    alist = alist[1:]
    alist = [type(item) for item in alist]
    return alist


def cal_difference_from_azimuth(heading, azimuth):
    if pd.isna(heading) or pd.isna(azimuth):
        return None
    difference = abs(heading - azimuth)
    if difference > 180:
        difference = 360 - difference
    acute_angle = min(difference, 180 - difference)
    return acute_angle


def get_angles(heading_file, elevation_azimuth_file, outfile):
    raw_df = pd.read_csv(heading_file,  delimiter=';')
    angles_df = pd.read_csv(elevation_azimuth_file,  delimiter=',')

    heading = raw_df[["Heading (0° = East, counterclockwise) - (GT Heading) [rad]"]]
    heading_df = heading.copy()

    heading_df["Heading (0° = East, counterclockwise) - (GT Heading) [degrees]"] = (
        np.degrees(heading_df["Heading (0° = East, counterclockwise) - (GT Heading) [rad]"]))
    heading_df = heading_df.drop(columns=["Heading (0° = East, counterclockwise) - (GT Heading) [rad]"])

    # combine heading, elevation and azimuth
    df = pd.concat([heading_df.reset_index(drop=True), angles_df.reset_index(drop=True)], axis=1)

    # Convert heading angle from [-180 - +180] to [0-360]
    df['Heading (0° = East, counterclockwise) - (GT Heading) [degrees]'] = (
        df['Heading (0° = East, counterclockwise) - (GT Heading) [degrees]'].apply(lambda x: x + 360 if x < 0 else x))

    # Rotate heading angles 90 degrees to the left to move 0 = North
    df["Heading (0° = North, counterclockwise)"] = (
        df["Heading (0° = East, counterclockwise) - (GT Heading) [degrees]"].apply(lambda x: (x - 90) % 360))

    # Convert heading from counterclockwise to clockwise
    df["heading"] = df["Heading (0° = North, counterclockwise)"].apply(lambda x: (360 - x) % 360)

    diff_df = df[["heading", "azimuth", 'elevation']].copy()

    # Apply the difference function to both columns
    diff_df['angle'] = diff_df.apply(lambda row: cal_difference_from_azimuth(row['heading'], row['azimuth']), axis=1)

    # Convert the acute angle to the range 0-90
    diff_df['diff_azimuth'] = diff_df['angle'].apply(lambda x: min(x, 90) if pd.notna(x) else x)
    diff_df = diff_df.drop(columns=["heading", "azimuth", "angle"])

    diff_df.to_csv(outfile, index=False)


def create_mapping_data():
    # extracting the coordinates for mapping -------------------------------------------------------------------------
    extract_raw('smartLoc_data/Berlin_PotsdamerPlatz/RXM-RAWX.csv',
                'mapping_data/PotsdamerPlatz-coordinates.csv',
                [2, 4, 33])  # long, lat, NLOS label

    extract_raw('smartLoc_data/Berlin_Gendarmenmarkt/RXM-RAWX.csv',
                'mapping_data/Gendarmenmarkt-coordinates.csv',
                [2, 4, 33])  # long, lat, NLOS label

    extract_raw('smartLoc_data/Frankfurt_MainTower/RXM-RAWX.csv',
                'mapping_data/MainTower-coordinates.csv',
                [2, 4, 33])  # long, lat, NLOS label

    extract_raw('smartLoc_data/Frankfurt_WestendTower/RXM-RAWX.csv',
                'mapping_data/WestendTower-coordinates.csv',
                [2, 4, 33])  # long, lat, NLOS label

    # combining the berlin datasets
    merge_csv(['mapping_data/PotsdamerPlatz-coordinates.csv',
               'mapping_data/Gendarmenmarkt-coordinates.csv'], 'mapping_data/berlin.csv')

    # combining the frankfurt datasets
    merge_csv(['mapping_data/MainTower-coordinates.csv',
               'mapping_data/WestendTower-coordinates.csv'], 'mapping_data/frankfurt.csv')


def create_ml_data():

    # Extracting the features from each dataset ---------------------------------------------------------------------
    # ------ CNO, pseudorange std, NLOS label
    extract_raw('smartLoc_data/Berlin_PotsdamerPlatz/RXM-RAWX.csv',
                'processed_data/PotsdamerPlatz.csv',
                [28, 29, 33], ";")  # cno, pseudorange std, NLOS label
    extract_raw('smartLoc_data/Berlin_Gendarmenmarkt/RXM-RAWX.csv',
                'processed_data/Gendarmenmarkt.csv',
                [28, 29, 33], ";")  # cno, pseudorange std, NLOS label
    extract_raw('smartLoc_data/Frankfurt_MainTower/RXM-RAWX.csv',
                'processed_data/mainTower.csv',
                [28, 29, 33], ";")  # cno, pseudorange std, NLOS label
    extract_raw('smartLoc_data/Frankfurt_WestendTower/RXM-RAWX.csv',
                'processed_data/westTower.csv',
                [28, 29, 33], ";")  # cno, pseudorange std, NLOS label

    # ------ elevation and difference of azimuth angles

    extract_raw('elevation_data/RXM-RAWX_berlin_potsdamer_platz.csv',
                'processed_data/PotsdamerPlatz_elevation.csv',
                [34, 35], ",")  # elevation, azimuth
    add_header_to_csv('processed_data/PotsdamerPlatz_elevation.csv', ["elevation", "azimuth"])
    get_angles('smartLoc_data/Berlin_PotsdamerPlatz/RXM-RAWX.csv',
               'processed_data/PotsdamerPlatz_elevation.csv',
               'processed_data/PotsdamerPlatz_angles.csv')

    extract_raw('elevation_data/RXM-RAWX_berlin_gendarmenmarkt.csv',
                'processed_data/berlin_gendarmenmarkt_elevation.csv',
                [34, 35], ",")  # elevation, azimuth
    add_header_to_csv('processed_data/berlin_gendarmenmarkt_elevation.csv', ["elevation", "azimuth"])
    get_angles('smartLoc_data/Berlin_Gendarmenmarkt/RXM-RAWX.csv',
               'processed_data/berlin_gendarmenmarkt_elevation.csv',
               'processed_data/berlin_gendarmenmarkt_angles.csv')

    extract_raw('elevation_data/RXM-RAWX_frankfurt1_maintower.csv',
                'processed_data/frankfurt1_maintower_elevation.csv',
                [34, 35], ",")  # elevation, azimuth
    add_header_to_csv('processed_data/frankfurt1_maintower_elevation.csv', ["elevation", "azimuth"])
    get_angles('smartLoc_data/Frankfurt_MainTower/RXM-RAWX.csv',
               'processed_data/frankfurt1_maintower_elevation.csv',
               'processed_data/frankfurt1_maintower_angles.csv')

    extract_raw('elevation_data/RXM-RAWX_frankfurt2_westendtower.csv',
                'processed_data/frankfurt2_westendtower_elevation.csv',
                [34, 35], ",")  # elevation, azimuth
    add_header_to_csv('processed_data/frankfurt2_westendtower_elevation.csv', ["elevation", "azimuth"])
    get_angles('smartLoc_data/Frankfurt_WestendTower/RXM-RAWX.csv',
               'processed_data/frankfurt2_westendtower_elevation.csv',
               'processed_data/frankfurt2_westendtower_angles.csv')

    # combine the cno, pr std, nlos, elevation and diff_azimuth --------------------------------------------------
    merge_elevation_date("processed_data/PotsdamerPlatz_angles.csv",
                         "processed_data/PotsdamerPlatz.csv",
                         "processed_data/PotsdamerPlatz_full.csv")

    merge_elevation_date("processed_data/berlin_gendarmenmarkt_angles.csv",
                         "processed_data/Gendarmenmarkt.csv",
                         "processed_data/Gendarmenmarkt_full.csv")

    merge_elevation_date("processed_data/frankfurt1_maintower_angles.csv",
                         "processed_data/mainTower.csv",
                         "processed_data/mainTower_full.csv")

    merge_elevation_date("processed_data/frankfurt2_westendtower_angles.csv",
                         "processed_data/westTower.csv",
                         "processed_data/westTower_full.csv")

    # # combine the extracted feature subsets into a full subset ----------------------------------------------
    merge_csv(['processed_data/PotsdamerPlatz_full.csv',
               'processed_data/Gendarmenmarkt_full.csv',
               'processed_data/mainTower_full.csv',
               'processed_data/westTower_full.csv'], 'processed_data/merge.csv')

    # randomly select 50,000 NLOS and 50,000 LOS measurements from the full dataset --------------------------
    balance_data('processed_data/merge.csv', 'balanced_data.csv')


if __name__ == '__main__':
    # create_mapping_data()
    create_ml_data()



