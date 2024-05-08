import csv
import pandas as pd
import numpy as np


def extract_raw(in_filepath, out_filepath, col_list):
    """
     Extracts columns from the raw CSV and places them into a new CSV
    :param in_filepath: input csv filepath
    :param out_filepath: output csv filepath
    :param col_list:  list of col indexes
    :return: None
    """

    with open(in_filepath, "r", newline="") as infile:
        reader = csv.reader(infile, delimiter=";")

        with open(out_filepath, "w", newline="") as outfile:
            writer = csv.writer(outfile, delimiter=",")

            for row in reader:
                extracted_row = []
                for col in col_list:
                    extracted_row.append(row[col])
                writer.writerow(extracted_row)

    print("\nExtraction completed. The selected columns have been saved to " + out_filepath)


def merge_csv(file_list):
    dataframes = []
    for file in file_list:
        df = pd.read_csv(file)
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv('processed_data/merged.csv', index=False)

    print("\nMerge completed. The selected columns have been saved to 'processed_data/merged.csv.")


def balance_data(file):
    df = pd.read_csv(file)
    nlos_column = 'NLOS (0 == no, 1 == yes, # == No Information)'
    nlos_rows = df[df[nlos_column] == '1'].sample(n=50000, random_state=np.random.RandomState())
    los_rows = df[df[nlos_column] == '0'].sample(n=50000, random_state=np.random.RandomState())
    sampled_df = pd.concat([nlos_rows, los_rows])
    sampled_df.to_csv('processed_data/balanced_data.csv', index=False)

    print("\nBalancing completed. 50,000 NLOS and 50,000 LOS measurements have been randomly selected"
          "and saved in 'processed_data/balanced_data.csv.")


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
            elif row[col] == '#' :
                alist.append('N/A')
    return alist


if __name__ == '__main__':
    # extracts only the necessary columns from the raw data and creates a new csv
    extract_raw('smartLoc_data/Berlin_PotsdamerPlatz/RXM-RAWX.csv',
                'processed_data/PotsdamerPlatz.csv',
                [28, 29, 33])    # cno, pseudorange std, NLOS label

    extract_raw('smartLoc_data/Berlin_Gendarmenmarkt/RXM-RAWX.csv',
                'processed_data/Gendarmenmarkt.csv',
                [28, 29, 33])    # cno, pseudorange std, NLOS label

    extract_raw('smartLoc_data/Frankfurt_MainTower/RXM-RAWX.csv',
                'processed_data/mainTower.csv',
                [28, 29, 33])    # cno, pseudorange std, NLOS label

    extract_raw('smartLoc_data/Frankfurt_WestendTower/RXM-RAWX.csv',
                'processed_data/westTower.csv',
                [28, 29, 33])    # cno, pseudorange std, NLOS label

    # combine the extracted subsets into a full subset
    merge_csv(['processed_data/PotsdamerPlatz.csv',
               'processed_data/Gendarmenmarkt.csv',
               'processed_data/mainTower.csv',
               'processed_data/westTower.csv'])

    # randomly select 50,000 NLOS and 50,000 LOS measurements from the full dataset
    balance_data('processed_data/merged.csv')
