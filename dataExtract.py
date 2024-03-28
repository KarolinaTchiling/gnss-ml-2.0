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

    print("\nExtraction completed. The selected columns have been saved to " + out_filepath + ".\n")


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




