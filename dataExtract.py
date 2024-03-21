import csv
import pandas as pd
import numpy as np


def extract_raw(filepath):

    with open(filepath, "r", newline="") as infile:
        reader = csv.reader(infile, delimiter=";")

        with open('proccessed_data/PotsdamerPlatz-RAWX.csv', "w", newline="") as outfile:
            writer = csv.writer(outfile, delimiter=",")

            for row in reader:
                # 0 = GPSweek, 1 = GPSSeconds,
                # 2 = GT Longitude [deg], 4 = Latitude (GT Lat) [deg]
                # 28  = c/no, 33 = NLOS information
                extract_row = [row[0], row[1], row[2], row[4], row[28], row[33]]
                writer.writerow(extract_row)

    print("\nExtraction completed. The selected columns have been saved to 'proccessed_data/PotsdamerPlatz-RAWX.csv'.\n")


def extract_coordinates(filepath, col, type):

    alist = []
    with open(filepath, "r", newline="") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            alist.append(row[col])

    alist = alist[1:]
    alist = [type(item) for item in alist]
    return alist


def extract_NLOS(filepath, col):
    alist = []
    with open(filepath, "r", newline="") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            if row[col] == '0':
                alist.append('LOS')
            elif row[col] == '1':
                alist.append('NLOS')
    return alist



