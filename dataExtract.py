import csv
import pandas as pd
import numpy as np

def extract(filepath):

    with open(filepath, "r", newline="") as infile:
        reader = csv.reader(infile, delimiter=";")

        with open('proccessed_data/PotsdamerPlatz-RAWX.csv', "w", newline="") as outfile:
            writer = csv.writer(outfile, delimiter=",")

            for row in reader:
                # row[0] = GPSweek, row[1] = GPSSeconds, row[28] = c/no, row[33] = NLOS information
                extract_row = [row[0], row[1], row[28], row[33]]
                writer.writerow(extract_row)

    print("\nExtraction completed. The selected columns have been saved to 'proccessed_data/PotsdamerPlatz-RAWX.csv'.\n")


