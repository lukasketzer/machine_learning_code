################### Baseline Model for Project 2

import csv
import numpy as np

# read file
with open(r'..\..\data\data.csv', mode='r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    first_row_flag = True
    # read data into np.array as float
    column = 3
    weight_column = []
    for row in reader:
        if(first_row_flag):
            first_row_flag = False
            continue
        weight_column.append(float(row[column]))
        
    # calculate baseline == mean
    mean = np.mean(np.array(weight_column))
    print(f"the baseline for column {column} without (!) a testset is:")
    print(mean)
    
    function = lambda _: mean