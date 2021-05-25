import pandas as pd
import openpyxl

enrollment_map_fields = ('1st Sem', '2nd Sem', '3rd Sem', '4th Sem', '5th Sem', '6th Sem', '7th Sem', '8th Sem', 'Grad in 4', 'Grad in 5', 'Grad in 6')


def enrollment_map_depths(filename):
    depths = {}
    data = pd.read_excel(filename, dtype=str)
    for index, row in data.iterrows():
        student = row['id_num']
        depths[student] = map_depth_for(row)
    return depths


def map_depth_for(row):
    best = 0
    for i, field in enumerate(enrollment_map_fields):
        print(field, row[field])
        if row[field] == '1':
            best = i + 1
    return best


