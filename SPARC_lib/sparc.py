import pandas as pd
import unittest
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
        if row[field] == '1':
            best = i + 1
    return best


def dollars2float(dollars: str) -> float:
    return float(dollars[1:].replace(",", ""))


def unzip(tuple_values):
    # From https://appdividend.com/2020/10/19/how-to-unzip-list-of-tuples-in-python/#:~:text=%20How%20to%20Unzip%20List%20of%20Tuples%20in,zip...%204%202%3A%20Using%20List%20Comprehension%20More%20
    return tuple(zip(*tuple_values))


class Tests(unittest.TestCase):
    def test_dollars(self):
        self.assertEqual(12345.6, dollars2float("$12345.60"))

    def test_unzip(self):
        self.assertEqual((('Moe', 'Larry', 'Curly'), (1, 2, 3)), unzip([('Moe', 1), ('Larry', 2), ('Curly', 3)]))
