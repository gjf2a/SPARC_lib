import pandas as pd
import unittest
import openpyxl
from functools import total_ordering


def enrollment_map_depths(filename):
    depths = {}
    data = pd.read_excel(filename, dtype=str)
    for index, row in data.iterrows():
        student = row['id_num']
        depths[student] = map_depth_for(row)
    return depths


def map_depth_for(row):
    sem_fields = ('1st Sem', '2nd Sem', '3rd Sem', '4th Sem', '5th Sem', '6th Sem', '7th Sem', '8th Sem')
    grad_fields = ('Grad in 4', 'Grad in 5', 'Grad in 6')

    for i, field in enumerate(grad_fields):
        if row[field] == '1':
            return i + 9

    best = 0
    for i, field in enumerate(sem_fields):
        if row[field] == '1':
            best = i + 1
    return best


def dollars2float(dollars: str) -> float:
    return float(dollars[1:].replace(",", ""))


def unzip(tuple_values):
    # From https://appdividend.com/2020/10/19/how-to-unzip-list-of-tuples-in-python/#:~:text=%20How%20to%20Unzip%20List%20of%20Tuples%20in,zip...%204%202%3A%20Using%20List%20Comprehension%20More%20
    return tuple(zip(*tuple_values))


@total_ordering
class Ratio:
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    def __repr__(self):
        return f'Ratio({self.numerator}, {self.denominator})'

    def __float__(self):
        return self.numerator / self.denominator

    def __lt__(self, other):
        return float(self) < float(other)

    def __eq__(self, other):
        return float(self) == float(other)

    def percent(self):
        return f'{self.numerator}/{self.denominator} ({format(float(self) * 100, ".2f")}%)'


def conditional_probability(prior_condition, posterior_condition, data):
    yes = 0
    no = 0
    for record in data:
        if prior_condition(record):
            if posterior_condition(record):
                yes += 1
            else:
                no += 1
    return Ratio(yes, yes + no)


class Tests(unittest.TestCase):
    def test_dollars(self):
        self.assertEqual(12345.6, dollars2float("$12345.60"))

    def test_unzip(self):
        self.assertEqual((('Moe', 'Larry', 'Curly'), (1, 2, 3)), unzip([('Moe', 1), ('Larry', 2), ('Curly', 3)]))

    def test_cond_prob(self):
        nums = [x for x in range(100)]
        self.assertEqual(conditional_probability(lambda x: x % 2 == 1, lambda x: x > 10, nums), Ratio(45, 50))
