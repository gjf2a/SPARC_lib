from dataclasses import dataclass
from typing import List

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import unittest
import openpyxl
from functools import total_ordering


def grouped_bar_plot(nested_data, x_label, y_label, x_labels, bar_labels, colors=None, width=0.1, figsize=(10, 8), dpi=100):
    if colors is None:
        colors = ['blue']
    colors = InfiniteRepeatingList(colors)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    X = np.arange(len(x_labels))
    for i in range(len(nested_data)):
        ax.bar(X + i * width, nested_data[i], color = colors[i], width = width, label=bar_labels[i])
    plt.xticks(ticks=[n for n in range(len(x_labels))], labels=x_labels)
    plt.legend(loc="upper left")
    table_data = nested_data[:]
    for i in range(len(table_data)):
        table_data[i] = [bar_labels[i]] + table_data[i]
    return make_markdown_table([y_label] + [f"{x_label}: " + x_labels[0]] + x_labels[1:], nested_data)

#def make_nested_data(data_set, )


def make_interval_label(value_list, i):
    if i + 1 == len(value_list):
        return f"{value_list[i]}+"
    else:
        return f"{value_list[i]}-{value_list[i+1]}"


def make_interval_label_list(value_list):
    return [make_interval_label(value_list, i) for i in range(len(value_list))]


def enrollment_map_depths(filename):
    depths = {}
    data = pd.read_excel(filename, dtype=str)
    for index, row in data.iterrows():
        student = row['id_num']
        depths[student] = map_depth_for(row)
    return depths


def map_depth_for(row):
    sem_fields = ('1st Sem', '2nd Sem', '3rd Sem', '4th Sem', '5th Sem', '6th Sem', '7th Sem', '8th Sem')
    best = 0
    for i, field in enumerate(sem_fields):
        if row[field] in ('I', '1'):
            best = i + 1
    return best


def dollars2float(dollars: str) -> float:
    return float(dollars[1:].replace(",", ""))


def unzip(tuple_values):
    # From https://appdividend.com/2020/10/19/how-to-unzip-list-of-tuples-in-python/#:~:text=%20How%20to%20Unzip%20List%20of%20Tuples%20in,zip...%204%202%3A%20Using%20List%20Comprehension%20More%20
    return tuple(zip(*tuple_values))


def intervals_from(xs):
    return zip(xs, xs[1:] + [None])


def in_interval(value, x, x_next):
    return value >= x and (x_next is None or x_next > value)


def make_markdown_table(headers: List[str], data: List) -> str:
    s = f"| {' | '.join(headers)} |\n| {' | '.join([(len(header) - 1) * '-' + ':' for header in headers])} |\n"
    for row in data:
        s += f"| {' | '.join([str(item) for item in row])} |\n"
    return s


def grade2points(grade):
    return max(0.0, 4.0 - (ord(grade.upper()) - ord('A')))


@total_ordering
class Ratio:
    def __init__(self, numerator=0, denominator=0):
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

    def defined(self):
        return self.denominator != 0

    def count(self, prior, posterior):
        if prior:
            self.denominator += 1
            if posterior:
                self.numerator += 1

    def observe(self, prior_condition, posterior_condition, record):
        if prior_condition(record):
            self.denominator += 1
            if posterior_condition(record):
                self.numerator += 1


# Ratios are intended to represent observations. Consequently, summing them isn't a matter
# of finding common denominators and adding; we instead sum the denominators and numerators
# to produce an overall ratio.
def observation_sum(ratios: List[Ratio]) -> Ratio:
    return Ratio(sum([r.numerator for r in ratios]), sum([r.denominator for r in ratios]))


def conditional_probability(prior_condition, posterior_condition, data):
    r = Ratio()
    for record in data:
        r.observe(prior_condition, posterior_condition, record)
    return r


def matching_records(prior_condition, posterior_condition, data):
    yes = []
    no = []
    for record in data:
        if prior_condition(record):
            if posterior_condition(record):
                yes.append(record)
            else:
                no.append(record)
    return yes, no


def find_biggest_jump(ratio_list: List[Ratio]) -> int:
    biggest = 0
    biggest_size = 0.0
    for i in range(1, len(ratio_list)):
        if not ratio_list[i-1].defined():
            if ratio_list[i].defined():
                return i
        elif ratio_list[i].defined():
            size = abs(float(ratio_list[i]) - float(ratio_list[i-1]))
            if size > biggest_size:
                biggest = i
                biggest_size = size
    return biggest


class InfiniteRepeatingList:
    def __init__(self, values2repeat):
        self.values = values2repeat[:]

    def __getitem__(self, key):
        return self.values[key % len(self.values)]


@dataclass
class Course:
    discipline: str
    number: str
    section: str
    title: str
    grade: str
    year: int
    term: str


def course_info(code: str, title: str, grade: str, yr_term: str) -> Course:
    if type(code) == str:
        code_parts = code.split()
        if len(code_parts) == 2:
            code_parts.append('')
        discipline, number, section = code_parts
        year, term = yr_term.split("_")
        return Course(discipline, number, section, title, grade, int(year), term)


class Tests(unittest.TestCase):
    def test_dollars(self):
        self.assertEqual(12345.6, dollars2float("$12345.60"))

    def test_unzip(self):
        self.assertEqual((('Moe', 'Larry', 'Curly'), (1, 2, 3)), unzip([('Moe', 1), ('Larry', 2), ('Curly', 3)]))

    def test_count(self):
        r = Ratio()
        for i in range(100):
            r.count(i % 2 == 0, i % 3 == 0)
        self.assertEqual(r, Ratio(17, 50))

    def test_obs_sum(self):
        ratios = [Ratio(1, 3), Ratio(2, 4), Ratio(4, 5)]
        self.assertEqual(observation_sum(ratios), Ratio(7, 12))

    def test_cond_prob(self):
        nums = [x for x in range(100)]
        self.assertEqual(conditional_probability(lambda x: x % 2 == 1, lambda x: x > 10, nums), Ratio(45, 50))

    def test_make_markdown(self):
        table = make_markdown_table(["Alpha", "Beta", "Gamma"], [(1, 2, 3), (3, 6, 9)])
        target = """| Alpha | Beta | Gamma |
| ----: | ---: | ----: |
| 1 | 2 | 3 |
| 3 | 6 | 9 |
"""
        self.assertEqual(table, target)

    def test_repeating_list(self):
        rep = InfiniteRepeatingList(['a', 'b', 'c'])
        self.assertEqual(rep[0], 'a')
        self.assertEqual(rep[1], 'b')
        self.assertEqual(rep[2], 'c')
        self.assertEqual(rep[3], 'a')
        self.assertEqual(rep[4], 'b')
        self.assertEqual(rep[5], 'c')
        self.assertEqual(rep[27], 'a')
        self.assertEqual(rep[28], 'b')
        self.assertEqual(rep[29], 'c')

    def test_grade_pts(self):
        for (g, p) in [('A', 4.0), ('B', 3.0), ('C', 2.0), ('D', 1.0), ('F', 0.0)]:
            self.assertEqual(p, grade2points(g))

    def test_intervals(self):
        intervals = list(intervals_from([1, 3, 5, 7]))
        self.assertEqual([(1, 3), (3, 5), (5, 7), (7, None)], intervals)
        for test in ((((1, True), (2, True), (3, False)), (1, 3)), (((2, False), (3, True), (4, True), (5, False)), (3, 5))):
            for outcome in test[0]:
                self.assertEqual(in_interval(outcome[0], test[1][0], test[1][1]), outcome[1])

    def test_course(self):
        c1 = course_info("ENGL 234  04", "Creative Nonfiction - The Essay", "A", "2019_2S")
        self.assertEqual(c1, Course("ENGL", '234', '04', "Creative Nonfiction - The Essay", "A", 2019, "2S"))
        c2 = course_info('MATH 130', 'Calculus I', 'CR', '2014_2S')
        self.assertEqual(c2, Course("MATH", '130', '', 'Calculus I', 'CR', 2014, '2S'))

    def test_biggest_jump(self):
        tests = [[Ratio(3, 5), Ratio(7, 7), Ratio(8, 8), Ratio(13, 13), Ratio(21, 21)],
                 [Ratio(19, 30), Ratio(24, 25), Ratio(44, 45), Ratio(79, 80), Ratio(129, 132)],
                 [Ratio(9, 12), Ratio(20, 23), Ratio(31, 34), Ratio(68, 71), Ratio(112, 116)],
                 [Ratio(3, 5), Ratio(17, 17), Ratio(20, 21), Ratio(27, 27), Ratio(47, 47)],
                 [Ratio(4, 5), Ratio(7, 9), Ratio(6, 6), Ratio(10, 11), Ratio(20, 20)]]
        answers = [1, 1, 1, 1, 2]
        for i in range(len(tests)):
            self.assertEqual(answers[i], find_biggest_jump(tests[i]))

        self.assertEqual(1, find_biggest_jump([Ratio(0, 0), Ratio(1, 2), Ratio(3, 3)]))