from dataclasses import dataclass
from typing import *

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import unittest
import openpyxl
from functools import total_ordering


def zipped_sorted_counts(data, xs, cond):
    baseline = sorted([(x, len([n for n in data if cond(n, x)])) for x in xs], key=lambda n: -n[1])
    return [(x, count) for (x, count) in baseline if count > 0]


def sorted_condition_plot(data, x_label, xs, cond, y_label, figsize=(10, 3)):
    xs, data = unzip(zipped_sorted_counts(data, xs, cond))
    grouped_bar_plot([data], x_label, y_label, xs, '', [y_label], figsize=figsize,
                     legend_loc="upper right")
    return grouped_markdown_table([data], x_label, y_label, xs, '', [y_label])


def one_condition_plot(data, x_label, xs, cond, y_label):
    return two_condition_plot(data, lambda n, x, bar: cond(n, x), x_label, xs, '', [y_label], y_label, colors=['blue'])


def two_condition_counts(data, cond, xs, bars):
    return [[len([n for n in data if cond(n, x, bar)]) for x in xs] for bar in bars]


def two_condition_plot(data, cond, x_label, xs, bar_label, bars, y_label, colors=None, x_labeler=lambda x: str(x),
                       bar_labeler=lambda bar: str(bar), figsize=(10, 3), dpi=100):
    counts = two_condition_counts(data, cond, xs, bars)
    grouped_bar_plot(counts, x_label, y_label, [x_labeler(x) for x in xs], bar_label,
                     [bar_labeler(bar) for bar in bars], colors, figsize, dpi)
    return grouped_markdown_table(counts, x_label, y_label, xs, bar_label, bars, x_labeler=x_labeler,
                                  bar_labeler=bar_labeler)


def conditional_ratios(data, xs, bars, prior, posterior):
    return [[conditional_probability(lambda n: prior(n, x, bar), lambda n: posterior(n, x, bar), data)
             for x in xs]
            for bar in bars]


def zipped_sorted_ratios(data, xs, prior, posterior):
    baseline = [(x, conditional_probability(lambda n: prior(n, x), lambda n: posterior(n, x), data)) for x in xs]
    data = [(x, ratio) for (x, ratio) in baseline if ratio.defined()]
    return sorted(data, key=lambda p: -float(p[1]))


def sorted_conditional_plot(data, x_label, xs, post_label, prior, posterior, x_labeler=lambda x: str(x), figsize=(10, 3)):
    xs, ratios = unzip(zipped_sorted_ratios(data, xs, prior, posterior))
    xs = [x_labeler(x) for x in xs]
    probs = [float(r) for r in ratios]
    grouped_bar_plot([probs], x_label, post_label, xs, '', [post_label], figsize=figsize, legend_loc="upper right")
    return grouped_markdown_table([ratios], x_label, post_label, xs, '', [post_label], lambda r: r.percent())


def conditional_plot(data, x_label, xs, bar_label, bars, post_label, prior, posterior, colors=None,
                     x_labeler=lambda x: str(x), bar_labeler=lambda bar: str(bar), figsize=(10, 3), dpi=100,
                     legend_loc='upper left'):
    ratios = conditional_ratios(data, xs, bars, prior, posterior)
    x_labels = [x_labeler(x) for x in xs]
    bar_labels = [bar_labeler(bar) for bar in bars]
    probs = [[float(r) if r.defined() else 0.0 for r in rs] for rs in ratios]
    grouped_bar_plot(probs, x_label, post_label, x_labels, bar_label, bar_labels, colors, figsize, dpi, legend_loc)
    return grouped_markdown_table(ratios, x_label, post_label, x_labels, bar_label, bar_labels, lambda r: r.percent())


def interval_ratio_plot(data, x_label, xs, x_getter, y_label, y_test, bar_label, bars, bar_getter, colors=None,
                        figsize=(10, 3), dpi=100, legend_loc="upper left"):
    return conditional_plot(data, x_label, intervals_from(xs), bar_label, intervals_from(bars), y_label,
                            lambda n, x, bar: in_interval(x_getter(n, bar[0]), x) and in_interval(bar_getter(n), bar),
                            lambda n, x, bar: y_test(n, x[0], bar[0]),
                            colors, make_range_label, make_range_label, figsize, dpi, legend_loc)


def zipped_sorted_averages(data, value_getter, labels_from, label_matcher):
    labels = set()
    for row in data:
        for label in labels_from(row):
            labels.add(label)
    label_averages = {label: Averager() for label in labels}
    for row in data:
        for label in labels:
            if label_matcher(row, label):
                label_averages[label].add(value_getter(row))
    unsorted = [(label, avg.average()) for label, avg in label_averages.items() if avg.defined()]
    return sorted(unsorted, key=lambda p: -p[1])


def sorted_average_plot(data, x_label, y_label, value_getter, labels_from, label_matcher, figsize=(10, 3)):
    xs, averages = unzip(zipped_sorted_averages(data, value_getter, labels_from, label_matcher))
    grouped_bar_plot([averages], x_label, y_label, xs, '', [y_label], figsize=figsize, legend_loc="upper right")
    return grouped_markdown_table([averages], x_label, y_label, xs, '', [y_label], convert=lambda avg: '{:.4f}'.format(avg))


def grouped_bar_plot(nested_data, x_label, y_label, x_labels, bar_label, bar_labels, colors=None,
                     figsize=(10, 3), dpi=100, legend_loc='upper left'):
    if colors is None:
        colors = ['blue']
    colors = InfiniteRepeatingList(colors)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    X = np.arange(len(x_labels))
    width = 1.0 / (len(nested_data) + 1.5)
    for i in range(len(nested_data)):
        ax.bar(X + i * width, nested_data[i], color=colors[i], width=width,
               label=f'{bar_label} {bar_labels[i]}'.strip())
    plt.xticks(ticks=[n for n in range(len(x_labels))], labels=x_labels)
    plt.legend(loc=legend_loc)


def grouped_markdown_table(nested_data, x_label, y_label, x_labels, bar_label, bar_labels, convert=lambda d: d,
                           x_labeler=lambda x: str(x), bar_labeler=lambda bar: str(bar)):
    table_data = []
    for i, values in enumerate(nested_data):
        row = [bar_labeler(bar_labels[i])]
        for value in values:
            row.append(convert(value))
        table_data.append(row)
    x_labels = [x_labeler(x) for x in x_labels]
    return f'## {y_label}\n\n' + make_markdown_table([bar_label] + [f"{x_label}: {x_labels[0]}"] + x_labels[1:],
                                                     table_data)


def list2dict(items):
    return {key: value for key, value in items}


def merge_lists(items1, items2):
    d1 = list2dict(items1)
    d2 = list2dict(items2)
    result = {}
    for k1, v1 in d1.items():
        if k1 in d2:
            result[k1] = (v1, d2[k1])
    return result


def labeled_scatter(x_label, x_zipped, y_label, y_zipped, refline=None, x_convert=lambda x: x, y_convert=lambda y: y,
                    figsize=(8, 7)):
    label2xy = merge_lists(x_zipped, y_zipped)
    xs = [x for x, y in label2xy.values()]
    ys = [y for x, y in label2xy.values()]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.scatter(xs, ys)
    if refline:
        plt.plot(refline[0], refline[1])
    for label, xy in label2xy.items():
        plt.annotate(label, xy)


def make_interval_label(value_list, i):
    make_range_label(value_list[i], value_list[i + 1] if i + 1 < len(value_list) else None)


def make_range_label(start, end=None):
    if end is None and type(start) == tuple and len(start) == 2:
        start, end = start
    if end is None:
        return f"{start}+"
    else:
        if type(start) == int and type(end) == int:
            end -= 1
        if start == end:
            return f"{start}"
        else:
            return f"{start}-{end}"


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
    return list(zip(xs, xs[1:] + [None]))


def in_interval(value, start, end=None):
    if value is None:
        return False
    else:
        if end is None and type(start) == tuple and len(start) == 2:
            start, end = start
        return value >= start and (end is None or end > value)


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
        return f'{self.numerator}/{self.denominator} ({format(float(self) * 100, ".2f") + "%" if self.defined() else "Undefined"})'

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
        if not ratio_list[i - 1].defined():
            if ratio_list[i].defined():
                return i
        elif ratio_list[i].defined():
            size = abs(float(ratio_list[i]) - float(ratio_list[i - 1]))
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

    def matches(self, discipline: str, number: str) -> bool:
        return self.discipline == discipline and self.number == number

    def weight(self) -> float:
        if (self.discipline == 'LBST' and self.number == '101') or self.discipline in ('DANA', 'TARA', 'MUSA', 'PACT'):
            return 0.25
        else:
            return 1.0

    def semester_number(self, entrance_year):
        number = 1 + 2 * (self.year - int(entrance_year))
        if self.term == '2S':
            number += 1
        return number


def course_info(code: str, title: str, grade: str, year: str, term=None) -> Course:
    if type(code) == str:
        code_parts = code.split()
        while len(code_parts) < 3:
            code_parts.append('')
        while len(code_parts) > 3:
            code_parts.pop()
        discipline, number, section = code_parts
        if term is None:
            year, term = year.split("_")
        return Course(discipline, number, section, title, grade, int(year), term)


def load_course_table(courses):
    student2courses = {}
    for index, row in courses.iterrows():
        id_num = row['id_num']
        if id_num not in student2courses:
            student2courses[id_num] = []
        student2courses[id_num].append(
            course_info(row['crs_cde'], row['crs_title'], row['grade_cde'], row['yr_cde'], row['trm_cde']))
    return student2courses


def build_term_to_courses(courses, entrance_year):
    term2courses = {}
    for course in courses:
        term = course.semester_number(entrance_year)
        if term not in term2courses:
            term2courses[term] = []
        term2courses[term].append(course)
    return term2courses


def lowest_grade_from(courses):
    courses = [course.grade for course in courses if course.grade in 'ABCDF']
    return max(courses) if len(courses) > 0 else None


def has_taken(courses: List[Course], discipline: str, number: str) -> bool:
    return has_match(courses, lambda course: course.matches(discipline, number))


def has_match(courses: List[Course], predicate) -> bool:
    for course in courses:
        if predicate(course):
            return True
    return False


def first_grade_for(courses: List[Course], discipline: str, number: str) -> str:
    for course in courses:
        if course.matches(discipline, number):
            return course.grade


class Averager:
    def __init__(self, total=0, count=0):
        self.total = total
        self.count = count

    def __repr__(self):
        return f'Averager({self.total}, {self.count})'

    def __add__(self, other: 'Averager'):
        return Averager(self.total + other.total, self.count + other.count)

    def add(self, value, weight=1.0):
        if type(value) == str:
            if value in 'ABCDF':
                value = grade2points(value)
            else:
                return None
        self.total += value * weight
        self.count += weight

    def average(self):
        if self.defined():
            return self.total / self.count

    def defined(self):
        return self.count > 0


def discipline_averagers(courses: List[Course]) -> Dict[str,Averager]:
    result = {}
    for course in courses:
        if course.discipline not in result:
            result[course.discipline] = Averager()
        result[course.discipline].add(course.grade, course.weight())
    return result


def category_gpas(courses: List[Course], category: List[str], exclusions: List[str] = None) -> (float, float):
    averagers = discipline_averagers(courses)
    category_value = Averager()
    other = Averager()
    for discipline, averager in averagers.items():
        if not exclusions or discipline not in exclusions:
            if discipline in category:
                category_value += averager
            else:
                other += averager
    return category_value.average(), other.average()


def get_term_data(row, suffix):
    return {i: float(row[f'Term {i} {suffix}']) for i in range(1, 9)}


class Histogram:
    def __init__(self, hist=None):
        self.hist = {} if hist is None else hist

    def __repr__(self):
        return f"Histogram({self.hist})"

    def bump(self, key):
        if key not in self.hist:
            self.hist[key] = 0
        self.hist[key] += 1

    def count_for(self, key):
        return self.hist.get(key, 0)

    def total_count(self):
        return sum(self.hist.values())

    def all_labels(self):
        return self.hist.keys()

    def mode(self):
        return max([(count, key) for (key, count) in self.hist.items()])[1]

    def ranking(self):
        return [(key, count) for (count,key) in reversed(sorted([(count, key) for (key, count) in self.hist.items()]))]


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
        for test in (
        (((1, True), (2, True), (3, False)), (1, 3)), (((2, False), (3, True), (4, True), (5, False)), (3, 5))):
            for outcome in test[0]:
                self.assertEqual(in_interval(outcome[0], test[1][0], test[1][1]), outcome[1])
                self.assertEqual(in_interval(outcome[0], test[1]), outcome[1])

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

    def test_percent(self):
        ratios = [(Ratio(1, 10), '1/10 (10.00%)'), (Ratio(2, 3), '2/3 (66.67%)'), (Ratio(0, 0), '0/0 (Undefined)')]
        for ratio, output in ratios:
            self.assertEqual(output, ratio.percent())

    def test_grouped_markdown(self):
        data = [[Ratio(3, 5), Ratio(7, 7), Ratio(8, 8), Ratio(13, 13), Ratio(21, 21)],
                [Ratio(19, 30), Ratio(24, 25), Ratio(44, 45), Ratio(79, 80), Ratio(129, 132)],
                [Ratio(9, 12), Ratio(20, 23), Ratio(31, 34), Ratio(68, 71), Ratio(112, 116)],
                [Ratio(3, 5), Ratio(17, 17), Ratio(20, 21), Ratio(27, 27), Ratio(47, 47)],
                [Ratio(4, 5), Ratio(7, 9), Ratio(6, 6), Ratio(10, 11), Ratio(20, 20)]]
        md = grouped_markdown_table(data, 'Career GPA after Semester 1', 'Fraction retained in Semester 2',
                                    ['0-2.0', '2.0-2.5', '2.5-3.0', '3.0-3.5', '3.5+'], 'Median Zip9 Income',
                                    ['0-40000', '40000-80000', '80000-120000', '120000-160000', '160000+'],
                                    lambda d: d.percent())
        print(md)

    def test_hist(self):
        hist = Histogram()
        values = [10, 15, 20]
        for i in range(len(values)):
            for j in range(values[i]):
                hist.bump(i)

        for i in range(len(values)):
            self.assertEqual(hist.count_for(i), values[i])

        self.assertEqual(len(values), len(hist.all_labels()))
        self.assertEqual(sum(values), hist.total_count())
        self.assertEqual(2, hist.mode())
        self.assertEqual([(2, 20), (1, 15), (0, 10)], hist.ranking())
