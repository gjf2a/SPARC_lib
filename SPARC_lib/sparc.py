from dataclasses import dataclass
from typing import *

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import unittest
import openpyxl
from functools import total_ordering
import math


def zipped_sorted_counts(data, xs, cond):
    """For each x, counts the number of true instances of cond(n, x) for each n in data.
    It then sorts in descending order by counts and returns a list of (x, count) pairs."""
    baseline = sorted([(x, len([n for n in data if cond(n, x)])) for x in xs], key=lambda n: -n[1])
    return [(x, count) for (x, count) in baseline if count > 0]


def sorted_condition_plot(data, x_label, xs, cond, y_label, figsize=(10, 3)):
    """Creates bar chart and returns a markdown table of data from zipped_sorted_counts()."""
    xs, data = unzip(zipped_sorted_counts(data, xs, cond))
    grouped_bar_plot([data], x_label, y_label, xs, '', [y_label], figsize=figsize,
                     legend_loc="upper right")
    return grouped_markdown_table([data], x_label, y_label, xs, '', [y_label])


def one_condition_plot(data, x_label, xs, cond, y_label, x_labeler=lambda x: str(x)):
    """cond() is a function of two arguments: a value from data, and an x value. Each bar height
    represents the number of data elements for which cond(n, x) is true. """
    return two_condition_plot(data, lambda n, x, bar: cond(n, x), x_label, xs, '', [y_label], y_label, colors=['blue'],
                              x_labeler=x_labeler)


def two_condition_counts(data, cond, xs, bars):
    """For each element of bars, creates a list of counts for each n in data for which cond(n, x, bar) is true."""
    return [[len([n for n in data if cond(n, x, bar)]) for x in xs] for bar in bars]


def two_condition_plot(data, cond, x_label, xs, bar_label, bars, y_label, colors=None, x_labeler=lambda x: str(x),
                       bar_labeler=lambda bar: str(bar), figsize=(10, 3), dpi=100, legend_loc="lower left"):
    """cond() is a function of three arguments: a value from data, an x value, and a bar value.
    Each bar height represents the number of data elements for which cond(n, x, bar) is true."""
    counts = two_condition_counts(data, cond, xs, bars)
    grouped_bar_plot(counts, x_label, y_label, [x_labeler(x) for x in xs], bar_label,
                     [bar_labeler(bar) for bar in bars], colors, figsize, dpi, legend_loc)
    return grouped_markdown_table(counts, x_label, y_label, xs, bar_label, bars, 0, add_totals=True,
                                  x_labeler=x_labeler, bar_labeler=bar_labeler)


def conditional_ratios(data, xs, bars, prior, posterior):
    """For each triple (n, x, bar) from data, xs, and bars, returns P(posterior(n, x, bar) | prior(n, x, bar))."""
    return [[conditional_probability(lambda n: prior(n, x, bar), lambda n: posterior(n, x, bar), data)
             for x in xs]
            for bar in bars]


def matching_indices(outer_len, inner_len, bool_identity, match_func):
    """Returns a subset of indices from the range [0, outer_len) for which:
    * match_func(i, j) (for each j in [0, inner_len)) is true either:
      * If bool_identity is false, for any j
      * If bool_identity is true, for all j"""
    return [i for i in range(outer_len) if match_checker(i, inner_len, bool_identity, match_func)]


def match_checker(i, inner_len, bool_identity, match_func):
    """Helper function for inner loop of matching_indices."""
    bools = (match_func(i, j) for j in range(inner_len))
    return all(bools) if bool_identity else any(bools)


def any_and_all_indices(outer_len, inner_len, getter, min_any, min_all):
    """Filters indices so that, when calling getter(i, j), only indices that meet the min_any
    and min_all constraints are included."""
    indices_any = matching_indices(outer_len, inner_len, False, lambda i, j: getter(i, j) >= min_any)
    indices_all = matching_indices(outer_len, inner_len, True, lambda i, j: getter(i, j) >= min_all)
    return sorted(list(set(indices_any).intersection(indices_all)))


def min_filtered_ratios(data, xs, bars, prior, posterior, min_any_x, min_all_x, min_any_bar, min_all_bar):
    """Filters to include only the results from conditional_ratios() whose denominators meet
    the constraints given by the four min parameters.

    Returns three items:
    1. Valid x indices.
    2. Valid bar indices.
    3. Nested list of conditional ratios, with bars as the outer index and xs as the inner index.
    """
    ratios = conditional_ratios(data, xs, bars, prior, posterior)
    x_indices = any_and_all_indices(len(xs), len(bars), lambda x, bar: ratios[bar][x].denominator, min_any_x, min_all_x)
    bar_indices = any_and_all_indices(len(bars), len(xs), lambda bar, x: ratios[bar][x].denominator, min_any_bar,
                                      min_all_bar)
    result = []
    for bar_index in bar_indices:
        bar = []
        for x_index in x_indices:
            bar.append(ratios[bar_index][x_index])
        result.append(bar)
    return [xs[i] for i in x_indices], [bars[i] for i in bar_indices], result


def zipped_sorted_ratios(data, xs, prior, posterior):
    """For each x from xs:
    1. Creates a list with pairs of (x, conditional probability P(posterior(n, x) | prior(n, x)) for n in data)
    2. Sorts the list in descending order of probability, ignoring Ratios with zeros for numerator or denominator."""
    baseline = [(x, conditional_probability(lambda n: prior(n, x), lambda n: posterior(n, x), data)) for x in xs]
    data = [(x, ratio) for (x, ratio) in baseline if ratio.defined() and ratio.numerator > 0]
    return sorted(data, key=lambda p: -float(p[1]))


def sorted_conditional_plot(data, x_label, xs, post_label, prior, posterior, x_labeler=lambda x: str(x),
                            figsize=(10, 3)):
    """Creates bar chart and returns a markdown table of data from zipped_sorted_ratios()."""
    xs, ratios = unzip(zipped_sorted_ratios(data, xs, prior, posterior))
    xs = [x_labeler(x) for x in xs]
    probs = [float(r) for r in ratios]
    grouped_bar_plot([probs], x_label, post_label, xs, '', [post_label], figsize=figsize, legend_loc="upper right")
    return grouped_markdown_table([ratios], x_label, post_label, xs, '', [post_label],
                                  Ratio(0, 0), convert=lambda r: r.percent())


def conditional_plot(data, x_label, xs, bar_label, bars, post_label, prior, posterior, colors=None,
                     x_labeler=lambda x: str(x), bar_labeler=lambda bar: str(bar), figsize=(10, 3), dpi=100,
                     legend_loc='upper left', min_any_x=0, min_all_x=0, min_any_bar=0, min_all_bar=0, add_totals=True):
    """Creates bar chart and returns a markdown table of data from min_filtered_ratios()."""
    xs, bars, ratios = min_filtered_ratios(data, xs, bars, prior, posterior, min_any_x, min_all_x, min_any_bar,
                                           min_all_bar)
    x_labels = [x_labeler(x) for x in xs]
    bar_labels = [bar_labeler(bar) for bar in bars]
    probs = [[float(r) if r.defined() else 0.0 for r in rs] for rs in ratios]
    grouped_bar_plot(probs, x_label, post_label, x_labels, bar_label, bar_labels, colors, figsize, dpi, legend_loc)
    return grouped_markdown_table(ratios, x_label, post_label, x_labels, bar_label, bar_labels,
                                  Ratio(0, 0), add_totals=add_totals, convert=lambda r: r.percent())


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
    return grouped_markdown_table([averages], x_label, y_label, xs, '', [y_label],
                                  0.0, convert=lambda avg: '{:.4f}'.format(avg))


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


def grouped_markdown_table(nested_data, x_label, y_label, x_labels, bar_label, bar_labels,
                           additive_identity=0, add_totals=False, convert=lambda d: d,
                           x_labeler=lambda x: str(x), bar_labeler=lambda bar: str(bar)):
    table_data = []
    x_labels = [x_labeler(x) for x in x_labels]
    if add_totals:
        nested_data = totaled_nested_data(nested_data, additive_identity)
        bar_labels = bar_labels + ["Total"]
        x_labels.append("Total")
    for i, values in enumerate(nested_data):
        row = [bar_labeler(bar_labels[i]) if bar_labels[i] != 'Total' else bar_labels[i]]
        for value in values:
            row.append(convert(value))
        table_data.append(row)
    return f'## {y_label}\n\n' + make_markdown_table([bar_label] + [f"{x_label}: {x_labels[0]}"] + x_labels[1:],
                                                     table_data)


def totaled_nested_data(nested_data, additive_identity):
    totaled = []
    for row in nested_data:
        augmented = row[:]
        augmented.append(sum(row, additive_identity))
        totaled.append(augmented)
    bottom_row = []
    for i in range(len(totaled[0])):
        bottom_row.append(sum([totaled[j][i] for j in range(len(totaled))], additive_identity))
    totaled.append(bottom_row)
    return totaled


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


def basic_scatter(x_label, xs, y_label, ys, alpha=1.0, figsize=(8, 7)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.scatter(xs, ys, alpha=alpha)


def filtered_scatter(x_label, y_label, records, x_select, y_select, record_filter=lambda r: True, alpha=1.0,
                     figsize=(8, 7)):
    filtered = [record for record in records if record_filter(record)]
    xs = [x_select(record) for record in filtered]
    ys = [y_select(record) for record in filtered]
    basic_scatter(x_label, xs, y_label, ys, alpha, figsize)


def labeled_scatter(x_label, x_zipped, y_label, y_zipped, x_convert=lambda x: x, y_convert=lambda y: y, refline=None,
                    figsize=(8, 7)):
    label2xy = merge_lists(x_zipped, y_zipped)
    xs = [x_convert(x) for x, y in label2xy.values()]
    ys = [y_convert(y) for x, y in label2xy.values()]
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
    s = f"| {' | '.join(headers)} |\n| {' | '.join([(max(1, len(header) - 1)) * '-' + ':' for header in headers])} |\n"
    for row in data:
        s += f"| {' | '.join([str(item) for item in row])} |\n"
    return s


def grade2points(grade):
    return max(0.0, 4.0 - (ord(grade.upper()) - ord('A')))


def percent_str_from(num, denom):
    return format(num * 100 / denom, ".2f") + "%" if denom != 0 else "Undefined"


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
        if self.defined() and other.defined():
            return float(self) < float(other)
        elif self.defined() and not other.defined():
            return True
        else:
            return False

    def __eq__(self, other):
        if self.defined() and other.defined():
            return float(self) == float(other)
        else:
            return self.defined() == other.defined()

    def __add__(self, other: 'Ratio'):
        return Ratio(self.numerator + other.numerator, self.denominator + other.denominator)

    def __neg__(self):
        return Ratio(-self.numerator, -self.denominator)

    def __sub__(self, other: 'Ratio'):
        return self + (-other)

    def percent(self):
        return f'{self.numerator}/{self.denominator} ({percent_str_from(self.numerator, self.denominator)})'

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
        return Course(discipline, number, section, title.strip(), grade.strip(), int(year), term)


def int_filter_nan(value):
    if type(value) == str and value.isdigit():
        return int(value)


def float_filter_nan(value):
    if type(value) == str:
        return float(value)


def str_filter_nan(value):
    return value if type(value) == str else ''


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

    def percent_str(self):
        return percent_str_from(self.total, self.count)

    def defined(self):
        return self.count > 0


def discipline_averagers(courses: List[Course]) -> Dict[str, Averager]:
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


def great_circle(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    lon1 = math.radians(lon1)
    lon2 = math.radians(lon2)
    value = math.acos(math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(abs(lon1 - lon2)))
    value = math.degrees(value)
    return 69.0 * value


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

    def ranking(self, min_count=0):
        return [(key, count) for (count, key) in
                reversed(sorted([(count, key) for (key, count) in self.hist.items() if count >= min_count]))]


def plot_histogram_counts(hist: Histogram, x_label, figsize=(10, 3), dpi=100,
                          legend_loc="lower right", min_count=0):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    x_labels, xs = unzip(hist.ranking(min_count))
    plt.bar(x_labels, xs)
    plt.xticks(ticks=[n for n in range(len(x_labels))], labels=x_labels)
    plt.legend(loc=legend_loc)


class ConfusionMatrix:
    def __init__(self, data_list, predict_func, actual_func):
        self.true_pos = 0
        self.false_pos = 0
        self.true_neg = 0
        self.false_neg = 0

        for datum in data_list:
            if predict_func(datum):
                if actual_func(datum):
                    self.true_pos += 1
                else:
                    self.false_pos += 1
            else:
                if actual_func(datum):
                    self.false_neg += 1
                else:
                    self.true_neg += 1

    def precision(self):
        denominator = self.true_pos + self.false_pos
        return self.true_pos / denominator if denominator != 0 else None

    def recall(self):
        denominator = self.true_pos + self.false_neg
        return self.true_pos / denominator if denominator != 0 else None

    def total_pos(self):
        return self.true_pos + self.false_pos


def precision_recall_points(threshold_list, data_list, predict_func_maker, actual_func):
    points = []
    for threshold in threshold_list:
        predict_func = predict_func_maker(threshold)
        matrix = ConfusionMatrix(data_list, predict_func, actual_func)
        points.append((threshold, matrix.true_pos, matrix.false_pos, matrix.true_neg, matrix.false_neg,
                       matrix.precision(), matrix.recall()))
    return points


def scaled_pr_points(pr_points, target_pop_size):
    scaled_points = []
    for (threshold, true_pos, false_pos, true_neg, false_neg, precision, recall) in pr_points:
        pop = true_pos + true_neg + false_pos + false_neg
        tps = true_pos * target_pop_size // pop
        fps = false_pos * target_pop_size // pop
        tns = true_neg * target_pop_size // pop
        fns = false_neg * target_pop_size // pop
        scaled_points.append((threshold, tps, fps, tns, fns, precision, recall))
    return scaled_points


def precision_recall_markdown(threshold_header, pr_points):
    return make_markdown_table([threshold_header, "True +", "False +", "True -", "False -", "Precision", "Recall"],
                               pr_points)


def precision_recall_auc(pr_points):
    x_y_points = [(r, p) for (t, tp, fp, tn, fn, p, r) in pr_points if r is not None and p is not None]
    return area_under_curve(x_y_points)


def area_under_curve(x_y_points):
    area = 0
    for i in range(len(x_y_points) - 1):
        x1, y1 = x_y_points[i]
        x2, y2 = x_y_points[i + 1]
        assert x2 >= x1
        area += (x2 - x1) * ((y1 + y2) / 2)
    return area


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
                                    Ratio(0, 0), add_totals=True, convert=lambda d: d.percent())
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

    def test_ratio_inequalities(self):
        ratios = [Ratio(8, 436), Ratio(53, 2378), Ratio(7, 111), Ratio(2, 22), Ratio(195, 1770), Ratio(420, 1946),
                  Ratio(5, 12)]
        for i in range(1, len(ratios)):
            self.assertTrue(ratios[i - 1] < ratios[i])
            self.assertTrue(ratios[i] > ratios[i - 1])

    def test_ratio_arithmetic(self):
        total = Ratio(22, 29) + Ratio(3, 4)
        self.assertEqual(total, Ratio(25, 33))
        diff = total - Ratio(1, 2)
        self.assertEqual(diff, Ratio(24, 31))

    def test_zipped_sorted_counts(self):
        data = [i for i in range(1, 21)]
        num_factors_counts = zipped_sorted_counts(data, [i for i in range(1, 8)],
                                                  lambda n, x: x == len([m for m in range(1, n + 1) if n % m == 0]))
        self.assertEqual(num_factors_counts, [(2, 8), (4, 5), (6, 3), (3, 2), (1, 1), (5, 1)])

    def test_matching_indices(self):
        matching_any = matching_indices(100, 10, False, lambda i, j: i >= (1 + j) * 5)
        self.assertEqual([i for i in range(5, 100)], matching_any)
        matching_all = matching_indices(100, 10, True, lambda i, j: i >= (1 + j) * 5)
        self.assertEqual([i for i in range(50, 100)], matching_all)

    # def test_filtered_ratios(self):
    #    data = [(1, 2), (2, 2), (3, 2), (3, 4), (2, 4), (10, 2), (10, 5), (8, 3)]
    #    min_filtered_ratios(data, xs, bars, prior, posterior, min_denominator_any, min_denominator_all)
