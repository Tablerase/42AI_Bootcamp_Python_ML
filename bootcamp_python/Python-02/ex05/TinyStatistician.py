import numpy
from math import floor, ceil, sqrt
from typing import List, Union


class TinyStatistician:
    def mean(self, input_lst: Union[List, numpy.ndarray]):
        """Calculate mean
        The mean is the average of the numbers.

        https://www.mathsisfun.com/mean.html
        """
        res = sum(x for x in input_lst)
        res = res / len(input_lst)
        return res

    def median(self, input_lst: Union[List, numpy.ndarray]):
        """Calculate median
        It's the middle of a sorted list of numbers.

        https://www.mathsisfun.com/median.html
        """
        len_lst = len(input_lst)
        input_lst.sort()
        if len_lst % 2 == 0:
            # Even
            middle1 = int(len_lst / 2)
            middle2 = int(len_lst / 2 + 1)
            median1 = input_lst[middle1 - 1]
            median2 = input_lst[middle2 - 1]
            median = (median1 + median2) / 2
            return median
        else:
            # Odd
            middle = floor(len_lst / 2)
            median = input_lst[middle]
            return median

    def quartiles(self, input_lst: Union[List, numpy.ndarray]):
        """Calculate quartiles
        Quartiles are the values that divide a list of numbers into quarters

        https://www.mathsisfun.com/data/quartiles.html
        """
        len_lst = len(input_lst)
        if len_lst == 0:
            return None

        input_lst.sort()
        buffer = []
        q1 = None
        q3 = None
        quart_percent = 0.25
        for x in input_lst:
            buffer.append(x)
            if len(buffer) >= len_lst * 3 * quart_percent and q3 == None:
                q3 = float(buffer[-1])
            if len(buffer) >= len_lst * quart_percent and q1 == None:
                q1 = float(buffer[-1])
        return [q1, q3]

    def var(self, input_lst: Union[List, numpy.ndarray]):
        """Calculate variances
        It is the average of the squared differences from the Mean.
        Usage: calculate the standard derivation (a measure of how spread out numbers are)

        Calculate the Mean (the simple average of the numbers)
        Then for each number: subtract the Mean and square the result (the squared difference).
        Then calculate the average of those squared differences.
        https://www.mathsisfun.com/data/standard-deviation.html
        """
        mean = self.mean(input_lst)
        variance: float = 0
        for num in input_lst:
            diff: float = num - mean
            sqr_diff: float = diff ** 2
            variance = variance + sqr_diff
        variance = variance / len(input_lst)
        return variance


if __name__ == "__main__":
    t = TinyStatistician()
    lst_x = [1, 42, 300, 10, 59]
    print(f"{'Mean':_^60}")
    lst_a = [3, 7, 5, 13, 20, 23, 39, 23, 40, 23, 14, 12, 56, 23, 29]
    print(lst_a)
    res = t.mean(lst_a)
    print(f"Positive mean: {res}")
    lst_b = [3, -7, 5, 13, -2]
    print(lst_b)
    res = t.mean(lst_b)
    print(f"Negative mean: {res}")
    print(lst_x)
    res = t.mean(lst_x)
    print(f"Test: {res}")

    print(f"{'Median':_^60}")
    lst_a = [3, 13, 7, 5, 21, 23, 39, 23, 40, 23, 14, 12, 56, 23, 29]
    print(lst_a)
    res = t.median(lst_a)
    print(f"Odd list: {res}")
    lst_b = [3, 13, 7, 5, 21, 23, 23, 40, 23, 14, 12, 56, 23, 29]
    print(lst_b)
    res = t.median(lst_b)
    print(f"Even list: {res}")
    print(lst_x)
    res = t.median(lst_x)
    print(f"Test: {res}")

    print(f"{'Quartiles':_^60}")
    lst_a = [5, 7, 4, 4, 6, 2, 8]
    print(lst_a)
    res = t.quartiles(lst_a)
    print(f"Odd list: {res}")
    lst_b = [1, 3, 3, 4, 5, 6, 6, 7, 8, 8]
    print(lst_b)
    res = t.quartiles(lst_b)
    print(f"Even list: {res}")
    print(lst_x)
    res = t.quartiles(lst_x)
    print(f"Test: {res}")

    print(f"{'Variances':_^60}")
    lst_a = [600, 470, 170, 430, 300]
    print(lst_a)
    res = t.var(lst_a)
    # Expected: 21,704
    print(f"Variance: {res}")
    print(lst_x)
    res = t.var(lst_x)
    # Expected: 12279.439999999999
    print(f"Test: {res}")
