import numpy as np


class TinyStatistician:
    def mean(self, x: list | np.ndarray):
        if not isinstance(x, (list, np.ndarray)):
            return None
        if len(x) == 0:
            return None
        res = sum(a for a in x)
        res = res / len(x)
        return res

    def __quartiles(self, x: list | np.ndarray):
        quart_ratio = 0.25
        buffer = []
        quartiles = []
        x.sort()
        for value in x:
            buffer.append(value)
            if len(quartiles) == 0 and len(buffer) >= len(x) * 1 * quart_ratio:
                quartiles.append(buffer[-1])
            if len(quartiles) == 1 and len(buffer) >= len(x) * 2 * quart_ratio:
                quartiles.append(buffer[-1])
            if len(quartiles) == 2 and len(buffer) >= len(x) * 3 * quart_ratio:
                quartiles.append(buffer[-1])
        return quartiles

    def median(self, x: list | np.ndarray):
        if not isinstance(x, (list, np.ndarray)):
            return None
        if len(x) == 0:
            return None
        return self.__quartiles(x)[1]

    def quartile(self, x: list | np.ndarray):
        if not isinstance(x, (list, np.ndarray)):
            return None
        if len(x) == 0:
            return None
        quartiles = self.__quartiles(x)
        return [quartiles[0], quartiles[2]]

    def percentile(self, x: list | np.ndarray, p: int, nearest_rank: bool = False):
        if not isinstance(x, (list, np.ndarray)):
            return None
        if len(x) == 0:
            return None
        ratio = 0.1
        buffer = []
        rank_position = None
        x.sort()
        for pos, value in enumerate(x):
            buffer.append(value)
            if len(buffer) >= len(x) * ratio * p:
                rank_position = pos
        if nearest_rank == True:
            return buffer[pos]
        else:
            # Linear interpolation for percentiles
            n = len(x)
            p_index = (n - 1) * (p / 100)
            floor_index = int(np.floor(p_index))
            ceil_index = int(np.ceil(p_index))

            if floor_index == ceil_index:
                return x[floor_index]

            # Linear interpolation formula: lower_value + fraction * (upper_value - lower_value)
            # https://en.wikipedia.org/wiki/Linear_interpolation#Linear_interpolation_as_an_approximation
            fraction = p_index - floor_index
            return x[floor_index] + fraction * (x[ceil_index] - x[floor_index])

    def var(self, x: list | np.ndarray):
        if not isinstance(x, (list, np.ndarray)):
            return None
        if len(x) == 0:
            return None
        n = len(x)
        sum_diff: float = 0
        mean = self.mean(x)
        for i in x:
            diff: float = i - mean
            sum_diff += diff ** 2
        variance: float = sum_diff / n
        return variance

    def std(self, x: list | np.ndarray):
        if not isinstance(x, (list, np.ndarray)):
            return None
        if len(x) == 0:
            return None
        return np.sqrt(self.var(x))


if __name__ == "__main__":
    a = [1, 42, 300, 10, 59]
    res = TinyStatistician().mean(a)
    print(res)
    # Output:
    # 82.4
    res = TinyStatistician().median(a)
    print(res)
    # Output:
    # 42.0
    res = TinyStatistician().quartile(a)
    print(res)
    # Output:
    # [10.0, 59.0]
    res = TinyStatistician().percentile(a, 10)
    print(res)
    # Output:
    # 4.6
    res = TinyStatistician().percentile(a, 15)
    print(res)
    # Output:
    # 6.4
    res = TinyStatistician().percentile(a, 20)
    print(res)
    # Output:
    # 8.2
    res = TinyStatistician().var(a)
    print(res)
    # Output:
    #  12279.44
    res = TinyStatistician().std(a)
    print(res)
    # Output:
    #  110.8126347
