import sys
from typing import List
from time import sleep, time

"""
The function will display the progress of a for loop

https://docs.python.org/3/reference/expressions.html#yield-expressions
Not allowed: https://tqdm.github.io/
"""

def ft_progress(input_list: List):
    max_len = len(input_list)
    start_time = time()
    bar_len = 20
    for pos in range(len(input_list)):
        # time update
        elapsed_time = time() - start_time
        interval_time = elapsed_time / (pos + 1 if pos > 0 else 1)
        eta_time = interval_time * max_len
        # bar update
        ratio = (pos/max_len)
        filled = int(ratio * bar_len)
        bar = '=' * (filled) + '>' + ' ' * (bar_len - filled - 1)
        print(f"ETA {eta_time:.2f}s [{ratio:4.0%}][{bar}] {pos}/{max_len} | elapsed time {elapsed_time:.2f}s", end='\r')

        yield input_list[pos]


if __name__ == "__main__":
    # listy = range(1000)
    # ret = 0
    # for elem in ft_progress(listy):
    #     ret += (elem + 3) % 5
    #     sleep(0.01)
    # print()
    # print(ret)
    listy = range(3333)
    ret = 0
    for elem in ft_progress(listy):
        ret += elem
        sleep(0.005)
    print()
    print(ret)