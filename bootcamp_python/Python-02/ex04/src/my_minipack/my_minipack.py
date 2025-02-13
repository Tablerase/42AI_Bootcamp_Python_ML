import sys
from typing import List
from time import sleep, time
import time
from random import randint
import os
from functools import wraps


def ft_progress(input_list: List):
    """
    The function will display the progress of a for loop

    Args:
        input_list (list) : List that will be iterate over

    Resources:
        https://docs.python.org/3/reference/expressions.html#yield-expressions
    """
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
        print(
            f"ETA {eta_time:.2f}s [{ratio:4.0%}][{bar}] {pos}/{max_len} | elapsed time {elapsed_time:.2f}s", end='\r')

        yield input_list[pos]


def log(func):
    """
    This decorator logged the execution time of the wrapped function into a file

    Args:
        func (_type_): function that will be wrap

    Returns:
        _type_: function return

    Example:
        ```python
        @log
        def make_coffee(self):
            if self.start_machine():
                for _ in range(20):
                    time.sleep(0.1)
                    self.water_level -= 1
                print(self.boil_water())
                print("Coffee is ready!")
        ```
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Parameters
        user = os.getenv('USER')
        function_name: str = func.__name__
        # Timers
        start_time = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end_time: float = (time.perf_counter_ns() - start_time) / 1000000
        if end_time > 1000:
            end_time /= 1000
            unit = 's'
        else:
            unit = 'ms'
        exec_time = f"{end_time:.3f} {unit:2}"
        with open('./machine.log', 'a') as f:
            to_log: str = f"({user})Running: {function_name.replace('_',' ').title():<20} [ exec-time = {exec_time} ]\n"
            # to_log: str = f"({user})Running: {function_name.replace('_',' ').title():<20} [ exec-time = {exec_time} ]"
            # print(to_log)
            f.write(to_log)
        return result
    return wrapper
