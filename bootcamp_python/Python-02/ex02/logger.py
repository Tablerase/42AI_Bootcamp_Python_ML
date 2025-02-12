import time
from random import randint
import os
from functools import wraps

# ... your definition of log decorator...


def log(func):
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


class CoffeeMachine():
    water_level = 100

    @log
    def start_machine(self):
        if self.water_level > 20:
            return True
        else:
            print("Please add water!")
            return False

    @log
    def boil_water(self):
        return "boiling..."

    @log
    def make_coffee(self):
        if self.start_machine():
            for _ in range(20):
                time.sleep(0.1)
                self.water_level -= 1
            print(self.boil_water())
            print("Coffee is ready!")

    @log
    def add_water(self, water_level):
        time.sleep(randint(1, 5))
        self.water_level += water_level
        print("Blub blub blub...")


if __name__ == "__main__":
    machine = CoffeeMachine()
    for i in range(0, 5):
        machine.make_coffee()
    machine.make_coffee()
    machine.add_water(70)
