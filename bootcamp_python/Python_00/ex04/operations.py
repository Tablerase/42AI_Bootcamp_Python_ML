import sys

def sum(arg1, arg2):
    try:
        sum = arg1 + arg2
        return sum
    except Exception as e:
        return f"ERROR: {e}"

def diff(arg1, arg2):
    try:
        result = arg1 - arg2
        return result
    except Exception as e:
        return f"ERROR: {e}"

def prod(arg1, arg2):
    try:
        result = arg1 * arg2
        return result
    except Exception as e:
        return f"ERROR: {e}"

def quotient(arg1, arg2):
    try:
        result = arg1 / arg2
        return result
    except Exception as e:
        return f"ERROR: {e}"

def remainder(arg1, arg2):
    try:
        result = arg1 % arg2
        return result
    except Exception as e:
        return f"ERROR: {e}"

def operations(arg1: int, arg2: int):
    """Do the following basic operations

    Infos:
        Sum: A+B
        Difference: A-B
        Product: A*B
        Quotient: A/B
        Remainder: A%B

    Args:
        arg1 (int): value 1
        arg2 (int): value 2
    Examples:
    ```python
        operations(10,3)
        Sum: 13
        Difference: 7
        Product: 30
        Quotient: 3.3333...
        Remainder: 1

        operations(1,0)
        Sum: 1
        Difference: 1
        Product: 0
        Quotient: ERROR (division by zero)
        Remainder: ERROR (modulo by zero
    ```
    """
    # • If an operation is impossible, print an error message instead of a numerical result.
    print(f"Sum:\t\t{sum(arg1, arg2)}")
    print(f"Difference:\t{diff(arg1, arg2)}")
    print(f"Product:\t{prod(arg1, arg2)}")
    print(f"Quotient:\t{quotient(arg1, arg2)}")
    print(f"Remainder:\t{remainder(arg1, arg2)}")

    return

if __name__ == "__main__":
    # If more or less than two arguments are provided or if one of the arguments is not
    # an integer, print an error message.
    # If no argument is provided, do nothing or print an usage.
    if len(sys.argv) == 1 or len(sys.argv) != 3:
        print('Usage: python operations.py arg1 arg2', file=sys.stderr)
    
        
    arg1 : int = int(sys.argv[1])
    arg2 : int = int(sys.argv[2])
    try:
        assert isinstance(arg1, int), 'only int'
        assert isinstance(arg2, int), 'only int'
        operations(arg1, arg2)
    except AssertionError as e:
        print("AssertionError:", e, file=sys.stderr)