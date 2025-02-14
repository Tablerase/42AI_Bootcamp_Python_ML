import sys


def ft_reduce(function_to_apply, iterable):
    """Apply function of two arguments cumulatively.
    Args:
        function_to_apply: a function taking an iterable.
        iterable: an iterable object (list, tuple, iterator).
    Return:
        A value, of same type of elements in the iterable parameter.
        None if the iterable can not be used by the function.
    """
    try:
        iterator = iter(iterable)
        value = next(iterator)

        for element in iterator:
            # print(value, element, iterator)
            value = function_to_apply(value, element)
        return value
    except:
        return None


if __name__ == "__main__":
    lst = ['H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
    print(ft_reduce(lambda u, v: u + v, lst))
    # Output:
    # "Hello world"
