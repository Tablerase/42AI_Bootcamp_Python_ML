import sys


def ft_filter(function_to_apply, iterable):
    """Filter the result of function apply to all elements of the iterable.
    Args:
        function_to_apply: a function taking an iterable.
        iterable: an iterable object (list, tuple, iterator).
    Return:
        An iterable.
        None if the iterable can not be used by the function.
    """
    try:
        result = []
        for item in iterable:
            if function_to_apply(item):
                result.append(item)
                yield item
        return result

    except:
        return None


if __name__ == "__main__":
    x = [1, 2, 3, 4, 5]
    # Example 2:
    print(ft_filter(lambda dum: not (dum % 2), x))
    # Output:
    # <generator object ft_filter at 0x7f709c777d00> # The adress will be different
    print(list(ft_filter(lambda dum: not (dum % 2), x)))
    # Output:
    # [2, 4]
