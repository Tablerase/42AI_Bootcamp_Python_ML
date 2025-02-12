import sys


def what_are_the_vars(*args, **kwargs):
    """Create vars from inputs and inject them into a ObjectC

    Return:
        objectC with all the vars
        None is case of error
    """
    result = ObjectC()
    try:
        # Add args first:
        for i, arg in enumerate(args):
            setattr(result, f'var_{i}', arg)
        # Add kwarg:
        for key, arg in kwargs.items():
            if hasattr(result, key):
                raise AttributeError(f"Attribute already exist: {key, arg}")
            setattr(result, key, arg)

        return result
    except AttributeError as e:
        # print(e)
        return None


class ObjectC(object):
    def __init__(self, *args, **kwargs):
        pass


def doom_printer(obj):
    if obj is None:
        print("ERROR")
        print("end")
        return
    for attr in dir(obj):
        if attr[0] != '_':
            value = getattr(obj, attr)
            print("{}: {}".format(attr, value))
    print("end")


if __name__ == "__main__":
    obj = what_are_the_vars(7)
    doom_printer(obj)
    obj = what_are_the_vars(None, [])
    doom_printer(obj)
    obj = what_are_the_vars("ft_lol", "Hi")
    doom_printer(obj)
    obj = what_are_the_vars()
    doom_printer(obj)
    obj = what_are_the_vars(12, "Yes", [0, 0, 0], a=10, hello="world")
    doom_printer(obj)
    obj = what_are_the_vars(42, a=10, var_0="world")
    doom_printer(obj)
    obj = what_are_the_vars(42, "Yes", a=10, var_2="world")
    doom_printer(obj)
