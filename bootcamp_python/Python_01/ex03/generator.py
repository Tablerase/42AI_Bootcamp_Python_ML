import sys
from random import random
from typing import List


def generator(text: str, sep: str = " ", option=None):
    """
    Splits the text according to sep value and yields the substrings.
    """
    allowed_options = {
        'shuffle': lambda x: sorted(x, key=lambda k: random()),
        'unique': lambda x: list(dict.fromkeys(x)),
        'ordered': lambda x: sorted(x)
    }
    try:
        if not isinstance(text, (str)):
            raise TypeError()
        if not option in allowed_options.keys() and option is not None:
            raise ValueError()
        # Separation
        splitted_text = text.split(sep)

        # Options
        if option is not None:
            # print(splitted_text)
            func_action = allowed_options.get(option, splitted_text)
            splitted_text = func_action(splitted_text)
            # print(splitted_text)

        for word in splitted_text:
            yield word

    except Exception:
        # return e
        print("ERROR")
        return


if __name__ == "__main__":
    text = "Le Lorem Ipsum est simplement du faux texte."
    for word in generator(text):
        print(word)

    print(f"{'':_^60}")

    for word in generator(text, sep=' ', option='shuffle'):
        print(word)

    print(f"{'':_^60}")
    for word in generator(text, sep=' ', option='ordered'):
        print(word)

    print(f"{'':_^60}")
    text = "Lorem Ipsum Lorem Ipsum"
    for word in generator(text, sep=' ', option='unique'):
        print(word)

    print(f"{'':_^60}")
    for word in generator(text, sep=' ', option='oifdksajfdkkajrdered'):
        print(word)
