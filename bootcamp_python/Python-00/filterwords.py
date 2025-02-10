import sys
from typing import List
import string

"""
Make a program that takes a string S and an integer N as argument and prints the
list of words in S that contains more than N non-punctuation characters.

• Words are separated from each other by space characters
• Punctuation symbols must be removed from the printed list: they are neither part
of a word nor a separator
• The program must contain at least one list comprehension expression.

If the number of argument is different from 2, or if the type of any argument is invalid,
the program prints an error message.
"""

def filter_words(words: str, min_len: int):
    print("Words", words)
    cleaned_words : str = ''.join([c for c in words if c not in string.punctuation])
    print("Clean words", cleaned_words)
    words_list : List[str] = cleaned_words.split()
    print("Split words_list", words_list)
    result_list : List[str] = [word for word in words_list if len(word) > min_len]
    print("Result:", result_list)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filterwords.py string_arg integer_arg", file=sys.stderr)
        exit(1)
    
    try:
        input_str = str(sys.argv[1])
        input_number = int(sys.argv[2])
        assert isinstance(input_str, str), "Arg1 should be a string"
        assert isinstance(input_number, int), "Arg2 should be an integer"
        assert input_number > 0, "Arg2 should be positive"
        filter_words(input_str, input_number)
    except ValueError as e:
        print("ValueError: ", e, file=sys.stderr)
    except AssertionError as e:
        print("AssertionError: ", e, file=sys.stderr)
    