import sys
from typing import List

"""
Make a program that takes a string as argument and encodes it into Morse code.

• The program supports space and alphanumeric characters
• An alphanumeric character is represented by dots . and dashes -:
• A space character is represented by a slash /
• Complete morse characters are separated by a single space

If more than one argument is provided, merge them into a single string with each
argument separated by a single space character.

If no argument is provided, do nothing or print an usage.

https://morsecode.world/international/morse2.html
"""

MORSE_CODE = {
    'a': '.-',
    'b': '-...',
    'c': '-.-.',
    'd': '-..',
    'e': '.',
    'f': '..-.',
    'g': '--.',
    'h': '....',
    'i': '..',
    'j': '.---',
    'k': '-.-',
    'l': '.-..',
    'm': '--',
    'n': '-.',
    'o': '---',
    'p': '.--.',
    'q': '--.-',
    'r': '.-.',
    's': '...',
    't': '-',
    'u': '..-',
    'v': '...-',
    'w': '.--',
    'x': '-..-',
    'y': '-.--',
    'z': '--..',
    '1': '.----',
    '2': '..---',
    '3': '...--',
    '4': '....-',
    '5': '.....',
    '6': '-....',
    '7': '--...',
    '8': '---..',
    '9': '----.',
    '0': '-----',
    ' ': '/'
}

def sos(words: List[str]):
    # print(words)
    result_words : List[str] = []
    for word in words:
        # print(word)
        morse_word : str = ""
        for c in word.lower():
            # print(MORSE_CODE[c])
            morse_word = morse_word + MORSE_CODE[c] + ' '
        result_words.append(morse_word)
        
    print(''.join(result_words)[:-1])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sos.py string1 (string2) (...)", file=sys.stderr)
        exit(1)
    try:
        input_strings : List[str] = sys.argv[1::]
        print(input_strings)
        # result_string = ' '.join(input_strings)
        # print(result_string)
        for word in input_strings:
            for c in word:
                if c.isalnum() == True or c.isspace() == True:
                    continue
                else:
                    raise Exception("Non alphanumerical values not supported")
        sos(input_strings)
    except Exception as e:
        print(e, file=sys.stderr)
