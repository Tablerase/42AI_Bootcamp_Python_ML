import sys
import string

def text_analyzer(text = ''):
    """This function counts the number of upper characters, lower characters,
punctuation and spaces in a given text.

    Args:
        text (str): An input text if u want multiple lines user '\"\"\"'
    Examples:
        ```python
        text_analyze("Hello There")

        multiline = \"\"\"Hello
        There\"\"\"
        text_analyze(multiline)
        ```
    """
    printable : int = 0
    upper_letters: int = 0
    lower_letters: int = 0
    punctuation_letters: int = 0
    spaces_letters: int = 0
    try:
        assert isinstance(text, str), "argument is not a string"
        if len(text) == 0:
            text = input("What is the text to analyze ?\n")
        # else:
            # text = sys.argv[1]
        for c in text:
            if c.isprintable():
                printable += 1
            if c.isspace():
                spaces_letters += 1
            if c.islower():
                lower_letters += 1
            if c.isupper():
                upper_letters += 1
            if c in string.punctuation:
                punctuation_letters += 1

        print(
f'''The text contains {printable} printable character(s):
- {upper_letters} upper letter(s)
- {lower_letters} lower letter(s)
- {punctuation_letters} punctuation mark(s)
- {spaces_letters} space(s)'''
        )
            
    except AssertionError as e:
        print("AssertionError:", e, file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("Usage: python count.py (arg1)", file=sys.stderr)
    text_analyzer(sys.argv[1])
