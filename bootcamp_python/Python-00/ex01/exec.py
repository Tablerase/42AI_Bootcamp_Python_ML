import sys

def exec(inputs):
    if len(inputs) < 2:
        print(
            'Usage: python exec.py arg1 ...'
        )
        return ""

    result = ' '.join(inputs[1:])[::-1].swapcase()
    print(result)

if __name__ == "__main__":
    exec(sys.argv)