import sys
from typing import List


class Evaluator:
    def zip_evaluate(coefs: List[float], words: List[str]) -> float:
        if not len(coefs) == len(words):
            return -1
        result = sum(a * len(b) for a, b in zip(coefs, words))
        return result

    def enumerate_evaluate(coefs: List[float], words: List[str]) -> float:
        if not len(coefs) == len(words):
            return -1
        result = sum(
            coefs[i] * len(word) for i, word in enumerate(words))
        return result


if __name__ == "__main__":
    words = ["Le", "Lorem", "Ipsum", "est", "simple"]
    coefs = [1.0, 2.0, 1.0, 4.0, 0.5]
    result = Evaluator.zip_evaluate(coefs, words)
    print(result)

    print(f"{'':_^60}")

    result = Evaluator.enumerate_evaluate(coefs, words)
    print(result)

    print(f"{'':_^60}")

    words = ["Le", "Lorem", "Ipsum", "n’", "est", "pas", "simple"]
    coefs = [0.0, -1.0, 1.0, -12.0, 0.0, 42.42]
    result = Evaluator.enumerate_evaluate(coefs, words)
    print(result)
