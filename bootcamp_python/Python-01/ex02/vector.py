import sys
# typing deprecate after 3.9
from typing import List, Tuple

"""

Column vector
---
|A1|
|A2|
|⠇|
|n|
---

Row vector
-----------
|B1 B2 ... n|
-----------

Transpose = switch (Row <-> Column)
[A1, A2, ..., n]

Dot product
---------
|A1 * B1|
|A2 * B2|
|...*...| == A1 * B1 + A2 * B2 + ... * ... + An * Bn
|An * Bn|
---------

[ 📹 Youtube - Vectors - Essence of linear algebra](https://youtu.be/fNk_zzaMoSs?si=nukJqaKyoSkP-tFA)
https://en.wikipedia.org/wiki/Row_and_column_vectors
https://en.wikipedia.org/wiki/Transpose

"""


class Vector:
    def __init__(self, values: List[List[float]]):
        self.shape: Tuple[int, int]
        self.values: List[List[float]] = []
        # Column
        if isinstance(values, list) and len(values) > 1:
            for coord in values:
                if len(coord) > 1:
                    raise ValueError(
                        "Column should have only one coord per list")
            self.shape = (len(values), 1)
        # row
        elif isinstance(values, list) and len(values) == 1:
            self.shape = (1, len(values[0]))
        else:
            raise TypeError(f"Invalid type: {values.__dict__}")
        self.values = values

    def __str__(self):
        return f"Vector({self.values})"

    def check_shape(self, vector):
        if not isinstance(vector, Vector):
            raise TypeError("Operation requieres another vector instance")
        if not self.shape == vector.shape:
            raise ValueError(
                f"Incompatible shapes: {self.shape} and {vector.shape}")

    def dot(self, vector: 'Vector') -> float:
        """Dot product between vectors"""
        self.check_shape(vector)
        # Rows [[a1, a2, ...]] . [[b1, b2, ...]]
        if self.shape[0] == 1:
            result = sum(
                a * b for a, b in zip(self.values[0], vector.values[0]))
        # Column [[a1], [a2], ...] . [[b1], [b2], ...]
        else:
            result = sum(a[0] * b[0]
                         for a, b in zip(self.values, vector.values))
        return result

    def T(self) -> 'Vector':
        """Transpose vector
        Row -> Column: [[a1, a2, ...]] -> [[a1], [a2], ...]]
        Column -> Row: [[a1], [a2], ...] -> [[a1, a2, ...]]
        """
        # Row to Column
        if self.shape[0] == 1:
            new_values = [[x] for x in self.values[0]]
        # Column to Row
        else:
            new_values = [[x[0] for x in self.values]]
        return Vector(new_values)

    def __add__(self, other) -> 'Vector':
        """Add a vector with another vector"""
        self.check_shape(other)
        # For row vectors: [[a1, a2, ...]] + [[b1, b2, ...]]
        if self.shape[0] == 1:
            new_values = [a + b for a, b in zip(self.values, other.values)]
        # For column vectors: [[a1], [a2], ...] + [[b1], [b2], ...]
        else:
            new_values = [[a[0] + b[0]]
                          for a, b in zip(self.values, other.values)]
        return Vector(new_values)

    def __radd__(self, other) -> 'Vector':
        """Add a vector with another vector"""
        return self.__add__(other)

    def __sub__(self, other) -> 'Vector':
        """Substract vector by a vector"""
        self.check_shape(other)
        # For row vectors: [a1, a2, ...] - [b1, b2, ...]
        if self.shape[0] == 1:
            new_values = [a - b for a, b in zip(self.values, other.values)]
        # For column vectors: [[a1], [a2], ...] - [[b1], [b2], ...]
        else:
            new_values = [[a[0] - b[0]]
                          for a, b in zip(self.values, other.values)]
        return Vector(new_values)

    def __rsub__(self, other) -> 'Vector':
        """Substract vector by a vector"""
        return self.__sub__(other)

    def __mul__(self, scalar: float) -> 'Vector':
        """Multiply vector by scalar (vector * scalar)"""
        if not isinstance(scalar, (int, float)):
            raise TypeError("Scalar must be a number")
        # Row vector
        if self.shape[0] == 1:
            new_values = [x * scalar for x in self.values]
        # Column vector
        else:
            new_values = [[x[0] * scalar] for x in self.values]
        return Vector(new_values)

    def __rmul__(self, scalar: float) -> 'Vector':
        """Multiply scalar by vector (scalar * vector)"""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> 'Vector':
        """Divide vector by scalar (vector / scalar)"""
        if not isinstance(scalar, (int, float)):
            raise TypeError("Scalar must be a number")
        if scalar == 0:
            raise ZeroDivisionError("Division by zero")
        # Row vector
        if self.shape[0] == 1:
            new_values = [[x / scalar for x in row] for row in self.values]
        # Column vector
        else:
            new_values = [[x[0] / scalar] for x in self.values]
        return Vector(new_values)

    def __rtruediv__(self, scalar: float) -> 'Vector':
        """Handle scalar / vector - not supported"""
        raise NotImplementedError(
            "Division of a scalar by a Vector is not defined")
