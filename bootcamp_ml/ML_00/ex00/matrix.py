# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import sys


class Matrix:
    def __createMatrix(self, rows: int, columns: int) -> list[list[float]]:
        matrix: list[list[float]] = []
        for row in range(rows):
            new_row = []
            for column in range(columns):
                new_row.append(0)
            matrix.append(new_row)
        return matrix

    def __init__(self, data: list[list[float]] | tuple[int, int]):
        self.data: list[list[float]] = []
        self.shape: tuple[int, int] = None
        if isinstance(data, tuple):
            # Error
            if len(data) != 2:
                raise TypeError(
                    f"Invalid tuple size (rows, columns) format")
            # Define self.data
            rows = data[0]
            columns = data[1]
            self.data = self.__createMatrix(rows, columns)
            self.shape = (rows, columns)
        elif isinstance(data, list) and all(isinstance(row, list) for row in data):
            # Check same amounts of columns
            columns = len(data[0])
            if not all(len(row) == columns for row in data):
                raise ValueError(f"Columns of differents size: {data}")
            # Define self.data
            self.data = data
            rows = len(data)
            self.shape = (rows, columns)
        else:
            raise TypeError(
                "Only list of list[float] or shape tuple(int, int) supported")

    def __str__(self):
        return f"Matrix({self.data})"

    def T(self):
        self.shape = (self.shape[1], self.shape[0])
        new_data = self.__createMatrix(self.shape[0], self.shape[1])
        # fill new matrix
        for i, row in enumerate(self.data):
            for j, column_value in enumerate(row):
                new_data[j][i] = column_value
        self.data = new_data
        return self

    def __check_compatibility(self, other: "Matrix"):
        if not self.shape == other.shape:
            raise ValueError(f"Incompatible matrix operations")

    # add : only matrices of same dimensions.
    def __add__(self, other: "Matrix"):
        self.__check_compatibility(other)
        op_matrix = self.__createMatrix(other.shape[0], other.shape[1])
        for i, (row_self, row_other) in enumerate(zip(self.data, other.data)):
            for j in range(self.shape[1]):
                op_matrix[i][j] = row_self[j] + row_other[j]
        return Matrix(op_matrix)

    def __radd__(self, other: "Matrix"):
        return self.__add__(other)

    # sub : only matrices of same dimensions.
    def __sub__(self, other: "Matrix"):
        self.__check_compatibility(other)
        op_matrix = self.__createMatrix(other.shape[0], other.shape[1])
        for i, (row_self, row_other) in enumerate(zip(self.data, other.data)):
            for j in range(self.shape[1]):
                op_matrix[i][j] = row_self[j] - row_other[j]
        return Matrix(op_matrix)

    def __rsub__(self, other: "Matrix"):
        return self.__sub__(other)

    # div : only scalars.

    def __truediv__(self, scalar: float):
        if not isinstance(scalar, (int, float)):
            raise TypeError("Scalar must be a number")
        if scalar == 0:
            raise ZeroDivisionError("Division by zero")
        new_matrix = self.__createMatrix(self.shape[0], self.shape[1])
        for i, row in enumerate(self.data):
            for j in range(self.shape[1]):
                new_matrix[i][j] = row[j] / scalar
        return new_matrix

    def __rtruediv__(self, scalar: float):
        if not isinstance(scalar, (int, float)):
            raise TypeError("Scalar must be a number")
        new_matrix = self.__createMatrix(self.shape[0], self.shape[1])
        for i, row in enumerate(self.data):
            for j in range(self.shape[1]):
                if row[j] == 0:
                    raise ZeroDivisionError(
                        f"Division by zero at {j} {row} in {self.data}")
                new_matrix[i][j] = scalar / row[j]
        return Matrix(new_matrix)

    # mul : scalars, vectors and matrices , can have errors with vectors and matrices,
    # returns a Vector if we perform Matrix * Vector mutliplication.

    def __mul__(self, scalar: float):
        if not isinstance(scalar, (int, float)):
            raise TypeError("Scalar must be a number")
        new_matrix = self.__createMatrix(self.shape[0], self.shape[1])
        for i, row in enumerate(self.data):
            for j in range(self.shape[1]):
                new_matrix[i][j] = row[j] * scalar
        return new_matrix

    def __rmul__(self, scalar: float):
        return self.__mul__(self, scalar)

        # def __mul__(self, other: "Matrix"):
        # raise ValueError("Shape not compatible to do matrix multiplication")

        # def __rmul__(self, other: "Matrix"):
        # return self.__mul__(other)
