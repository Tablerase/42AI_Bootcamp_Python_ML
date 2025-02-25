import sys
from matrix import Matrix, Vector

if __name__ == "__main__":
    m_tuple = Matrix((3, 3))
    print("Test tuple", m_tuple.shape, m_tuple)
    try:
        m_matrix = Matrix([[0, 1], [1]])
    except Exception as e:
        print(e)
    m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    print("Shape", m1.shape)
    # Output:
    # (3, 2)
    m1.T()
    print("Post-transpose:", m1)
    # Output:
    # Matrix([[0., 2., 4.], [1., 3., 5.]])
    print("Post-tranpose shape:", m1.T().shape)
    # Output:
    # (2, 3)

    m1 = Matrix([
        [5, 2],
        [0, 1],
        [1, 9]
    ])
    m2 = Matrix([
        [2, 3],
        [4, 1],
        [0, 2]
    ])
    res = m1 + m2
    print(m1, m2, "Addition m1 + m2", res)
    # Output
    # Matrix([[7, 5], [4, 2], [1, 11]])
    m1 = Matrix([
        [-42],
        [27],
        [-3]
    ])
    res = m1 / 3
    print(m1, "Division m1 / 3:", res)
    # Output
    # Matrix([[-14.0], [9.0], [-1.0]])

    try:
        v1 = Vector([[2, 1, 3]])
        print(v1)
        v2 = Vector([[2, 4], [2, 2]])
        # Output error
    except Exception as e:
        print(e)

    v1 = Vector([[1, 3, -5]])
    v2 = Vector([[4], [-2], [-1]])
    res = v1.dot(v2)
    print(v1, v2, "Dot product v1.v2: ", res)

    m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
                 [0.0, 2.0, 4.0, 6.0]])
    m2 = Matrix([[0.0, 1.0],
                 [2.0, 3.0],
                 [4.0, 5.0],
                 [6.0, 7.0]])
    res = m1 * m2
    print(m1, m2, "m1 * m2", res)
    # Output:
    # Matrix([[28., 34.], [56., 68.]])

    m1 = Matrix([[0.0, 1.0, 2.0],
                 [0.0, 2.0, 4.0]])
    v1 = Vector([[1], [2], [3]])
    res = m1 * v1
    print(m1, v1, "m1 x v1: ", res)
    # Output:
    # Matrix([[8], [16]])
    # Or: Vector([[8], [16]

    v1 = Vector([[1], [2], [3]])
    v2 = Vector([[2], [4], [8]])
    res = v1 + v2
    print(v1, v2, "v1 + v2: ", res)
    # Output:
    # Vector([[3],[6],[11]])
