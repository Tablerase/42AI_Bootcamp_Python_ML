import numpy as np

"""
https://numpy.org/doc/2.2/
https://numpy.org/doc/2.2/reference/routines.array-creation.html
"""


class NumpyCreator:
    def from_list(self, lst, dtype=None):
        try:
            if not isinstance(lst, list):
                raise ValueError("Only list accepted")
            res = np.array(lst, dtype=dtype)
            return res
        except:
            return None

    def from_tuple(self, tpl, dtype=None):
        try:
            if not isinstance(tpl, tuple):
                raise ValueError("Only tuple")
            res = np.array(tpl, dtype=dtype)
            return res
        except:
            return None

    def from_iterable(self, itr, dtype=None):
        try:
            if not hasattr(itr, '__iter__'):
                raise ValueError("Only iterator")
            res = np.fromiter(itr, dtype=dtype)
            return res
        except:
            return None

    def from_shape(self, shape, value=0, dtype=None):
        try:
            if not isinstance(shape, tuple):
                raise ValueError("Shape is a tuple")
            res = np.full(shape, value, dtype=dtype)
            return res
        except:
            return None

    def random(self, shape, dtype=None):
        try:
            if not isinstance(shape, tuple):
                raise ValueError("Shape is a tuple")
            res = np.random.rand(*shape).astype(dtype)
            return res
        except:
            return None

    def identity(self, n, dtype=None):
        try:
            res = np.identity(n, dtype=dtype)
            return res
        except:
            return None


if __name__ == "__main__":
    npc = NumpyCreator()
    print(f"{'From List':_^60}")
    res = npc.from_list([[1, 2, 3], [6, 3, 4]])
    print(res)
    # Output :
    # array([[1, 2, 3],
    # [6, 3, 4]])
    res = npc.from_list([[1, 2, 3], [6, 4]])
    print(res)
    # Output :
    # None
    res = npc.from_list([[1, 2, 3], ['a', 'b', 'c'], [6, 4, 7]])
    print(res, res.dtype)
    # Output :
    # array([[’1’,’2’,’3’],
    # [’a’,’b’,’c’],
    # [’6’,’4’,’7’], dtype=’<U21’])
    res = npc.from_list(((1, 2), (3, 4)))
    print(res)
    # Output :
    # None

    print(f"{'From Tuple':_^60}")
    res = npc.from_tuple(("a", "b", "c"))
    print(res)
    # Output :
    # array([’a’, ’b’, ’c’])
    res = npc.from_tuple(["a", "b", "c"])
    print(res)
    # Output :
    # None

    print(f"{'From Iter':_^60}")
    res = npc.from_iterable(range(5))
    print(res)
    # Output :
    # array([0, 1, 2, 3, 4])

    print(f"{'From Shape':_^60}")
    shape = (3, 5)
    res = npc.from_shape(shape)
    print(res)
    # Output :
    # array([[0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0]])

    print(f"{'Random':_^60}")
    res = npc.random(shape)
    print(res)
    # Output :
    # array([[0.57055863, 0.23519999, 0.56209311, 0.79231567, 0.213768 ],
    # [0.39608366, 0.18632147, 0.80054602, 0.44905766, 0.81313615],
    # [0.79585328, 0.00660962, 0.92910958, 0.9905421 , 0.05244791]])

    print(f"{'Identity':_^60}")
    res = npc.identity(4)
    print(res)
    # Output :
    # array([[1., 0., 0., 0.],
    # [0., 1., 0., 0.],
    # [0., 0., 1., 0.],
    # [0., 0., 0., 1.]])
