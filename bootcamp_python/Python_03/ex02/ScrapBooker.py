import numpy as np

"""
https://numpy.org/doc/2.2/user/absolute_beginners.html#creating-matrices
https://numpy.org/doc/2.2/user/quickstart.html#indexing-slicing-and-iterating

https://numpy.org/doc/2.2/user/how-to-index.html#create-subsets-of-larger-matrices
https://www.sharpsightlabs.com/blog/numpy-axes-explained/
https://numpy.org/doc/2.2/reference/generated/numpy.delete.html#numpy-delete
"""


class ScrapBooker:
    def crop(self, array: np.ndarray, dim: tuple, position=(0, 0)):
        """
        Crops the image as a rectangle via dim arguments (being the new height
        and width of the image) from the coordinates given by position arguments.

        Args:
            array: numpy.ndarray
            dim: tuple of 2 integers.
            position: tuple of 2 integers.
        Returns:
            new_arr: the cropped numpy.ndarray.
            None: (if the combination of parameters is not possible).
        Raises:
            This function should not raise any Exception.
        """
        try:
            height, width = dim
            start_y, start_x = position
            if (start_y + height > array.shape[0]) or (start_x + width > array.shape[1]):
                raise ValueError(
                    f"Combination of parameters not possible:\narray dim:{array.ndim} array shape:{array.shape} parameters: {dim} {position}")

            new_arr = array[start_y:start_y + height, start_x:start_x + width]
            return new_arr
        except Exception as e:
            print(e)
            return None

    def thin(self, array: np.ndarray, n, axis):
        """
        Deletes every n-th line pixels along the specified axis (0: vertical, 1: horizontal)
        Args:
            array: numpy.ndarray.
            n: non null positive integer lower than the number of row/column of the array
            (depending of axis value).
            axis: positive non null integer.
        Returns:
            new_arr: thined numpy.ndarray.
            None: (if the combination of parameters is not possible).
        Raises:
            This function should not raise any Exception.
        """
        try:
            if n <= 0 or axis not in [0, 1]:
                raise ValueError(
                    "parameters: positive non null only, axis 1 or 0")
            if not n < array.shape[axis]:
                raise ValueError(f"Shape: {array.shape} , n {n} is too big")
            indices = np.arange(n - 1, array.shape[axis], n)
            new_arr = np.delete(array, indices, axis=axis)
            return new_arr
        except Exception as e:
            print(e)
            return None

    def juxtapose(self, array: np.ndarray, n, axis):
        """
        Juxtaposes n copies of the image along the specified axis.
        Args:
            array: numpy.ndarray.
            n: positive non null integer.
            axis: integer of value 0 or 1.
        Returns:
            new_arr: juxtaposed numpy.ndarray.
            None: (if the combination of parameters is not possible).
        Raises:
            This function should not raise any Exception.
        """
        try:
            if n <= 0 or axis not in [0, 1]:
                raise ValueError(
                    "parameters: positive non null only, axis 1 or 0")
            new_arr = np.tile(array, (n, 1) if axis == 0 else (1, n))
            return new_arr
        except Exception as e:
            print(e)
            return None

    def mosaic(self, array, dim):
        """
        Makes a grid with multiple copies of the array. The dim argument specifies
        the number of repetition along each dimensions.
        Args:
        -----
        array: numpy.ndarray.
        dim: tuple of 2 integers.
        Return:
        -------
        new_arr: mosaic numpy.ndarray.
        None (combinaison of parameters not compatible)
        """


if __name__ == "__main__":
    spb = ScrapBooker()
    print(f"{'Crop':_^60}")
    arr1 = np.arange(0, 25).reshape(5, 5)
    print(arr1)
    res = spb.crop(arr1, (3, 1), (1, 0))
    print(res)
    # Output :
    # array([[ 5],
    # [10],
    # [15]])

    print(f"{'Thin':_^60}")
    #! Question example validity
    #! arr2 = np.array("A B C D E F G H I".split() * 6).reshape(-1, 9)

    lst_arr2 = "A B C D E F G H I J K".split()
    arr2 = np.array(lst_arr2 * 11).reshape(-1, 11)
    print(arr2)
    #! res = spb.thin(arr2, 3, 0)
    res = spb.thin(arr2, 3, 1)
    print(res)
    # Output :
    # array([[’A’, ’B’, ’D’, ’E’, ’G’, ’H’, ’J’, ’K’],
    # [’A’, ’B’, ’D’, ’E’, ’G’, ’H’, ’J’, ’K’],
    # [’A’, ’B’, ’D’, ’E’, ’G’, ’H’, ’J’, ’K’],
    # [’A’, ’B’, ’D’, ’E’, ’G’, ’H’, ’J’, ’K’],
    # [’A’, ’B’, ’D’, ’E’, ’G’, ’H’, ’J’, ’K’],
    # [’A’, ’B’, ’D’, ’E’, ’G’, ’H’, ’J’, ’K’]], dtype=’<U1’)

    print(f"{'Juxtapose':_^60}")
    arr3 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    print(arr3)
    res = spb.juxtapose(arr3, 3, 1)
    print(res)
    # Output :
    # array([[1, 2, 3, 1, 2, 3, 1, 2, 3],
    # [1, 2, 3, 1, 2, 3, 1, 2, 3],
    # [1, 2, 3, 1, 2, 3, 1, 2, 3]])

    #! No mozaic example
