from ex01.ImageProcessor import ImageProcessor
import numpy as np
import matplotlib as plt

"""
To resolve import:

export PYTHONPATH=/home/rcutte/Desktop/piscine_python_ml/bootcamp_python/Python-03

Then:
py ColorFilter.py
"""

"""
http://www.lighthouse3d.com/tutorials/glsl-12-tutorial/toon-shader-version-ii/
https://www.konlabs.com/articles_data/cel_shading/index.html
"""


class ColorFilter:
    def invert(self, array: np.ndarray):
        """
        Inverts the color of the image received as a numpy array.

        Args:
            array: numpy.ndarray corresponding to the image.
        Return:
            array: numpy.ndarray corresponding to the transformed image.
            None: otherwise.
        Raises:
            This function should not raise any Exception.
        Restrictions:
            ◦ Authorized functions: .copy
                https://numpy.org/doc/2.2/reference/generated/numpy.copy.html#numpy.copy
            ◦ Authorized operators: +,-,=
        """
        try:
            res = array.copy()
            return -res
        except Exception as e:
            print(e)
            return None

    def to_blue(self, array: np.ndarray):
        """
        Applies a blue filter to the image received as a numpy array.

        Args:
            array: numpy.ndarray corresponding to the image.
        Return:
            array: numpy.ndarray corresponding to the transformed image.
            None: otherwise.
        Raises:
            This function should not raise any Exception.
        Restrictions:
            ◦ Authorized functions: .copy, .zeros,.shape,.dstack.
            ◦ Authorized operators: =
        """
        try:
            height, width, _ = array.shape
            red = np.zeros((height, width), dtype=array.dtype)
            green = np.zeros((height, width), dtype=array.dtype)
            blue = array[:, :, 2]
            res = np.dstack((red, green, blue))
            return res
        except Exception as e:
            print(e)
            return None

    def to_green(self, array: np.ndarray):
        """
        Applies a green filter to the image received as a numpy array.

        Args:
            array: numpy.ndarray corresponding to the image.
        Return:
            array: numpy.ndarray corresponding to the transformed image.
            None: otherwise.
        Raises:
            This function should not raise any Exception.
        Restrictions:
            ◦ Authorized functions: .copy
            ◦ Authorized operators: =
        """
        try:
            arr = np.copy(array)
            # red to 0
            arr[:, :, 0] = 0
            # green = array[:, :, 1]
            # blue to 0
            arr[:, :, 2] = 0
            return arr
        except Exception as e:
            print(e)
            return None

    def to_red(self, array: np.ndarray):
        """
        Applies a red filter to the image received as a numpy array.

        Args:
            array: numpy.ndarray corresponding to the image.
        Return:
            array: numpy.ndarray corresponding to the transformed image.
            None: otherwise.
        Raises:
            This function should not raise any Exception.
        Restrictions:
            ◦ Authorized functions: .copy, .to_red,.to_blue.
            ◦ Authorized operators: -,+, =.
        """
        try:
            arr = np.copy(array)
            # arr[:, :, 0] = 0
            # green to 0
            arr[:, :, 1] = 0
            # blue to 0
            arr[:, :, 2] = 0
            return arr
        except Exception as e:
            print(e)
            return None

    def to_celluloid(self, array):
        """
        Applies a celluloid filter to the image received as a numpy array.
        Celluloid filter must display at least four thresholds of shades.
        Be careful! You are not asked to apply black contour on the object,
        you only have to work on the shades of your images.

        Remarks:
            celluloid filter is also known as cel-shading or toon-shading.
        Args:
            array: numpy.ndarray corresponding to the image.
        Return:
            array: numpy.ndarray corresponding to the transformed image.
            None: otherwise.
        Raises:
            This function should not raise any Exception.
        Restrictions:
            ◦ Authorized functions: .copy, .arange,.linspace, .min, .max.
            ◦ Authorized operators: =, <=, >, & (or and).
        """
        try:
            arr = np.copy(array)
            shades = np.array([0, 35, 75, 150, 235])

            # For each color channel
            for channel in range(3):
                # For each threshold
                channel_data = arr[:, :, channel]
                for i in range(len(shades)-1):
                    mask = (channel_data > shades[i]) & (
                        channel_data <= shades[i+1])
                    channel_data[mask] = shades[i+1]
                arr[:, :, channel] = channel_data

            return arr
        except Exception as e:
            print(e)
            return None

    #! gray scale not done


if __name__ == "__main__":
    imp = ImageProcessor()
    print(f"{'Image Loading':_^60}")
    arr = imp.load('../ex01/42AI.png')
    imp.display(arr)

    clf = ColorFilter()
    print(f"{'Invert':_^60}")
    invert = clf.invert(arr)
    imp.display(invert)

    print(f"{'Blue':_^60}")
    blue = clf.to_blue(arr)
    imp.display(blue)

    print(f"{'Green':_^60}")
    green = clf.to_green(arr)
    imp.display(green)

    print(f"{'Red':_^60}")
    red = clf.to_red(arr)
    imp.display(red)

    print(f"{'Celluloid':_^60}")
    celluloid = clf.to_celluloid(arr)
    imp.display(celluloid)
