import cv2
import imutils
import numpy as np


class MotionDetector:
    # =======
    # DUNDERS
    # =======
    def __init__(self, weigth: float = 0.5) -> None:
        self.weigth = weigth
        self.__bg = None

    def __repr__(self) -> str:
        """Object representation when calling :func:`repr` or :class:`str`."""
        return f"{self.qualname}(weigth={self.weigth})"

    # =======
    # GETTERS
    # =======
    @property
    def clsname(self) -> str:
        """Class name (:class:`str`, read-only).

        This attribute is a wrapper around ``self.__class__.__name__``,
        """
        return self.__class__.__name__

    @property
    def qualname(self) -> str:
        """Fully qualified name (:class:`str`, read-only)

        This attribute represents the import name and includes package, sub-package(s)
        (if any), and the class name.
        """
        pkg, subpkg, *_ = __name__.split(".")
        return f"{pkg}.{subpkg}.{self.clsname}"

    # ==============
    # PUBLIC METHODS
    # ==============
    def update(self, img: np.ndarray) -> None:
        if self.__bg is None:
            self.__bg = img.copy().astype(float)
            return
        cv2.accumulateWeigthed(img, self.__bg, self.weigth)

    def detect(self, img: np.ndarray, threshold: int = 25) -> None:
        # pixels above threshold are set to 255
        img_delta = cv2.absdiff(self.__bg.astype("uint8"), img)
        img_binary = cv2.threshold(delta, threshold, 255, cv2.THRESH_BINARY)

        # smooth images to remove isolated peaks
        img_binary = cv2.errode(img_binary, None, iterations=2)
        img_binary = cv2.dilate(img_binary, None, iterations=2)

        # find contour
        contours = cv2.findContours(
            img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = imutils.grab_contour(contours)

        # if no contours were found, return None
        if len(contours) == 0:
            return None

        # initial box corners
        (x_min, y_min) = (np.inf, np.inf)
        (x_max, y_max) = (-np.inf, -np.inf)
        for contour in contours:
            # compute the bounding box of the contour and use it to
            # update the minimum and maximum bounding box regions
            (x, y, w, h) = cv2.boundingRect(contour)
            (x_min, y_min) = (min(x_min, x), min(y_min, y))
            (x_max, y_max) = (max(x_max, x + w), max(y_max, y + h))

        # otherwise, return a tuple of the thresholded image along
        # with bounding box
        return {
            "img_binary": img_binary,
            "coordinates": {
                "x": (x_min, x_max),
                "y": (y_min, y_max),
            },
        }
