"""Haar-Like Features model."""

import utils.integ_img as ii


class HaarFeature(object):
    """Haar-Like Features model."""

    def __init__(self, feature_type, position, width, height, threshold, polarity):
        """Initialize a Haar-Like Feature.

        Args:
            position (int, int): Top left corner where the feature begins (x,y)

            width (int): Width of the feature

            height (int): Height of the feature

            threshold (float): Feature threshold

            polarity (int): polarity of the feature -1 or 1

            feature_type (string): Type of features in reference.
                - 'type-2-y': 2 rectangles varying along the y axis; @ Type1
                - 'type-2-x': 2 rectangles varying along the x axis; @ Type2
                - 'type-3-y': 3 rectangles varying along the y axis; @ Type3
                - 'type-3-x': 3 rectangles varying along the x axis; @ Type4
                - 'type-4': 4 rectangles varying along x and y axis. @ Type5
                By default all features are extracted.

        References
        ----------
        .. [1] O. H. Jensen, "Implementing the Viola-Jones Face Detection Algorithm"
            https://github.com/Zhenye-Na/viola-jones-face-detection/blob/master/docs/Implementing%20the%20Viola-Jones%20Face%20Detection%20Algorithm.pdf

        """
        self.type = feature_type
        self.top_left = position
        self.bottom_right = (position[0] + width, position[1] + height)
        self.width = width
        self.height = height
        self.threshold = threshold
        if polarity:
            self.polarity = polarity
        else:
            self.polarity = -1
        self.weight = 1

    def _get_delta(self, itg_image):
        """Compute delta from Integral image within Haar-Like Features.

        Compare how close real scenario is to ideal case. The closer delta is
        to 1, the more likely we have found a Haar-Like Feature.

        Alogorithm:
            1. Calculate the sum of the 'white' pixel intensities: sum_{white}.
            2. Calculate the sum of the 'black' pixel intensities: sum_{black}.
            3. delta = sum_{black} - sum_{white}

        Args:
            itg_image (np.ndarray): integral image

        Returns:
            delta (float): score given by Haar-Like Features

        """
        # Initialize delta
        delta = 0

        # Size of feature window
        w = self.width
        h = self.height

        # Two columns/rows
        left_col = int(w / 2)
        up_row = int(h / 2)

        # Three columns/rows
        left_c = int(w / 3)
        middle_c = int(w / 3) * 2

        up_r = int(h / 3)
        middle_r = int(h / 3) * 2

        # Initial coordinates
        x1 = self.top_left[0]
        y1 = self.top_left[1]

        x2 = self.bottom_right[0]
        y2 = self.bottom_right[1]

        # compute delta
        if self.type == 'type-2-y':

            white_sum = ii.integrate(itg_image,
                                     [(x1, y1)], [(x2, y1 + left_col)])

            black_sum = ii.integrate(itg_image,
                                     [(x1, y1 + left_col + 1)], [(x2, y2)])

            delta = black_sum - white_sum

        elif self.type == 'type-2-x':
            black_sum = ii.integrate(itg_image,
                                     [(x1, y1)], [(x1 + up_row, y2)])

            white_sum = ii.integrate(itg_image,
                                     [(x1 + up_row + 1, y1)], [(x2, y2)])

            delta = black_sum - white_sum

        elif self.type == 'type-3-y':
            black_sum = ii.integrate(itg_image,
                                     [(x1, y1 + left_c)], [(x2, y1 + left_c + middle_c - 1)])

            # Here can use sth. like ii.integrate(ii, [(1, 0), (3, 3)], [(1, 2), (4, 5)])
            white_sum = ii.integrate(itg_image,
                                     [(x1, y1)], [(x2, y1 + left_c - 1)]) + ii.integrate(itg_image,
                                                                                         [(x1, y1 + left_c + middle_c)], [(x2, y2)])

            delta = black_sum - white_sum

        elif self.type == 'type-3-x':
            black_sum = ii.integrate(itg_image,
                                     [(x1 + up_r + 1, y1)], [(x1 + up_r + middle_r - 1, y2)])

            white_sum = ii.integrate(itg_image,
                                     [(x1, y1)], [(x1 + up_r, y2)]) + ii.integrate(itg_image,
                                                                                   [(x1 + up_r + middle_r, y1)], [(x2, y2)])

            delta = black_sum - white_sum

        elif self.type == 'type-4':
            # top left area
            first = ii.integrate(itg_image,
                                 self.top_left, [(int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2))])

            # top right area
            second = ii.integrate(itg_image,
                                  [(int(self.top_left[0] + self.width / 2), self.top_left[1])], [(self.bottom_right[0], int(self.top_left[1] + self.height / 2))])

            # bottom left area
            third = ii.integrate(itg_image,
                                 [(self.top_left[0], int(self.top_left[1] + self.height / 2))], [(int(self.top_left[0] + self.width / 2), self.bottom_right[1])])

            # bottom right area
            fourth = ii.integrate(itg_image,
                                  [(int(self.top_left[0] + self.width / 2), int(self.top_left[1] + self.height / 2))], [self.bottom_right])

            delta = second + third - (first + fourth)

        return delta

    def _get_vote(self, itg_image):
        """Get vote of this feature for given integral image.

        Args:
            itg_image (np.ndarray): Integral image

        Returns:
            1 iff this feature votes positively, otherwise -1

        """
        # Compute delta
        score = self._get_delta(itg_image)

        if score < self.polarity * self.threshold:
            return self.weight
        else:
            return -1 * self.weight
