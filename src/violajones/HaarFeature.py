"""Haar-Like Features model."""

import utils.integ_img as ii


class HaarFeature(object):
    """Haar-Like Features model."""

    def __init__(self, feature_type, position, width, height, threshold, polarity):
        """Initialize a Haar-Like Feature.

        Args:
        feature_type (string): Type of features
            - 'type-2-y': 2 rectangles varying along the y axis; @ Type1 in the reference
            - 'type-2-x': 2 rectangles varying along the x axis; @ Type2 in the reference
            - 'type-3-y': 3 rectangles varying along the y axis; @ Type3 in the reference
            - 'type-3-x': 3 rectangles varying along the x axis; @ Type4 in the reference
            - 'type-4': 4 rectangles varying along x and y axis. @ Type5 in the reference
            By default all features are extracted.

        References
        ----------
        .. [1] O. H. Jensen, "Implementing the Viola-Jones Face Detection Algorithm"
            https://github.com/Zhenye-Na/viola-jones-face-detection/blob/master/docs/Implementing%20the%20Viola-Jones%20Face%20Detection%20Algorithm.pdf

        position (int, int): Top left corner where the feature begins (x, y)

        width (int): Width of the feature

        height (int): Height of the feature

        threshold (float): Feature threshold

        polarity (int): polarity of the feature -1 or 1

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
        """Compute delta from Haar-Like Features.

        Compare how close rhe real scenario is to the ideal case

        Alogorithm:
            1. Calculate the sum of the 'white' pixel intensities: sum_{white}.
            2. Calculate the sum of the 'black' pixel intensities: sum_{black}.
            3. sum_{black} - sum_{white}

        The closer the value to 1, the more likely we have found a Haar-Like Feature.

        Args:
            itg_image (np.ndarray): integral image
        Returns:
            delta (float): score given by Haar-Like Features
        """
        # Initialize delta (score)
        delta = 0

        # # Size of feature window
        # w = self.width
        # h = self.height

        # # Two columns/rows
        # left_col = int(w / 2)
        # up_row = int(h / 2)

        # # Three columns/rows
        # left_c = int(w / 3)
        # middle_c = int(w / 3) * 2

        # up_r = int(h / 3)
        # middle_r = int(h / 3) * 2

        # # Initial coordinates
        # x1 = self.top_left[0]
        # y1 = self.top_left[1]

        # x2 = self.bottom_right[0]
        # y2 = self.bottom_right[1]

        # compute delta
        if self.type == 'type-2-y':

            white_sum = ii.integrate(itg_image, self.top_left, (self.top_left[0] + self.width, int(self.top_left[1] + self.height / 2)))

            black_sum = ii.integrate(itg_image, (self.top_left[0], int(self.top_left[1] + self.height / 2)), self.bottom_right)

            delta = black_sum - white_sum

        elif self.type == 'type-2-x':
            black_sum = ii.integrate(itg_image, self.top_left, (int(self.top_left[0] + self.width / 2), self.top_left[1] + self.height))

            white_sum = ii.integrate(itg_image, (int(self.top_left[0] + self.width / 2), self.top_left[1]), self.bottom_right)

            delta = black_sum - white_sum

        elif self.type == 'type-3-y':
            black_sum = ii.integrate(itg_image, self.top_left, (self.bottom_right[0], int(self.top_left[1] + self.height / 3)))

            white_sum = ii.integrate(itg_image, (self.top_left[0], int(self.top_left[1] + self.height / 3)), (self.bottom_right[0], int(self.top_left[1] + 2 * self.height / 3))) + ii.integrate(itg_image, (self.top_left[0], int(self.top_left[1] + 2 * self.height / 3)), self.bottom_right)

            delta = black_sum - white_sum

        elif self.type == 'type-3-x':

            black_sum = ii.integrate(itg_image, self.top_left, (int(self.top_left[0] + self.width / 3), self.top_left[1] + self.height))

            white_sum = ii.integrate(itg_image, (int(self.top_left[0] + self.width / 3), self.top_left[1]), (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1] + self.height)) + ii.integrate(itg_image, (int(self.top_left[0] + 2 * self.width / 3), self.top_left[1]), self.bottom_right)

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
            itg_image (np.ndarray): Integral image array
        Returns:
            1 iff this feature votes positively, otherwise -1
        """
        # Compute score
        score = self._get_delta(itg_image)

        if score < self.polarity * self.threshold:
            return self.weight
        else:
            return -1 * self.weight
