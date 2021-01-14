import cv2 as cv2


def compute(leftImagePath, rightImagePath) -> list:
    left_image = cv2.imread(leftImagePath, 0)
    right_image = cv2.imread(rightImagePath, 0)

    left_matcher = cv2.StereoSGBM_create(
        numDisparities=32,
        blockSize=2
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    left_disp = left_matcher.compute(left_image, right_image)
    right_disp = right_matcher.compute(right_image, left_image)

    # create DisparityWLSFilter
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.5)
    filtered_disp = wls_filter.filter(left_disp, left_image, disparity_map_right=right_disp)
    return filtered_disp
