from geometry_classes import Point, Line
import matplotlib.pyplot as plt
import math
import numpy as np
import cv2


def adjust_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    maximum = accumulator[-1]
    clip_hist_percent = (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    return (alpha, beta)


def lines_intersect(img, l1, l2):
    slope_1, shift_1 = get_line_slope_shift(l1)
    slope_2, shift_2 = get_line_slope_shift(l2)

    if slope_1 == slope_2:
        return False

    x, y = intersect_point(l1, l2)

    x_max = img.shape[1]
    y_max = img.shape[0]

    if x >= 0 and x < x_max and y >= 0 and y < y_max:
        return True
    return False


def intersect_point(l1, l2):
    slope_1, shift_1 = get_line_slope_shift(l1)
    slope_2, shift_2 = get_line_slope_shift(l2)

    x = (shift_2 - shift_1)/(slope_1 - slope_2)
    y = (slope_1*x + shift_1)
    x = int(x)
    y = int(y)

    return (x, y)


def lesser_angle_between(l1, l2):
    dot_product = (l1[2] - l1[0])*(l2[2] - l2[0]) + \
        (l1[3] - l1[1])*(l2[3] - l2[1])
    length1 = math.sqrt((l1[2] - l1[0])*(l1[2] - l1[0]) +
                        (l1[3] - l1[1])*(l1[3] - l1[1]))
    length2 = math.sqrt((l2[2] - l2[0])*(l2[2] - l2[0]) +
                        (l2[3] - l2[1])*(l2[3] - l2[1]))
    if abs(abs(dot_product/(length1 * length2)) - 1) < 0.00000000001:
        return 0
    return math.acos(abs(dot_product/(length1 * length2)))


def four_points_from(lines):
    left_l = []
    right_l = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if lesser_angle_between(lines[i], lines[j]) < np.pi / 18 and lesser_angle_between(lines[j], [0, 0, 100, 0]) > np.pi / 30 and lesser_angle_between(lines[j], [0, 0, 0, 100]) > np.pi / 30:
                left_l = lines[i]
                right_l = lines[j]
                break
    k1 = (left_l[2] - left_l[0])/(left_l[3] - left_l[1])
    k2 = (right_l[2] - right_l[0])/(right_l[3] - right_l[1])
    b1 = left_l[0] - k1 * left_l[1]
    b2 = right_l[0] - k2 * right_l[1]

    max_y1 = max(left_l[1], right_l[1])
    max_y2 = max(left_l[3], right_l[3])
    points = [(int(k1*max_y1 + b1), max_y1),
              (int(k2*max_y1 + b2), max_y1),
              (int(k1*max_y2 + b1), max_y2),
              (int(k2*max_y2 + b2), max_y2)]
    return np.array(points, np.float32)


def rectangle_from(points):
    minx = min(points[0][0], points[1][0], points[2][0], points[3][0])
    miny = min(points[0][1], points[1][1], points[2][1], points[3][1])
    maxx = max(points[0][0], points[1][0], points[2][0], points[3][0])
    maxy = max(points[0][1], points[1][1], points[2][1], points[3][1])

    return np.array([(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)], np.float32)


def equal(l1, l2):
    length1 = math.sqrt((l1[2] - l1[0])*(l1[2] - l1[0]) +
                        (l1[3] - l1[1])*(l1[3] - l1[1]))
    length2 = math.sqrt((l2[2] - l2[0])*(l2[2] - l2[0]) +
                        (l2[3] - l2[1])*(l2[3] - l2[1]))

    product = (l1[2] - l1[0])*(l2[2] - l2[0]) + (l1[3] - l1[1])*(l2[3] - l2[1])

    # if abs(length1) < np.finfo(np.float32).eps and abs(length2) < np.finfo(np.float32).eps:
    if (abs(product / (length1 * length2)) < math.cos(np.pi/30)):
        return False

    mx1 = (l1[0] + l1[2]) * 0.5
    mx2 = (l2[0] + l2[2]) * 0.5

    my1 = (l1[1] + l1[3]) * 0.5
    my2 = (l2[1] + l2[3]) * 0.5
    dist = math.sqrt((mx1 - mx2)*(mx1 - mx2) + (my1 - my2)*(my1 - my2))

    if (dist > max(length1, length2) * 0.5):
        return False

    return True


def mean_distance_between(lines):
    distances = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            dist_1 = math.sqrt(
                (lines[i][0] - lines[j][0])**2 + (lines[i][1] - lines[j][1])**2)
            dist_2 = math.sqrt(
                (lines[i][2] - lines[j][2])**2 + (lines[i][3] - lines[j][3])**2)
            distances.append(max(dist_1, dist_2))
    return np.mean(distances)


def dist_siglesser_than_mean(l1, l2, mean):
    dist_1 = math.sqrt((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2)
    dist_2 = math.sqrt((l1[2] - l2[2])**2 + (l1[3] - l2[3])**2)
    dist = max(dist_1, dist_2)
    return dist < mean/10


def merge_equal_by_mean(lines, mean):
    if (len(lines) == 1):
        return lines
    flags = [0] * len(lines)
    merged_lines = []
    for i in range(len(lines)):
        if flags[i] != 0:
            continue
        for j in range(i+1, len(lines)):
            # if dist_siglesser_than_mean(lines[i], lines[j], mean):
            # if lines[i] == lines[i] and lines[i] == lines[i]:
            #     continue
            # if lines[j] == lines[j] and lines[j] == lines[j]:
            #     continue
            if(equal(lines[i], lines[j])):
                #  if(dist_siglesser_than_mean(lines[i], lines[j], mean) or equal(lines[i], lines[j])):
                flags[j] = '(' + str(i) + ',' + str(j) + ')'
                lines[i] = merge_equals(lines[i], lines[j])
        merged_lines.append(lines[i])
    return merged_lines


def LineIteratorByX(line):
    lineSet = []
    dx = line[2] - line[0]
    dy = line[3] - line[1]
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    D = 2*dy - dx
    y = line[1]

    for x in range(line[0], line[2] + 1):
        lineSet.append((x, y))
        if D > 0:
            y = y + yi
            D = D + 2*(dy - dx)
        else:
            D = D + 2*dy
    return np.array(lineSet)


def LineIteratorByY(line):
    lineSet = []
    dx = line[2] - line[0]
    dy = line[3] - line[1]
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    D = 2*dx - dy
    x = line[0]

    for y in range(line[1], line[3] + 1):
        lineSet.append((x, y))
        if D > 0:
            x = x + xi
            D = D + 2*(dx - dy)
        else:
            D = D + 2*dx
    return np.array(lineSet)


def LineIterator(line):
    if abs(line[3] - line[1]) < abs(line[2] - line[0]):
        if line[0] > line[2]:
            return LineIteratorByX([line[2], line[3], line[0], line[1]])
        else:
            return LineIteratorByX(line)
    else:
        if line[1] > line[3]:
            return LineIteratorByY([line[2], line[3], line[0], line[1]])
        else:
            return LineIteratorByY(line)


def isWhite(point, img):
    kernel = np.array([
        [[255, 255, 255],
         [255, 255, 255],
         [255, 255, 255]],
        [[255, 255, 255],
         [255, 255, 255],
         [255, 255, 255]],
        [[255, 255, 255],
         [255, 255, 255],
         [255, 255, 255]]
    ])
    kernel[1, 1] = img[point[1], point[0]]
    if point[0] > 0:
        kernel[0, 1] = img[point[1], point[0] - 1]
    if point[1] > 0:
        kernel[1, 0] = img[point[1] - 1, point[0]]
    if point[0] > 0 and point[1] > 0:
        kernel[0, 0] = img[point[1] - 1, point[0] - 1]
    if point[0] < img.shape[1] - 1:
        kernel[2, 1] = img[point[1], point[0] + 1]
    if point[1] < img.shape[0] - 1:
        kernel[1, 2] = img[point[1] + 1, point[0]]
    if point[0] < img.shape[1] - 1 and point[1] < img.shape[0] - 1:
        kernel[2, 2] = img[point[1] + 1, point[0] + 1]
    if point[0] > 0 and point[1] < img.shape[0] - 1:
        kernel[0, 2] = img[point[1] + 1, point[0] - 1]
    if point[0] < img.shape[1] - 1 and point[1] > 0:
        kernel[2, 0] = img[point[1] - 1, point[0] + 1]
    white_count = 0
    for i in range(0, 3):
        for j in range(0, 3):
            if kernel[i, j, 0] > 200 and kernel[i, j, 1] > 200 and kernel[i, j, 2] > 200:
                white_count += 1
    if white_count < 5:
        return False
    return True


def find_white_lines_by_points(lines, img):
    white_lines = []
    for line in lines:
        lineIterator = LineIterator(line)
        whiteness = 0
        for point in lineIterator:
            if isWhite(point, img):
                whiteness += 1
        # print("______________________________")
        # print(whiteness)
        # print(len(lineIterator))
        if whiteness/len(lineIterator) > 0.7:
            white_lines.append(line)
    return white_lines


def partition_Y_axis(img):
    Y_size = img.shape[0]
    seg_start = 0
    seg_end = 4
    partition = []
    while seg_end < Y_size:
        partition.append((seg_start, seg_end))
        seg_start = seg_end
        seg_end += 4
    partition.append((seg_start, seg_end))
    return partition


def find_white_lines_by_segments(lines, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    white_lines = []
    Y_partition = partition_Y_axis(img)
    for line in lines:
        line_iterator = LineIterator(line)
        whiteness = 0
        for seg in Y_partition:
            if seg[0] <= line[1] and seg[1] >= line[3] or seg[0] <= line[3] and seg[1] >= line[1]:
                is_white_left = True
                is_white_center = True
                is_white_right = True
                for point in line_iterator:
                    if seg[0] <= point[1] and seg[1] >= point[1]:
                        # if img[point[1], max(point[0] - 5, 0), 0] <= 200 or img[point[1], max(point[0] - 5, 0), 1] <= 200 or img[point[1], max(point[0] - 5, 0), 2] <= 200:
                        #     is_white_left = False
                        # if img[point[1], point[0], 0] <= 200 or img[point[1], point[0], 1] <= 200 or img[point[1], point[0], 2] <= 200:
                        #     is_white_center = False
                        # if img[point[1], min(point[0] + 5, img.shape[1] - 1), 0] <= 200 or img[point[1], min(point[0] + 5, img.shape[1] - 1), 1] <= 200 or img[point[1], min(point[0] + 5, img.shape[1] - 1), 2] <= 200:
                        #     is_white_right = False
                        if img[point[1], max(point[0] - 5, 0), 1] < 100:
                            is_white_left = False
                        if img[point[1], point[0], 1] < 100:
                            is_white_center = False
                        if img[point[1], min(point[0] + 5, img.shape[1] - 1), 1] < 100:
                            is_white_right = False
                if is_white_left == True or is_white_center == True or is_white_right == True:
                    whiteness += 1
        if whiteness > 9:
            # print(whiteness)
            white_lines.append(line)
    return white_lines


class LineNumException(Exception):
    def __init__(self, message, img):
        self.message = message
        self.img = img


def param_get(y, img):
    flag = 0
    white_count = 0
    start_shift = None
    left_white = None
    right_white = None
    lane_widths = []
    line_widths = []
    for x in range(img.shape[1]):
        if flag == 0:
            if not np.array_equal(img[y, x], [0, 0, 0]):
                flag = 1
                if left_white is not None:
                    lane_widths.append(x - left_white)
                left_white = x
                if start_shift is None:
                    start_shift = left_white
        if flag == 1:
            if np.array_equal(img[y, x], [0, 0, 0]):
                flag = 0
                right_white = x
                line_widths.append(right_white - left_white)

    if flag == 1:
        line_widths.append(img.shape[1] - left_white)

    return line_widths, lane_widths


def get_parameters(img):
    # thresh = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY)
    # connectivity = 4
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    #     thresh, connectivity, cv2.CV_32S)

    # max_area = np.max([stats[i, cv2.CC_STAT_HEIGHT]
    #                   for i in range(0, num_labels)])

    # lane_index = []

    # for i in range(0, num_labels):
    #     x = stats[i, cv2.CC_STAT_LEFT]
    #     y = stats[i, cv2.CC_STAT_TOP]
    #     w = stats[i, cv2.CC_STAT_WIDTH]
    #     h = stats[i, cv2.CC_STAT_HEIGHT]
    #     area = stats[i, cv2.CC_STAT_AREA]
    #     (c_X, c_Y) = centroids[i]

    #     if area > max_area / 10:
    #         lane_index.append(i)

    # mean_lane_width = 0.0
    # mean_line_width = 0.0

    # # if (len(lane_index) != 3):
    # #     print("number of detected lines: ", len(lane_index))
    # #     plt.imshow(thresh)
    # #     plt.show()

    # prev = -1
    # for index in lane_index:
    #     x = stats[index, cv2.CC_STAT_LEFT]
    #     y = stats[index, cv2.CC_STAT_TOP]
    #     w = stats[index, cv2.CC_STAT_WIDTH]
    #     h = stats[index, cv2.CC_STAT_HEIGHT]
    #     area = stats[index, cv2.CC_STAT_AREA]
    #     (c_X, c_Y) = centroids[index]

    #     mean_line_width += w/len(lane_index)
    #     if prev != -1:
    #         mean_lane_width += abs(x - prev)/len(lane_index)
    #     prev = x

    # return int(mean_line_width), int(mean_lane_width)

    # means_line_width = []
    # for line in lines:
    #     lineIterator = LineIterator(line)
    #     line_width_points = []
    #     for point in lineIterator:
    #         if np.array_equal(img[point[1], point[0]], [0, 0, 0]):
    #             is_black = True
    #         else:
    #             is_black = False
    #         if is_black:
    #             starting_x = point[0]
    #             for x in range(starting_x, 0, -1):
    #                 if not np.array_equal(img[point[1], x], [0, 0, 0]):
    #                     break
    #             white_width_right = 0
    #             for x in range(starting_x, 0, -1):
    #                 white_width_right = white_width_right + 1
    #                 if np.array_equal(img[point[1], x], [0, 0, 0]):
    #                     break
    #             for x in range(starting_x, img.shape[0]):
    #                 if not np.array_equal(img[point[1], x], [0, 0, 0]):
    #                     break
    #             white_width_left = 0
    #             for x in range(starting_x, img.shape[0]):
    #                 white_width_left = white_width_left + 1
    #                 if np.array_equal(img[point[1], x], [0, 0, 0]):
    #                     break
    #             white_width = max(white_width_left, white_width_right)
    #         else:
    #             starting_x = point[0]
    #             white_width_right = 0
    #             for x in range(starting_x, 0, -1):
    #                 white_width_right = white_width_right + 1
    #                 if not np.array_equal(img[point[1], x], [0, 0, 0]):
    #                     break
    #             white_width_left = 0
    #             for x in range(starting_x, img.shape[0]):
    #                 white_width_left = white_width_left + 1
    #                 if not np.array_equal(img[point[1], x], [0, 0, 0]):
    #                     break
    #             white_width = white_width_left + white_width_right
    #         line_width_points.append(white_width)
    #     means_line_width.append(np.mean(line_width_points))
    # mean_line_width = np.mean(means_line_width)

    # means_lane_width = []
    # for line in lines:
    #     lineIterator = LineIterator(line)
    #     line_lane_points = []
    #     for point in lineIterator:
    #         if np.array_equal(img[point[1], point[0]], [0, 0, 0]):
    #             is_black = True
    #         else:
    #             is_black = False
    #         if is_black:
    #             starting_x = point[0]
    #             black_left = 0
    #             for x in range(starting_x, 0, -1):
    #                 if not np.array_equal(img[point[1], x], [0, 0, 0]) and black_left > mean_line_width:
    #                     break
    #                 else:
    #                     black_left = black_left + 1
    #             black_right = 0
    #             for x in range(starting_x, img.shape[0]):
    #                 if not np.array_equal(img[point[1], x], [0, 0, 0]) and black_right > mean_line_width:
    #                     break
    #                 else:
    #                     black_left = black_left + 1
    #             black_width = max(black_right, black_right)
    #         else:
    #             starting_x = point[0]
    #             black_left = 0
    #             for x in range(starting_x, 0, -1):
    #                 if np.array_equal(img[point[1], x], [0, 0, 0]):
    #                     break
    #             for x in range(starting_x, 0, -1):
    #                 if not np.array_equal(img[point[1], x], [0, 0, 0]) and black_left > mean_line_width:
    #                     break
    #                 else:
    #                     black_left = black_left + 1
    #             black_right = 0
    #             for x in range(starting_x, img.shape[0]):
    #                 if np.array_equal(img[point[1], x], [0, 0, 0]):
    #                     break
    #             for x in range(starting_x, img.shape[0]):
    #                 if not np.array_equal(img[point[1], x], [0, 0, 0]) and black_right > mean_line_width:
    #                     break
    #                 else:
    #                     black_left = black_left + 1
    #             black_width = max(black_right, black_right)
    #         line_lane_points.append(black_width)
    #     means_lane_width.append(np.mean(line_lane_points))
    # mean_lane_width = np.mean(means_lane_width)

    y1 = int(0.8 * img.shape[0])
    y2 = int(0.2 * img.shape[0])

    line_width_1, lane_width_1 = param_get(y1, img)
    line_width_2, lane_width_2 = param_get(y2, img)
    # print(y)
    # print(start_shift)
    # print(line_widths)
    # print(lane_widths)
    if len(line_width_1) != 3 and len(line_width_2) != 3:
        raise LineNumException("Wrong number of lines", img)

    if len(line_width_1) == 3:
        line_widths = line_width_1
        lane_widths = lane_width_1
    else:
        line_widths = line_width_2
        lane_widths = lane_width_2

    return int(np.mean(line_widths)), int(np.mean(lane_widths))
    # return start_shift, int(np.mean(line_widths)), int(np.mean(lane_widths))
    # return int(mean_line_width), int(mean_lane_width)


def draw_lines(lines, img, color=(255, 0, 0)):
    for x1, y1, x2, y2 in lines:
        # print(x1,y1,x2,y2)
        cv2.line(img, (x1, y1), (x2, y2), color, 3)


def Canny(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (7, 7), 0)
    return cv2.Canny(img, 50, 100)


def find_lines(img, minLineLength=100, maxLineGap=100):
    img = Canny(img)
    lines = cv2.HoughLinesP(img, 1, np.pi/180, 75,
                            minLineLength=minLineLength, maxLineGap=maxLineGap)
    lines = np.squeeze(lines, axis=1)
    return lines


def warp_image(img):
    # pts1 = np.float32([[79, 220], [432, 24], [79, 307], [432, 972]])
    # pts2 = np.float32([[79, 24], [432, 24], [79, 972], [432, 972]])
    pts1 = np.float32([[220, 79], [24, 432], [307, 79], [972, 432]])
    height = math.sqrt((pts1[1, 0] - pts1[0, 0]) **
                       2 + (pts1[1, 1] - pts1[0, 1])**2)
    ratio = img.shape[0]/img.shape[1]
    width = ratio * height
    pts2 = np.float32([[24, 432 - height], [24, 432],
                      [24 + width, 432 - height], [24 + width, 432]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    # H = cv2.findHomography(pts1, pts2)
    # warped_orig = cv2.warpPerspective(img, H[0], (1000, 1000))
    warped_orig = cv2.warpPerspective(img, M, img.shape[:2])

    return warped_orig


def is_white(point, img):
    for i in range(0, 6):
        if img[point[1], max(point[0] - i, 0), 0] > 200 and img[point[1], max(point[0] - i, 0), 1] > 200 and img[point[1], max(point[0] - i, 0), 2] > 200:
            return True
    for i in range(0, 6):
        if img[point[1], min(point[0] + i, img.shape[1] - 1), 0] > 200 and img[point[1], min(point[0] + i, img.shape[1] - 1), 1] > 200 and img[point[1], min(point[0] + i, img.shape[1] - 1), 2] > 200:
            return True
    return False


def is_segmented(line, img, road_mask):
    Y_partition = partition_Y_axis(img)
    line_iterator = LineIterator(line)
    black_points = []
    white_length = 0
    prev_white_length = 0
    is_white_left = True
    is_white_center = True
    is_white_right = True
    good_segments_num = 0
    segments_num = 0
    for point in line_iterator:
        # is_white_left = True
        # is_white_center = True
        # is_white_right = True
        # if img[point[1], max(point[0] - 5, 0), 0] <= 200 or img[point[1], max(point[0] - 5, 0), 1] <= 200 or img[point[1], max(point[0] - 5, 0), 2] <= 200:
        #   is_white_left = False
        # if img[point[1], point[0], 0] <= 200 or img[point[1], point[0], 1] <= 200 or img[point[1], point[0], 2] <= 200:
        #   is_white_center = False
        # if img[point[1], min(point[0] + 5, img.shape[1] - 1), 0] <= 200 or img[point[1], min(point[0] + 5, img.shape[1] - 1), 1] <= 200 or img[point[1], min(point[0] + 5, img.shape[1] - 1), 2] <= 200:
        #   is_white_right = False
        # if is_white_left == True or is_white_center == True or is_white_right == True:
        #   white_length += 1
        if is_white(point, img):
            white_length += 1
        else:
            black_points.append(point)
            if white_length != 0:
                if prev_white_length == 0:
                    prev_white_length = white_length
                else:
                    # print(prev_white_length)
                    # print(white_length)
                    if (prev_white_length/white_length >= 0.8 and prev_white_length/white_length <= 1) or (prev_white_length/white_length <= 1.2 and prev_white_length/white_length >= 1):
                        good_segments_num += 1
                prev_white_length = white_length
                white_length = 0
                segments_num += 1
        #  very_strange_copy[point[1], point[0]] = [0, 0, 0]
    if white_length != 0:
        # print(prev_white_length)
        # print(white_length)
        if (prev_white_length/white_length >= 0.8 and prev_white_length/white_length <= 1) or (prev_white_length/white_length <= 1.2 and prev_white_length/white_length >= 1):
            good_segments_num += 1
        segments_num += 1
    # print("**********************************")
    # print(good_segments_num)
    # print(segments_num)
    # print("__________________________________________________")
    blackness = 0
    for point in black_points:
        if is_white(point, road_mask):
            blackness += 1
    if len(black_points) != 0 and segments_num > 2 and blackness/len(black_points) > 0.7:
        return True
    return False
    # if segments_num != 0 and good_segments_num/segments_num > 0.5:
    #  return True
    # return False

    # if white_length/len(line_iterator) < 0.4:
    #  return True
    # return False


def calc_transformation_matrix(lines, img):
    flag = 0
    pts1 = []
    debug_lines = []
    # for line in lines:
    #   x0 = line[0]
    #   y0 = line[1]
    #   x1 = line[2]
    #   y1 = line[3]
    #   if x1 - x0 != 0 and abs((y1-y0)/(x1-x0)) > 1:
    #     pts1.append([x0, y0])
    #     pts1.append([x1, y1])
    #     print(line)
    #     debug_lines.append(line)
    #     flag += 1
    #   if flag == 2:
    #     break
    for line in lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if x1 - x0 != 0 and (y1-y0)/(x1-x0) > 1:
            pts1.append([x0, y0])
            pts1.append([x1, y1])
            # print(line)
            debug_lines.append(line)
            break
    for line in lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if x1 - x0 != 0 and (y1-y0)/(x1-x0) < -1:
            pts1.append([x0, y0])
            pts1.append([x1, y1])
            # print(line)
            debug_lines.append(line)
            break
    pts1 = np.float32(pts1)
    height = math.sqrt((pts1[1, 0] - pts1[0, 0]) **
                       2 + (pts1[1, 1] - pts1[0, 1])**2)
    ratio = img.shape[0]/img.shape[1]
    width = ratio * height
    if pts1[0, 0] > pts1[1, 0] and pts1[0, 1] < pts1[1, 1]:
        pts2 = np.float32([[pts1[1, 0], pts1[1, 1] - height], [pts1[1, 0], pts1[1, 1]], [
                          pts1[1, 0] + width, pts1[1, 1] - height], [pts1[1, 0] + width, pts1[1, 1]]])
    if pts1[0, 0] <= pts1[1, 0] and pts1[0, 1] < pts1[1, 1]:
        pts2 = np.float32([[pts1[1, 0], pts1[1, 1] - height], [pts1[1, 0], pts1[1, 1]], [
                          pts1[1, 0] - width, pts1[1, 1]], [pts1[1, 0] - width, pts1[1, 1] - height]])
    if pts1[0, 0] > pts1[1, 0] and pts1[0, 1] >= pts1[1, 1]:
        pts2 = np.float32([[pts1[1, 0], pts1[1, 1] + height], [pts1[1, 0], pts1[1, 1]], [
                          pts1[1, 0] + width, pts1[1, 1] + height], [pts1[1, 0] + width, pts1[1, 1]]])
    if pts1[0, 0] <= pts1[1, 0] and pts1[0, 1] >= pts1[1, 1]:
        pts2 = np.float32([[pts1[1, 0], pts1[1, 1] + height], [pts1[1, 0], pts1[1, 1]], [
                          pts1[1, 0] - width, pts1[1, 1] + height], [pts1[1, 0] - width, pts1[1, 1]]])

    # debug_img = np.copy(img)
    # draw_lines(debug_lines, debug_img)
    # cv2_imshow(debug_img)

    return cv2.getPerspectiveTransform(pts1, pts2)


def apply_tranformation_to_line(line, M):
    # doesn't work or so I think
    line = np.array(
        [[line[0], line[1], 0], [line[2], line[3], 0]], dtype='float32')
    line = np.array([line])
    line = cv2.transform(line, M)
    line = (line[0]).astype(int)
    print(line)
    line = [line[0, 0], line[0, 1], line[1, 0], line[1, 1]]
    return np.squeeze(line)


def transform_lines(lines, M):
    trans = []
    for line in lines:
        trans.append(apply_tranformation_to_line(line, M))
    return trans


def calc_transformation_matrix_v2(lines, img):
    flag = 0
    left_lines = []
    right_lines = []
    pts1 = []
    debug_lines = []
    # for line in lines:
    #   x0 = line[0]
    #   y0 = line[1]
    #   x1 = line[2]
    #   y1 = line[3]
    #   if x1 - x0 != 0 and abs((y1-y0)/(x1-x0)) > 1:
    #     pts1.append([x0, y0])
    #     pts1.append([x1, y1])
    #     print(line)
    #     debug_lines.append(line)
    #     flag += 1
    #   if flag == 2:
    #     break
    for line in lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if x1 - x0 != 0 and y1 - y0 != 0 and (y1-y0)/(x1-x0) > 0.1:
            # pts1.append([x0, y0])
            # pts1.append([x1, y1])
            # #print(line)
            # debug_lines.append(line)
            # break
            left_lines.append(line)
    for line in lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if x1 - x0 != 0 and y1 - y0 != 0 and (y1-y0)/(x1-x0) < -0.1:
            # pts1.append([x0, y0])
            # pts1.append([x1, y1])
            # #print(line)
            # debug_lines.append(line)
            # break
            right_lines.append(line)
    x0 = left_lines[0][0]
    y0 = left_lines[0][1]
    x1 = left_lines[0][2]
    y1 = left_lines[0][3]
    min = (y1-y0)/(x1-x0)
    myX0 = x0
    myY0 = y0
    myX1 = x1
    myY1 = y1
    deb = left_lines[0]
    for line in left_lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if (y1-y0)/(x1-x0) < min:
            min = (y1-y0)/(x1-x0)
            myX0 = x0
            myY0 = y0
            myX1 = x1
            myY1 = y1
            deb = line
    debug_lines.append(deb)
    pts1.append([myX0, myY0])
    pts1.append([myX1, myY1])
    x0 = right_lines[0][0]
    y0 = right_lines[0][1]
    x1 = right_lines[0][2]
    y1 = right_lines[0][3]
    max = (y1-y0)/(x1-x0)
    myX0 = x0
    myY0 = y0
    myX1 = x1
    myY1 = y1
    deb = right_lines[0]
    for line in right_lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if (y1-y0)/(x1-x0) > max:
            max = (y1-y0)/(x1-x0)
            myX0 = x0
            myY0 = y0
            myX1 = x1
            myY1 = y1
            deb = line
    debug_lines.append(deb)
    pts1.append([myX0, myY0])
    pts1.append([myX1, myY1])

    pts1 = np.float32(pts1)
    height = math.sqrt((pts1[1, 0] - pts1[0, 0]) **
                       2 + (pts1[1, 1] - pts1[0, 1])**2)
    ratio = img.shape[0]/img.shape[1]
    width = ratio * height
    if pts1[0, 0] > pts1[1, 0] and pts1[0, 1] < pts1[1, 1]:
        pts2 = np.float32([[pts1[1, 0], pts1[1, 1] - height], [pts1[1, 0], pts1[1, 1]], [
                          pts1[1, 0] + width, pts1[1, 1] - height], [pts1[1, 0] + width, pts1[1, 1]]])
    if pts1[0, 0] <= pts1[1, 0] and pts1[0, 1] < pts1[1, 1]:
        pts2 = np.float32([[pts1[1, 0], pts1[1, 1] - height], [pts1[1, 0], pts1[1, 1]], [
                          pts1[1, 0] - width, pts1[1, 1]], [pts1[1, 0] - width, pts1[1, 1] - height]])
    if pts1[0, 0] > pts1[1, 0] and pts1[0, 1] >= pts1[1, 1]:
        pts2 = np.float32([[pts1[1, 0], pts1[1, 1] + height], [pts1[1, 0], pts1[1, 1]], [
                          pts1[1, 0] + width, pts1[1, 1] + height], [pts1[1, 0] + width, pts1[1, 1]]])
    if pts1[0, 0] <= pts1[1, 0] and pts1[0, 1] >= pts1[1, 1]:
        pts2 = np.float32([[pts1[1, 0], pts1[1, 1] + height], [pts1[1, 0], pts1[1, 1]], [
                          pts1[1, 0] - width, pts1[1, 1] + height], [pts1[1, 0] - width, pts1[1, 1]]])

    debug_img = np.copy(img)
    draw_lines(debug_lines, debug_img)
    # cv2_imshow(debug_img)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return (M, debug_img, (int(pts1[0][0] + width + 500), int(pts1[0][1] + height + 500)))


def make_road_scheme(lines, segmented_lines, img, warped_size):
    scheme = np.zeros_like(img)
    # M = calc_transformation_matrix(lines, orig)
    all_lines = list(lines)
    all_lines.extend(segmented_lines)
    M = calc_transformation_matrix_v2(all_lines, img)

    print(len(lines))
    print(len(segmented_lines))

    mean = mean_distance_between(all_lines)

    lines = merge_equal_by_mean(lines, mean)
    segmented_lines = merge_equal_by_mean(segmented_lines, mean)

    draw_lines(lines, scheme)
    draw_lines(segmented_lines, scheme, color=(0, 0, 255))
    trans_scheme = cv2.warpPerspective(scheme, M, warped_size)

    print(len(lines))
    print(len(segmented_lines))

    # trans_lines = []
    # for line in lines:
    #   trans_line = apply_tranformation_to_line(line, M)
    #   trans_lines.append(trans_line)
    # draw_lines(trans_lines, scheme)
    # trans_seg_lines = []
    # for line in segmented_lines:
    #   trans_seg_line = apply_tranformation_to_line(line, M)
    #   trans_seg_lines.append(trans_seg_line)
    # draw_lines(trans_seg_lines, scheme, color = (0, 0, 255))

    return (scheme, trans_scheme)


def find_segmented_lines(lines, img, road_mask):
    seg_lines = []
    for line in lines:
        if is_segmented(line, img, road_mask):
            seg_lines.append(line)
    return seg_lines


def calc_transformation_matrix_v3(lines, img):
    flag = 0
    left_lines = []
    right_lines = []
    pts1 = []
    debug_lines = []
    # for line in lines:
    #   x0 = line[0]
    #   y0 = line[1]
    #   x1 = line[2]
    #   y1 = line[3]
    #   if x1 - x0 != 0 and abs((y1-y0)/(x1-x0)) > 1:
    #     pts1.append([x0, y0])
    #     pts1.append([x1, y1])
    #     print(line)
    #     debug_lines.append(line)
    #     flag += 1
    #   if flag == 2:
    #     break
    for line in lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if x1 - x0 != 0 and y1 - y0 != 0 and (y1-y0)/(x1-x0) > 0.1:
            # pts1.append([x0, y0])
            # pts1.append([x1, y1])
            # #print(line)
            # debug_lines.append(line)
            # break
            left_lines.append(line)
    for line in lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if x1 - x0 != 0 and y1 - y0 != 0 and (y1-y0)/(x1-x0) < -0.1:
            # pts1.append([x0, y0])
            # pts1.append([x1, y1])
            # #print(line)
            # debug_lines.append(line)
            # break
            right_lines.append(line)
    x0 = left_lines[0][0]
    y0 = left_lines[0][1]
    x1 = left_lines[0][2]
    y1 = left_lines[0][3]
    min = (y1-y0)/(x1-x0)
    myX0 = x0
    myY0 = y0
    myX1 = x1
    myY1 = y1
    deb = left_lines[0]
    for line in left_lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if (y1-y0)/(x1-x0) < min:
            min = (y1-y0)/(x1-x0)
            myX0 = x0
            myY0 = y0
            myX1 = x1
            myY1 = y1
            deb = line
    debug_lines.append(deb)
    pts1.append([myX0, myY0])
    pts1.append([myX1, myY1])
    x0 = right_lines[0][0]
    y0 = right_lines[0][1]
    x1 = right_lines[0][2]
    y1 = right_lines[0][3]
    max = (y1-y0)/(x1-x0)
    myX0 = x0
    myY0 = y0
    myX1 = x1
    myY1 = y1
    deb = right_lines[0]
    for line in right_lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if (y1-y0)/(x1-x0) > max:
            max = (y1-y0)/(x1-x0)
            myX0 = x0
            myY0 = y0
            myX1 = x1
            myY1 = y1
            deb = line
    debug_lines.append(deb)
    pts1.append([myX0, myY0])
    pts1.append([myX1, myY1])

    pts1 = np.float32(pts1)
    height = math.sqrt((pts1[1, 0] - pts1[0, 0]) ** 2
                       + (pts1[1, 1] - pts1[0, 1]) ** 2)
    ratio = img.shape[0]/img.shape[1]
    width = ratio * height
    print(pts1.astype(np.int64))
    pts1 = get_correct_config(pts1)

    # pts2 = np.float32([(100, 100), (100, 100 + height),
    #                   (100 + width, 100), (100 + width, 100 + height)])

    # pts2 = np.float32([(100 + width, 100), (100 + width, 100 + height),
    #                    (100, 100), (100, 100 + height)])

    pts2 = get_corresponding_pts(pts1, width, height)

    debug_img = np.copy(img)
    draw_lines(debug_lines, debug_img)
    print(width)
    print(height)
    print(pts1.astype(np.int64))
    print(pts2.astype(np.int64))
    # cv2_imshow(debug_img)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return (M, debug_img, (int(100 + width + 100), int(100 + height + 100)))


def calc_transformation_matrix_v4(lines, img):
    flag = 0
    left_lines = []
    right_lines = []
    pts1 = []
    debug_lines = []
    # for line in lines:
    #   x0 = line[0]
    #   y0 = line[1]
    #   x1 = line[2]
    #   y1 = line[3]
    #   if x1 - x0 != 0 and abs((y1-y0)/(x1-x0)) > 1:
    #     pts1.append([x0, y0])
    #     pts1.append([x1, y1])
    #     print(line)
    #     debug_lines.append(line)
    #     flag += 1
    #   if flag == 2:
    #     break
    for line in lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if x1 - x0 != 0 and y1 - y0 != 0 and (y1-y0)/(x1-x0) > 0.1:
            # pts1.append([x0, y0])
            # pts1.append([x1, y1])
            # #print(line)
            # debug_lines.append(line)
            # break
            left_lines.append(line)
    for line in lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if x1 - x0 != 0 and y1 - y0 != 0 and (y1-y0)/(x1-x0) < -0.1:
            # pts1.append([x0, y0])
            # pts1.append([x1, y1])
            # #print(line)
            # debug_lines.append(line)
            # break
            right_lines.append(line)
    x0 = left_lines[0][0]
    y0 = left_lines[0][1]
    x1 = left_lines[0][2]
    y1 = left_lines[0][3]
    min = (y1-y0)/(x1-x0)
    myX0 = x0
    myY0 = y0
    myX1 = x1
    myY1 = y1
    deb = left_lines[0]
    for line in left_lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if (y1-y0)/(x1-x0) < min:
            min = (y1-y0)/(x1-x0)
            myX0 = x0
            myY0 = y0
            myX1 = x1
            myY1 = y1
            deb = line
    debug_lines.append(deb)
    pts1.append([myX0, myY0])
    pts1.append([myX1, myY1])
    x0 = right_lines[0][0]
    y0 = right_lines[0][1]
    x1 = right_lines[0][2]
    y1 = right_lines[0][3]
    max = (y1-y0)/(x1-x0)
    myX0 = x0
    myY0 = y0
    myX1 = x1
    myY1 = y1
    deb = right_lines[0]
    for line in right_lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if (y1-y0)/(x1-x0) > max:
            max = (y1-y0)/(x1-x0)
            myX0 = x0
            myY0 = y0
            myX1 = x1
            myY1 = y1
            deb = line
    debug_lines.append(deb)
    pts1.append([myX0, myY0])
    pts1.append([myX1, myY1])

    pts1 = np.float32(pts1)
    height = math.sqrt((pts1[1, 0] - pts1[0, 0]) ** 2
                       + (pts1[1, 1] - pts1[0, 1]) ** 2)
    ratio = img.shape[0]/img.shape[1]
    width = ratio * height
    print(pts1.astype(np.int64))
    # pts1 = np.float32([(590, 800), (625, 600), (1710, 800), (1470, 600)])
    pts1 = np.float32([(370, 200), (70, 400), (990, 200), (1225, 400)])
    pts1 = get_correct_config(pts1)

    # pts2 = np.float32([(100, 100), (100, 100 + height),
    #                   (100 + width, 100), (100 + width, 100 + height)])

    # pts2 = np.float32([(100 + width, 100), (100 + width, 100 + height),
    #                    (100, 100), (100, 100 + height)])

    pts2 = get_corresponding_pts(pts1, width, height)

    debug_img = np.copy(img)
    draw_lines(debug_lines, debug_img)
    print(width)
    print(height)
    print(pts1.astype(np.int64))
    print(pts2.astype(np.int64))
    # cv2_imshow(debug_img)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return (M, debug_img, (int(100 + width + 100), int(100 + height + 100)))


def get_corresponding_pts(pts, width, height):
    if pts[0, 1] < pts[1, 1]:
        pts2 = np.float32([(100, 100), (100, 100 + height),
                           (100 + width, 100), (100 + width, 100 + height)])
    else:
        pts2 = np.float32([(100, 100 + height), (100, 100),
                           (100 + width, 100 + height), (100 + width, 100)])
    return pts2


def get_correct_config(pts):
    # ind = np.sort(pts[:, 0])
    # return np.take(pts, ind)

    if pts[0, 0] < pts[2, 0] and pts[0, 0] < pts[3, 0] and pts[1, 0] < pts[2, 0] and pts[1, 0] < pts[3, 0]:
        if pts[0, 0] > pts[1, 0] and pts[0, 1] < pts[1, 1]:
            if pts[2, 0] < pts[3, 0] and pts[2, 1] < pts[3, 1]:
                pts = pts
            elif pts[2, 0] >= pts[3, 0] and pts[2, 1] >= pts[3, 1]:
                pts = [pts[0], pts[1], pts[3], pts[2]]
            # else:
            #     print(np.uint(pts))
            #     raise Exception("Image lines cant have such config: left line")
        elif pts[0, 0] <= pts[1, 0] and pts[0, 1] >= pts[1, 1]:
            if pts[2, 0] < pts[3, 0] and pts[2, 1] < pts[3, 1]:
                pts = [pts[1], pts[0], pts[2], pts[3]]
            elif pts[2, 0] >= pts[3, 0] and pts[2, 1] >= pts[3, 1]:
                pts = [pts[1], pts[0], pts[3], pts[2]]
        #     else:
        #         print(np.uint(pts))
        #         raise Exception(
        #             "Image lines cant have such config: right line")
        # else:
        #     print(np.uint(pts))
        #     raise Exception("Image lines cant have such config: upper")
    elif pts[0, 0] > pts[2, 0] and pts[0, 0] > pts[3, 0] and pts[1, 0] > pts[2, 0] and pts[1, 0] > pts[3, 0]:
        if pts[2, 0] > pts[3, 0] and pts[2, 1] < pts[3, 1]:
            if pts[0, 0] < pts[1, 0] and pts[0, 1] < pts[1, 1]:
                pts = [pts[2], pts[3], pts[0], pts[1]]
            elif pts[0, 0] >= pts[1, 0] and pts[0, 1] >= pts[1, 1]:
                pts = [pts[3], pts[2], pts[0], pts[1]]
            # else:
            #     print(np.uint(pts))
            #     raise Exception(
            #         "Image lines cant have such config:shift left line")
        elif pts[2, 0] <= pts[3, 0] and pts[2, 1] >= pts[3, 1]:
            if pts[0, 0] < pts[1, 0] and pts[0, 1] < pts[1, 1]:
                pts = [pts[2], pts[3], pts[1], pts[0]]
            elif pts[0, 0] >= pts[1, 0] and pts[0, 1] >= pts[1, 1]:
                pts = [pts[3], pts[2], pts[1], pts[0]]
    #         else:
    #             print(np.uint(pts))
    #             raise Exception(
    #                 "Image lines cant have such config:shift right line")
    #     else:
    #         print(np.uint(pts))
    #         raise Exception("Image lines cant have such config:shift lower")
    # else:
    #     print(np.uint(pts))
    #     raise Exception("Image lines cant have such config: shift")
    return np.array(pts)


def dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def merge_equals(l1, l2):
    # slope1, shift1 = get_line_slope_shift(l1)
    # slope2, shift2 = get_line_slope_shift(l2)

    # slope = slope1/2 + slope2/2
    # shift = shift1/2 + shift2/2

    # return convert_slope_shift_to_line_by_x(min(l1[0], l1[2], l2[0], l2[2]), max(l1[0], l1[2], l2[0], l2[2]), slope, shift)

    # x1, y1, x2, y2 = l1
    # x3, y3, x4, y4 = l2
    # len1 = dist(x1, y1, x3, y3)
    # len2 = dist(x1, y1, x4, y4)
    # len3 = dist(x2, y2, x3, y3)
    # len4 = dist(x2, y2, x4, y4)

    # if len1 == max(len1, len2, len3, len4):
    #     return [x1, y1, x3, y3]
    # if len2 == max(len1, len2, len3, len4):
    #     return [x1, y1, x4, y4]
    # if len3 == max(len1, len2, len3, len4):
    #     return [x2, y2, x3, y3]
    # if len4 == max(len1, len2, len3, len4):
    #     return [x2, y2, x4, y4]

    if get_line_length(l1) < get_line_length(l2):
        return l2
    return l1


def get_line_length(line):
    x1, y1, x2, y2 = line
    return dist(x1, y1, x2, y2)


def get_dot_product(l1, l2):
    return (l1[2] - l1[0])*(l2[2] - l2[0]) + (l1[3] - l1[1])*(l2[3] - l2[1])


def get_line_slope_shift(line):
    x1, y1, x2, y2 = line
    if x2-x1 == 0:
        return math.inf, 0
    slope = (y2-y1)/(x2-x1)
    shift = y1 - slope * x1
    return slope, shift


def convert_slope_shift_to_line_by_y(y1, y2, slope, shift):
    if math.isinf(slope) and slope > 0:
        x1 = 0
        x2 = 0
    else:
        x1 = int((y1 - shift)/slope)
        x2 = int((y2 - shift)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return [x1, y1, x2, y2]


def convert_slope_shift_to_line_by_x(x1, x2, slope, shift):
    if math.isinf(slope) and slope > 0:
        y1 = 0
        y2 = 0
    else:
        y1 = int(slope*x1 + shift)
        y2 = int(slope*x2 + shift)
    x1 = int(x1)
    x2 = int(x2)
    return [x1, y1, x2, y2]


def coordinate_is_in_image(coor, img_size):
    return coor[0] >= 0 and coor[0] < img_size[0] and coor[1] >= 0 and coor[1] < img_size[1]


def line_is_in_image(line, img_size):
    return coordinate_is_in_image((line[0], line[1]), img_size) and coordinate_is_in_image((line[2], line[3]), img_size)


def crop_line_in_image(line, img_size):
    slope, shift = get_line_slope_shift(line)
    x1, y1, x2, y2 = 0, 0, 0, 0
    # if line[0] <= 0:
    #     if line[2] >= img_size[0]:
    #         line = convert_slope_shift_to_line_by_x(
    #             0, img_size[0] - 1, slope, shift)
    # if line[0] >= img_size[0]:
    #     if line[2] <= 0:
    #         line = convert_slope_shift_to_line_by_x(
    #             0, img_size[0] - 1, slope, shift)
    # if line[1] <= 0:
    #     if line[3] >= img_size[1]:
    #         line = convert_slope_shift_to_line_by_y(
    #             0, img_size[1] - 1, slope, shift)
    # if line[1] >= img_size[1]:
    #     if line[3] <= 0:
    #         line = convert_slope_shift_to_line_by_y(
    #             0, img_size[1] - 1, slope, shift)
    # if line[2] <= 0:
    #     if line[0] >= img_size[0]:
    #         line = convert_slope_shift_to_line_by_x(
    #             0, img_size[0] - 1, slope, shift)
    # if line[2] >= img_size[0]:
    #     if line[0] <= 0:
    #         line = convert_slope_shift_to_line_by_x(
    #             0, img_size[0] - 1, slope, shift)
    # if line[3] <= 0:
    #     if line[1] >= img_size[1]:
    #         line = convert_slope_shift_to_line_by_y(
    #             0, img_size[1] - 1, slope, shift)
    # if line[3] >= img_size[1]:
    #     if line[1] <= 0:
    #         line = convert_slope_shift_to_line_by_y(
    #             0, img_size[1] - 1, slope, shift)

    # if slope < 0:
    #     if line[0] < line[2]:
    #         x1 = line[0]
    #         y1 = line[1]
    #         x2 = line[2]
    #         y2 = line[3]
    #     else:
    #         x2 = line[0]
    #         y2 = line[1]
    #         x1 = line[2]
    #         y1 = line[3]
    #     # line = [max(0, x1), min(img_size[1] - 1, y1),
    #     #         min(img_size[0] - 1, x2), max(0, y2)]
    #     line = convert_slope_shift_to_line_by_x(
    #         max(0, x1), min(img_size[0] - 1, x2), slope, shift)
    # if slope > 0:
    #     if line[2] < line[0]:
    #         x1 = line[0]
    #         y1 = line[1]
    #         x2 = line[2]
    #         y2 = line[3]
    #     else:
    #         x2 = line[0]
    #         y2 = line[1]
    #         x1 = line[2]
    #         y1 = line[3]
    #     # line = [min(img_size[0] - 1, x1), min(img_size[1] - 1, y1),
    #     #         max(0, x2), max(0, y2)]
    #     line = convert_slope_shift_to_line_by_x(
    #         min(img_size[0] - 1, x1), max(0, x2), slope, shift)

    if line[0] <= 0:
        return convert_slope_shift_to_line_by_x(0, max(0, line[2] - 1), slope, shift)
    if line[0] >= img_size[0]:
        return convert_slope_shift_to_line_by_x(img_size[0] - 1, max(0, line[2] - 1), slope, shift)
    if line[2] <= 0:
        return convert_slope_shift_to_line_by_x(max(line[0] - 1, 0), 0, slope, shift)
    if line[2] >= img_size[0]:
        return convert_slope_shift_to_line_by_x(max(0, line[0] - 1), img_size[0] - 1, slope, shift)
    if line[1] <= 0:
        return convert_slope_shift_to_line_by_y(0, max(0, line[3] - 1), slope, shift)
    if line[1] >= img_size[1]:
        return convert_slope_shift_to_line_by_y(img_size[1] - 1, max(line[3] - 1, 0), slope, shift)
    if line[3] <= 0:
        return convert_slope_shift_to_line_by_y(max(0, line[1] - 1), 0, slope, shift)
    if line[3] >= img_size[1]:
        return convert_slope_shift_to_line_by_y(max(line[1] - 1, 0), img_size[1] - 1, slope, shift)
    # print(line)
    return line


def stretch_line(line, img_size):
    slope, shift = get_line_slope_shift(line)
    return convert_slope_shift_to_line_by_y(0, img_size[1], slope, shift)
    # length = get_line_length(line)
    # if length < img_size[0] * 0.5:
    #     slope, shift = get_line_slope_shift(line)
    #     line = convert_slope_shift_to_line_by_y(
    #         min(line[1], line[3]), img_size[0] * 0.5, slope, shift)
    #     if line_is_in_image(line, img_size) == False:
    #         line = crop_line_in_image(line, img_size)
    # return line


def merge_lines(lines, img_size):
    # lines = merge_equal_by_mean(lines, mean_distance_between(lines))
    # print(lines)
    # for i in range(len(lines)):
    #     lines[i] = stretch_line(lines[i], img_size)
    # print(lines)
    # return merge_equal_by_mean(lines, mean_distance_between(lines))
    for i in range(len(lines)):
        lines[i] = stretch_line(lines[i], img_size)
    lines = merge_equal_by_mean(lines, mean_distance_between(lines))
    for i in range(len(lines)):
        # print(lines[i])
        lines[i] = crop_line_in_image(lines[i], img_size)
        # print(lines[i])
        # print(i)
    real_lines = []
    for line in lines:
        if line[0] != line[2] and line[1] != line[3]:
            real_lines.append(line)
    lines = real_lines
    return lines


def get_shorter_segment(l1, l2):
    len1 = get_line_length(l1)
    len2 = get_line_length(l2)
    return l1 if len1 < len2 else l2


def calc_transformation_matrix_v5(lines, img):
    left_lines = []
    right_lines = []
    pts1 = []
    debug_lines = []
    for line in lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if x1 - x0 != 0 and y1 - y0 != 0 and (y1-y0)/(x1-x0) > 0.5:
            left_lines.append(line)
    for line in lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if x1 - x0 != 0 and y1 - y0 != 0 and (y1-y0)/(x1-x0) < -0.5:
            right_lines.append(line)
    x0 = left_lines[0][0]
    y0 = left_lines[0][1]
    x1 = left_lines[0][2]
    y1 = left_lines[0][3]
    min = (y1-y0)/(x1-x0)
    myX0 = x0
    myY0 = y0
    myX1 = x1
    myY1 = y1
    deb = left_lines[0]
    for line in left_lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if (y1-y0)/(x1-x0) < min:
            min = (y1-y0)/(x1-x0)
            myX0 = x0
            myY0 = y0
            myX1 = x1
            myY1 = y1
            deb = line
    debug_lines.append(deb)
    pts1.append([myX0, myY0])
    pts1.append([myX1, myY1])
    x0 = right_lines[0][0]
    y0 = right_lines[0][1]
    x1 = right_lines[0][2]
    y1 = right_lines[0][3]
    max = (y1-y0)/(x1-x0)
    myX0 = x0
    myY0 = y0
    myX1 = x1
    myY1 = y1
    deb = right_lines[0]
    for line in right_lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if (y1-y0)/(x1-x0) > max:
            max = (y1-y0)/(x1-x0)
            myX0 = x0
            myY0 = y0
            myX1 = x1
            myY1 = y1
            deb = line
    debug_lines.append(deb)
    pts1.append([myX0, myY0])
    pts1.append([myX1, myY1])

    pts1 = np.float32(pts1)
    height = math.sqrt((pts1[1, 0] - pts1[0, 0]) ** 2
                       + (pts1[1, 1] - pts1[0, 1]) ** 2)
    ratio = img.shape[0]/img.shape[1]
    width = ratio * height
    # print(pts1.astype(np.int64))
    # pts1 = np.float32([(590, 800), (625, 600), (1710, 800), (1470, 600)])

    l1 = [pts1[0, 0], pts1[0, 1], pts1[1, 0], pts1[1, 1]]
    l2 = [pts1[2, 0], pts1[2, 1], pts1[3, 0], pts1[3, 1]]

    slope1, shift1 = get_line_slope_shift(l1)
    slope2, shift2 = get_line_slope_shift(l2)

    shorter_seg = get_shorter_segment(l1, l2)
    lower_point = get_lower_point(shorter_seg)

    # y1 = 0.8 * lower_point[1]
    # y2 = 0.7 * lower_point[1]
    y1 = lower_point[1]
    y2 = 0
    # print(y1, y2)
    l1 = convert_slope_shift_to_line_by_y(y1, y2, slope1, shift1)
    l2 = convert_slope_shift_to_line_by_y(y1, y2, slope2, shift2)

    l1 = crop_line_in_image(l1, (img.shape[1], img.shape[0]))
    l2 = crop_line_in_image(l2, (img.shape[1], img.shape[0]))

    pts1 = np.float32([(l1[0], l1[1]), (l1[2], l1[3]),
                      (l2[0], l2[1]), (l2[2], l2[3])])
    pts1 = get_correct_config(pts1)

    # pts2 = np.float32([(100, 100), (100, 100 + height),
    #                   (100 + width, 100), (100 + width, 100 + height)])

    # pts2 = np.float32([(100 + width, 100), (100 + width, 100 + height),
    #                    (100, 100), (100, 100 + height)])

    pts2 = get_corresponding_pts(pts1, width, height)

    debug_img = np.copy(img)
    draw_lines(debug_lines, debug_img)
    # print(img.shape)
    # print(width)
    # print(height)
    # print(pts1.astype(np.int64))
    # print(pts2.astype(np.int64))
    # cv2_imshow(debug_img)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return (M, debug_img, (int(100 + width + 100), int(100 + height)))


def get_lower_point(line):
    return (line[0], line[1]) if line[1] > line[3] else (line[2], line[3])


def get_upper_point(line):
    return (line[2], line[3]) if line[3] < line[1] else (line[0], line[1])


def n_tests(n):
    orig_means = []
    img_size = (1000, 1000)
    rng = np.random.default_rng(4)
    error = 0
    orig_score = 0.0
    mean_score = 0.0
    mean_score_circle_noise = 0.0
    for iter_num in range(n):
        try:
            lane_width = rng.integers(low=350, high=401)
            line_width = rng.integers(low=10, high=20)
            start_x = x = rng.integers(low=0, high=200)

            img = np.zeros((img_size[0], img_size[1], 3), np.uint8)

            lines = []
            for i in range(3):
                lines.append(Line(Point(x, 0), Point(x, img_size[1])))
                x += lane_width

            for line in lines:
                cv2.line(img, line.start(), line.end(), color=(
                    255, 255, 255), thickness=line_width)

            src = np.float32([lines[0].start(),
                              lines[2].start(),
                              lines[0].end(),
                              lines[2].end()])

            # dst = np.float32([(lines[0].start()[0] + 500, lines[0].start()[1]),
            #                   (lines[2].start()[0] - 200, lines[2].start()[1]),
            #                   (lines[0].end()[0] - 1000, lines[0].end()[1]),
            #                   (lines[2].end()[0] + 1000, lines[2].end()[1])])

            # M = cv2.getPerspectiveTransform(src, dst)

            M, dst = gen_Matrice(src, rng)

            warped_img = cv2.warpPerspective(
                img, M, (int(dst[3][0]), int(dst[3][1])))

            circle_noised = np.copy(warped_img)

            M_inv = np.linalg.inv(M)
            restored_img = cv2.warpPerspective(warped_img, M_inv, img_size)

            lin, lan = get_parameters(restored_img)

            orig_score += metric_res(line_width, lin, lan, lane_width)/n

            mean_score = run_algo(warped_img, mean_score,
                                  start_x, line_width, lane_width, n)

            # mean_score_circle_noise = run_circle_noise(
            #     circle_noised, mean_score_circle_noise, start_x, line_width, lane_width, n, rng)
            # print(mean_score_circle_noise)
            # plt.imshow(custom_restored_img)
            # plt.show()
            # white_lines = find_white_lines_by_segments(
            #     white_lines, lines_mask_copy)

            # if len(white_lines) != 3:
            #     test = np.copy(warped_img)

            #     draw_lines(white_lines, test)

            #     plt.imshow(test)
            #     plt.show()

            #     error = error + 1

            #     print("Number of white lines:", len(
            #         white_lines), "on iteration:", iter_num, ", error rate: ", error/n)
        except LineNumException as e:
            plt.imshow(e.img)
            plt.show()
            print(e.message)
            print(iter_num, start_x, line_width, lane_width)
        except:
            print(iter_num, start_x, line_width, lane_width)
        if iter_num % 10 and iter_num > 50:
            orig_means.append(
                ((orig_score*n/iter_num), (mean_score*n/iter_num)))
    print("orig: ", orig_score, " mean: ", mean_score)
    orig_means.append([orig_score, mean_score])
    return orig_means


def gen_dst_pts(src, rng):
    upper_shift_max = int((src[1][0] - src[0][0])/2 - 10)

    shift_a = rng.integers(low=200, high=upper_shift_max)
    shift_b = rng.integers(low=200, high=upper_shift_max)
    shift_c = rng.integers(low=800, high=1200)
    shift_d = rng.integers(low=800, high=1200)

    return np.float32([(src[0][0] + shift_a, src[0][1]),
                       (src[1][0] - shift_b, src[1][1]),
                       (src[2][0] - shift_c, src[2][1]),
                       (src[3][0] + shift_d, src[3][1])])


def gen_Matrice(src, rng):
    dst = gen_dst_pts(src, rng)

    matrice = cv2.getPerspectiveTransform(src, dst)

    while math.isclose(np.linalg.det(matrice), 0.0, abs_tol=1e-20):
        dst = gen_dst_pts(src, rng)
        matrice = cv2.getPerspectiveTransform(src, dst)
    return matrice, dst


def metric_res(delta, line_width, lane_width, width):
    # return abs(start - start_x) + abs(delta - line_width) + abs(width - lane_width)
    return abs(delta - line_width) + abs(width - lane_width)


def run_algo(warped_img, mean_score, start_x, line_width, lane_width, n):
    pos_lines = find_lines(warped_img)

    hsl = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HLS)
    lower = np.array([200, 200, 200], dtype="uint8")
    upper = np.array([255, 255, 255], dtype="uint8")
    lines_mask = cv2.inRange(warped_img, lower, upper)
    lines_mask_copy = cv2.bitwise_and(
        warped_img, warped_img, mask=lines_mask)
    white_lines = find_white_lines_by_segments(
        pos_lines, lines_mask_copy)

    white_lines = merge_lines(
        white_lines, (lines_mask_copy.shape[1], lines_mask_copy.shape[0]))

    test = np.copy(warped_img)

    custom_M_inv, debug_img, restored_img_size = calc_transformation_matrix_v5(
        white_lines, test)

    custom_restored_img = cv2.warpPerspective(
        warped_img, custom_M_inv, restored_img_size)

    # trans = transform_lines(white_lines, custom_M_inv)

    # trans_img = np.copy(custom_restored_img)

    # draw_lines(trans, trans_img, (255, 255, 0))

    # plt.imshow(trans_img)
    # plt.show()

    # start, delta, width = get_parameters(custom_restored_img)
    delta, width = get_parameters(custom_restored_img)

    # mean_score = mean_score + \
    #     metric_res(start, start_x, delta,
    #                line_width, width, lane_width)/n
    # print(mean_score, delta, line_width, width, lane_width)
    mean_score = mean_score + \
        metric_res(delta, line_width, width, lane_width)/n

    return mean_score


def run_circle_noise(warped_img, mean_score, start_x, line_width, lane_width, n, rng):
    circle_noise(warped_img, rng)

    # plt.imshow(warped_img)
    # plt.show()
    # print(warped_img.shape)

    return run_algo(warped_img, mean_score, start_x, line_width, lane_width, n)


def circle_noise(img, rng):
    n = 1000
    for iter_num in range(n):
        radius = rng.integers(low=1, high=3)
        center_x = rng.integers(low=10, high=img.shape[1] - 11)
        center_y = rng.integers(low=10, high=img.shape[0] - 11)
        cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), -1)


def calc_transformation_matrix_v6(lines, img):
    left_lines = []
    right_lines = []
    pts1 = []
    debug_lines = []
    for line in lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if x1 - x0 != 0 and y1 - y0 != 0 and (y1-y0)/(x1-x0) > 0.1:
            left_lines.append(line)
    for line in lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if x1 - x0 != 0 and y1 - y0 != 0 and (y1-y0)/(x1-x0) < -0.1:
            right_lines.append(line)
    x0 = left_lines[0][0]
    y0 = left_lines[0][1]
    x1 = left_lines[0][2]
    y1 = left_lines[0][3]
    min = (y1-y0)/(x1-x0)
    myX0 = x0
    myY0 = y0
    myX1 = x1
    myY1 = y1
    deb = left_lines[0]
    for line in left_lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if (y1-y0)/(x1-x0) < min:
            min = (y1-y0)/(x1-x0)
            myX0 = x0
            myY0 = y0
            myX1 = x1
            myY1 = y1
            deb = line
    debug_lines.append(deb)
    pts1.append([myX0, myY0])
    pts1.append([myX1, myY1])
    x0 = right_lines[0][0]
    y0 = right_lines[0][1]
    x1 = right_lines[0][2]
    y1 = right_lines[0][3]
    max = (y1-y0)/(x1-x0)
    myX0 = x0
    myY0 = y0
    myX1 = x1
    myY1 = y1
    deb = right_lines[0]
    for line in right_lines:
        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        if (y1-y0)/(x1-x0) > max:
            max = (y1-y0)/(x1-x0)
            myX0 = x0
            myY0 = y0
            myX1 = x1
            myY1 = y1
            deb = line
    debug_lines.append(deb)
    pts1.append([myX0, myY0])
    pts1.append([myX1, myY1])

    pts1 = np.float32(pts1)
    height = math.sqrt((pts1[1, 0] - pts1[0, 0]) ** 2
                       + (pts1[1, 1] - pts1[0, 1]) ** 2)
    ratio = img.shape[0]/img.shape[1]
    width = ratio * height
    # print(pts1.astype(np.int64))
    # pts1 = np.float32([(590, 800), (625, 600), (1710, 800), (1470, 600)])

    l1 = [pts1[0, 0], pts1[0, 1], pts1[1, 0], pts1[1, 1]]
    l2 = [pts1[2, 0], pts1[2, 1], pts1[3, 0], pts1[3, 1]]

    slope1, shift1 = get_line_slope_shift(l1)
    slope2, shift2 = get_line_slope_shift(l2)

    shorter_seg = get_shorter_segment(l1, l2)
    lower_point = get_lower_point(shorter_seg)

    # y1 = 0.8 * lower_point[1]
    # y2 = 0.7 * lower_point[1]
    y1 = lower_point[1]
    y2 = 0
    # print(y1, y2)
    l1 = convert_slope_shift_to_line_by_y(y1, y2, slope1, shift1)
    l2 = convert_slope_shift_to_line_by_y(y1, y2, slope2, shift2)

    l1 = crop_line_in_image(l1, (img.shape[1], img.shape[0]))
    l2 = crop_line_in_image(l2, (img.shape[1], img.shape[0]))

    pts1 = np.float32([(l1[0], l1[1]), (l1[2], l1[3]),
                      (l2[0], l2[1]), (l2[2], l2[3])])
    pts1 = get_correct_config(pts1)

    # pts2 = np.float32([(100, 100), (100, 100 + height),
    #                   (100 + width, 100), (100 + width, 100 + height)])

    # pts2 = np.float32([(100 + width, 100), (100 + width, 100 + height),
    #                    (100, 100), (100, 100 + height)])

    pts2 = get_corresponding_pts(pts1, width, height)

    debug_img = np.copy(img)
    draw_lines(debug_lines, debug_img)
    # print(img.shape)
    # print(width)
    # print(height)
    # print(pts1.astype(np.int64))
    # print(pts2.astype(np.int64))
    # cv2_imshow(debug_img)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return (M, debug_img, debug_lines, (int(100 + width + 100), int(100 + height)))
