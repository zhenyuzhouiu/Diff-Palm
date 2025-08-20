from statistics import mean
import numpy as np
from numpy import array


class Inter_class_filter:  # 判断两特征是否存在差异
    def __init__(self, d_threshold, filter_method='max', keypoint_num=5):
        self.threshold = d_threshold
        self.filter_method = filter_method
        self.num = keypoint_num

    def is_exist_difference(self, palm_feature_1, palm_border_1, palm_feature_2, palm_border_2):
        curve_difference = []
        for i in range(len(palm_feature_1)):  # 遍历手掌的每条线
            param_set = [palm_feature_1[i], palm_feature_2[i]]
            border_set = [palm_border_1[i], palm_border_2[i]]
            difference = self.get_difference(param_set, border_set, self.num)
            curve_difference.append(difference)
        # print(curve_difference)
        if self.filter_method == 'max':  # 只需要差异最大的大于阈值，即只有一条线不同就认为类内存在差异
            max_difference = max(curve_difference)
            if max_difference > self.threshold:  # 类内存在差异
                return True
            else:
                return False
        if self.filter_method == 'avg':  # 需要差异的均值大于阈值
            avg_difference = mean(curve_difference)
            # print(avg_difference)
            if avg_difference > self.threshold:  # 类内存在差异
                return True, avg_difference
            else:
                return False, avg_difference

    def get_difference(self, param_set, border_set, num):  # 计算两条线的差异
        y_set_1 = self.get_keypoint_y(param_set[0], border_set[0], num)
        y_set_2 = self.get_keypoint_y(param_set[1], border_set[1], num)
        difference_set = [abs(y_set_1[i] - y_set_2[i]) for i in range(len(y_set_1))]
        difference = sum(difference_set)/num
        return difference

    def get_keypoint_y(self, param, border, num):  # 计算一条线上num个均分点的y值
        poly = np.poly1d(param)
        keypoint_y = []
        distance = abs(border[1] - border[0])
        interval = distance / num
        for i in range(num):
            x = border[0] + interval * i
            keypoint_y.append(poly(x))
        return keypoint_y
