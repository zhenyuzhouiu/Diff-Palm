import numpy as np

class border:
    """
    By given the polynomial parameters and the start point and end point,
    it uses the slope of each sample point to calcuate the y+bias and y-bias.
    The bias is calculated by 0.08 / np.sin(self.calculate_tangent_angle(x)).

    Note:
        This module without any random sample function.
    """
    def __init__(self, param, lower_bound, upper_bound):
        self.param = param  # param中保存2阶多项式参数,即数列中保存三个数
        self.upper = upper_bound  # 边界在x轴上的上界
        self.lower = lower_bound  # 边界在x轴上的下界

    def backbone_fundation(self, x):  # 多项式方程，输入x，返回y
        """
        Just is a polynomial function
        """
        y = sum(param * x ** (len(self.param) - 1 - idx) for idx, param in enumerate(self.param))
        return y

    def calculate_tangent_angle(self, x_val):
        slope = 0
        if x_val == 0:
            slope = self.param[-2]
        else:
            for idx, p in enumerate(self.param):
                slope += (len(self.param) - 1 - idx) * p * x_val ** (len(self.param) - 2 - idx)
        # 计算切线方向向量
        tangent_vector = np.array([1, slope], dtype=float)
        # 计算垂直方向向量
        vertical_vector = np.array([0, 1], dtype=float)

        # 计算切线方向与垂直方向的夹角（弧度）
        angle_rad = np.arctan2(tangent_vector[1], tangent_vector[0]) - np.arctan2(vertical_vector[1],
                                                                                  vertical_vector[0])
        return angle_rad

    def border_define(self, step=0.05, is_branch=False):  # 定义多边形边界
        """
        The first half part of border save the (x, y+bias), and the second half part of border save the (x, y-bias)
        """
        # step表示多远进行一次边界点设置
        x = self.lower
        point_set = []  # 记录边界上的点集
        while x <= self.upper:
            y = self.backbone_fundation(x)
            # if is_branch:
            #     bias = 0.1
            # else:
            bias = 0.08 / np.sin(self.calculate_tangent_angle(x))  # np.sin函数的参数是弧度，而不是角度
            point_set.append((x, y + bias))
            point_set.append((x, y - bias))
            x = round(x + step, 2)
        border = [0] * len(point_set)  # 将点集重新排列，形成边界
        i = 0
        while i < len(point_set) / 2:
            border[i] = point_set[2 * i]
            border[-i - 1] = point_set[2 * i + 1]
            i = i + 1
        return border
