import math
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2
import numpy as np
from shapely import unary_union
from shapely.geometry import Polygon
from LeafPalm.method import *
from LeafPalm.border import *
import concurrent.futures


class TheLine:
    def __init__(self, id, initial_point):
        self.id = id  # 区域中的细线序号
        self.initial_point = initial_point
        self.certain_points = []
        self.certain_points.append(self.initial_point)


class Curve:
    def __init__(self, id, param, area_range, grow_direction, noise_weight, step=0.01, exist_branch=False, is_prolong=False,
                 is_cut=False):
        self.id = id  # 掌线序号
        self.param = param  # 多项式参数
        self.poly = np.poly1d(self.param)  # 多项式对象
        self.derivative = np.polyder(self.poly)  # 导数对象
        self.grow_direction = grow_direction  # 掌线生长方向
        self.area_range = area_range  # 掌线生长区域
        self.step = step  # 点的步长
        self.noise_weight = noise_weight
        self.branch_param = None
        if exist_branch:  # 若存在分叉，则计算分叉参数（一阶）
            self.start_x = random.uniform(min(area_range) + (max(area_range) - min(area_range)) / 3,  # 1/3-2/3
                                          max(area_range) - (max(area_range) - min(area_range)) / 3)
            start_y = sum(p * self.start_x ** (len(param) - 1 - idx) for idx, p in enumerate(param))
            angle = random.randrange(-60, -40, 1)  # 分叉角度
            self.branch_param = get_branch_equ(self.start_x, start_y, angle)  # 计算分叉参数
            self.is_prolong = is_prolong  # 是否延长
        self.is_cut = is_cut  # 是否截断
        self.branch_curve = None  # branch_curve
        self.Lines = self.init_lines(4)  # 掌线中的细纹列表,每条line单独生长
        self.backbone_area = None  # 主干多项式区域
        self.branch_area = None  # 分叉多项式区域
        self.area = self.get_area(area_range, self.branch_param)  # 整体多项式区域

    def get_area(self, area_range, branch_param):  # 获取区域，若存在分叉，则同时设置分叉curve
        # 根据给定的多项式方程以及每个点的斜率，计算采样点的上边界和下边界
        area_maker = border(self.param, area_range[0], area_range[1])
        self.backbone_area = Polygon(area_maker.border_define())
        if branch_param is not None:  # 确定分叉区域
            branch_upper = random.uniform(self.start_x + (max(area_range) - self.start_x) / 3,  # 1/3-3/4
                                          max(area_range) - (max(area_range) - self.start_x) / 4)
            if self.is_prolong:
                self.start_x = random.uniform(min(area_range),
                                              min(area_range) + (max(area_range) - min(area_range)) / 3)  # 0-1/3
            branch_maker = border(branch_param, lower_bound=self.start_x, upper_bound=branch_upper)
            self.branch_area = Polygon(branch_maker.border_define(is_branch=True))
            self.branch_curve = Curve(-1, branch_param, [self.start_x, branch_upper], grow_direction=1, noise_weight=self.noise_weight)
            return unary_union([self.backbone_area, self.branch_area])
        return self.backbone_area

    def init_lines(self, line_amount: int):  # 定义curve中的若干条line
        """
        For enhancing the reality of principle line, 
        it construct line_amount random bias line for a principle line.

        Args:
            line_amount (int):

        Returns:
            Lines (tuple[TheLine, TheLien, ...]):

        """
        if self.grow_direction == 1:  # 从左向右生长
            initial_y = self.poly(self.area_range[0])
            initial_point = [self.area_range[0], initial_y]
        else:
            initial_y = self.poly(self.area_range[1])
            initial_point = [self.area_range[1], initial_y]
        Lines = []
        if self.id == -1:  # 分叉
            for i in range(line_amount):
                Line = TheLine(i, Node(initial_point))  # 初始化Line
                Lines.append(Line)
        else:  # 非分叉
            for i in range(line_amount):
                half_height = 0.008 / np.sin(calculate_radian(self.param, self.area_range[0]))  # 可接受的偏移量
                random_bias = np.random.uniform(-half_height, half_height)  # 取随机y值
                initial_point = [initial_point[0], initial_point[1] + random_bias] # 根据当前点加点y轴上的噪声
                Line = TheLine(i, Node(initial_point))  # 初始化Line
                Lines.append(Line)
        return Lines

    def process_curve(self, curve):  # 单个curve生长
        self.growing(curve)
        if curve.is_cut:
            curve_cutting_x = random.uniform(       # 1/3-2/3
                min(curve.area_range) + (max(curve.area_range) - min(curve.area_range)) / 3,
                max(curve.area_range) - (max(curve.area_range) - min(curve.area_range)) / 3)
            # curve_cutting_x = curve.start_x-0.01
            curve_displace_points = []
            if curve.id == 1:
                for line in curve.Lines:
                    line_displace_points = [point for point in line.certain_points if
                                            point.position[0] < curve_cutting_x]

                    curve_displace_points.extend(line_displace_points)
                shift_points(curve_displace_points, random.uniform(-0.2, -0.1), random.uniform(-0.1, 0.1),
                             random.uniform(-20, 20), curve_displace_points[0].position)
            elif curve.id == 2:
                for line in curve.Lines:
                    line_displace_points = [point for point in line.certain_points if
                                            point.position[0] > curve_cutting_x]
                    curve_displace_points.extend(line_displace_points)
                if curve.branch_param is not None and not curve.is_prolong and curve_cutting_x <= curve.start_x:
                    for line in curve.branch_curve.Lines:
                        line_displace_points = line.certain_points
                        curve_displace_points.extend(line_displace_points)
                shift_points(curve_displace_points, random.uniform(0.1, 0.2), random.uniform(-0.1, 0.1),
                             random.uniform(-20, 20), curve_displace_points[0].position)
        return curve

    def growing(self, curve):  # curve中所有line并行生长
        """
        Using multi-thread and partial, the process_line function 
        can support multiple input parameters for multi-thread
        """
        # ==== self.process_line(self, curve, line)需要输入两个参数， 
        # ==== 线程池只能喂一个实参，所以先把curve固定，得到只收line的单参数版本。
        # ==== map(func, iterable)
        # 定义一个部分应用了 self.process_line 方法的新函数
        # 把self.process_line预先固定curve实参
        process_line_partial = partial(self.process_line, curve)
        # 使用 ThreadPoolExecutor 并行处理每条线
        with ThreadPoolExecutor() as executor:
            executor.map(process_line_partial, curve.Lines)

    def process_line(self, curve, line: TheLine):  # 单条line生长
        """
        Based on the derivate function of the line, the derivate of x point can be calculate. 
        The next point (x, y) will be calcualte by the step and theta and noise theta, and be 
        saved at the line.certain_points.

        Args:
            curve (Curve):
            line (TheLine):
        
        """
        is_continue = True
        step = curve.step  # 生长步长
        while is_continue:
            x = line.certain_points[-1].position[0]  # 当前节点的y值
            y = line.certain_points[-1].position[1]  # 当前节点的y值
            theta = np.arctan(curve.derivative(x))
            if curve.grow_direction == 0: # 
                theta = -theta

            theta_noise = np.random.randn() * self.noise_weight

            y_0 = curve.poly(line.certain_points[-1].position[0])  # 当前节点的标准y值
            dist = y - y_0 # y, which is saved on the TheLine, has some random noise 
            threshold = 0.01 / np.sin(calculate_radian(curve.param, x))  # 生长阈值
            if abs(dist) > threshold:  # 当前节点越界
                if dist > 0:  # 在标准曲线上方，向下生长
                    theta = theta - np.abs(theta) * np.abs(theta_noise)
                else:  # 在标准曲线下方，向上生长
                    theta = theta + np.abs(theta) * np.abs(theta_noise)
            else:  # 在阈值内
                theta = theta + theta_noise
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            if curve.grow_direction == 0:
                new_x = x - step * cos_theta
            else:
                new_x = x + step * cos_theta
            new_y = y + step * sin_theta
            new_node = Node([new_x, new_y])
            line.certain_points.append(new_node)

            is_continue = ((curve.grow_direction == 1 and new_x < curve.area_range[1])
                           or (curve.grow_direction == 0 and new_x > curve.area_range[0]))


def show(curve_set, thickness=25):
    
    scale_factor = 3000
    image = np.ones((scale_factor, scale_factor, 3), dtype=np.uint8) * 255

    for curve in curve_set:
        for line in curve.Lines:
            # opencv
            points = []
            for point in line.certain_points:
                scaled_x = int(point.position[0] * scale_factor) #+ 400
                scaled_y = int(image.shape[0] - point.position[1] * scale_factor) #- 200
                points.append([scaled_x, scaled_y])
                
            cv2.polylines(image, [np.array(points)], False, (0, 0, 0), thickness=thickness)
   
    image_size = 512
    image = cv2.resize(image, [image_size,]*2)

    secondary_line_range = [4, 8]
    secondary_line_num = random.randint(secondary_line_range[0], secondary_line_range[1])
    secondary_line = []
    
    for i in range(secondary_line_num):
        length = np.random.uniform(0.1, 0.3) * image.shape[0]
        thickness = np.random.randint(2, 5)
        start_point, end_point = draw_Secondary_line(image, length, (0, 0, 0), thickness, curve_set)
        secondary_line.append(np.concatenate((np.array(start_point),
                                              np.array(end_point),
                                              np.array([thickness])), axis=0))
    
    return image, np.array(secondary_line)


def PalmGrow(feature, ranges, thickness, noise_weight, is_branch=False, is_prolong=False, curve_1_is_cut=False, curve_2_is_cut=False):
    """

    Args:
        feature (list): [[five parameters of polynominal], [five parameters], [five paramters]]
        ranges (list): [[start x, end x], [start x, end x], [start x, end x]]
        thickness (int):
        noise_weight (float):
        is_branch (bool):
        is_prolong (bool):
        curve_1_is_cut (bool):
        curve_2_is_cut (bool):

    Returns:
        image (np.ndarry):

    """
    curve_params = [feature[idx] for idx in [-1, -2, 0]]
    curve_ranges = [ranges[idx] for idx in [-1, -2, 0]]
    curve_param_1, curve_param_2, curve_param_3 = curve_params
    curve_range_1, curve_range_2, curve_range_3 = curve_ranges

    # Initialize the Curve class by using the polynomial parametets within the range of  start point and end point
    # And the Curve class will init_lines with 4 lines for on curve
    curve_1 = Curve(1, curve_param_1, curve_range_1,
                    grow_direction=0, noise_weight=noise_weight, is_cut=curve_1_is_cut)
    curve_2 = Curve(2, curve_param_2, curve_range_2,
                    grow_direction=1, noise_weight=noise_weight, exist_branch=is_branch,
                    is_prolong=is_prolong, is_cut=curve_2_is_cut)
    curve_3 = Curve(3, curve_param_3, curve_range_3,
                    grow_direction=1, noise_weight=noise_weight)

    if is_branch:
        curve_branch = curve_2.branch_curve
        curve_set = [curve_branch, curve_1, curve_2, curve_3]
    else:
        curve_set = [curve_1, curve_2, curve_3]

    # with the multi-thread, the initialized "Lines" of "TheLine" 
    # will append the sampled point along the polynominal 
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(curve.process_curve, curve)
                   for curve in curve_set]
        concurrent.futures.wait(futures)

    # 归一化
    for curve in curve_set:
        for line in curve.Lines:
            for node in line.certain_points:
                node.position[0] = (node.position[0] / 2.0) + 0.5
                node.position[1] = (node.position[1] / 2.0) + 0.5

    # 显示
    image, secondary_line = show(curve_set, thickness)
    return image, curve_set, secondary_line
