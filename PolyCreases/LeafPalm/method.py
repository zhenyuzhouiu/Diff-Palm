import concurrent.futures
import math
import random

import cv2
import numpy as np


class Node:
    def __init__(self, position):
        self.position = np.array(position)  # 点的坐标
        self.son = None
        self.father = None
        self.is_cut_start = False


def calculate_radian(param, x_val):
    """
    From the curent x_val point, it can estimate the slope of x_val point. 
    And calucate the radian angle difference between slope and +x axis.

    Args:
        param ():
        x_val ():
    Returns:
        angle_rad (): 
    """
    slope = 0
    for idx, p in enumerate(param):
        slope += (len(param) - 1 - idx) * p * x_val ** (len(param) - 2 - idx)
    # 计算切线方向向量
    tangent_vector = np.array([1, slope], dtype=float)
    # 计算垂直方向向量
    vertical_vector = np.array([0, 1], dtype=float)
    # 计算切线方向与垂直方向的夹角（弧度）
    angle_rad = np.arctan2(*tangent_vector) - np.arctan2(*vertical_vector)
    return angle_rad


def draw_Secondary_line(img: np.ndarray, 
                        length: float,
                        color: tuple,
                        thickness: int, 
                        curve_set):  # 随机绘制二级线
    curve = [curve for curve in curve_set if curve.id == 3][0]
    start_point = [random.randint(0, img.shape[0]), random.randint(0, img.shape[0])]
    dx = abs(curve.area_range[1] - curve.area_range[0])
    dy = abs(curve.poly(curve.area_range[1]) - curve.poly(curve.area_range[0]))
    tan_angle = dy / dx
    angle = np.arctan(tan_angle)
    angle_noise = np.random.randn() * 0.3
    angle = angle + angle_noise

    if random.randint(0, 1) == 1:
        angle = -(np.pi / 2 - angle)

    # 计算终止点的坐标
    end_point = (
        int(start_point[0] + length * np.cos(angle)),
        int(start_point[1] + length * np.sin(angle))
    )
    # 使用cv2.line()函数绘制直线
    cv2.line(img, start_point, end_point, color, thickness)

    return start_point, end_point



def get_branch_equ(start_x, start_y, angle):  # 根据起始点与方向，计算分叉的一阶多项式
    """
    From a start point and a random generated angle, it can return the slope and c
    Args:
        start_x (float):
        start_y (float):
        angle (float):
    Returns:
        [slope, c] (tuple): return the slope and the intercept (the point at which a line crosses the x- or y-axis)

    """
    angle_rad = math.radians(angle)
    # 计算斜率
    slope = math.tan(angle_rad)
    # 使用斜截式方程 y = mx + c，将斜率和点 (x1, y1) 替换进去，解出截距 c
    c = start_y - slope * start_x
    # 返回斜截式的方程
    return [slope, c]


def shift_points(points, shift_x, shift_y, angle_degrees, center=(0, 0)):
    # 上下左右偏移,以及旋转
    angle_radians = np.radians(angle_degrees)
    for point in points:
        x, y = point.position
        point.position = [x + shift_x, y + shift_y]  # 平移
        x, y = point.position
        point.position = [  # 旋转
            center[0] + (x - center[0]) * np.cos(angle_radians) - (y - center[1]) * np.sin(angle_radians),
            center[1] + (x - center[0]) * np.sin(angle_radians) + (y - center[1]) * np.cos(angle_radians)
        ]
