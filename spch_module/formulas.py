"""Модуль для всяких разных формул
"""
from typing import Iterable, List
import numpy as np
from numpy import linalg as LA
def calc_c00(x_val:Iterable[float], y_val:Iterable[float], power:int)->List[float]:
    """[summary]

    Args:
        x_val (Iterable[float]): Массив ординат
        y_val (Iterable[float]): Массив абсцис
        power (int): Показатель максимальной степени

    Returns:
        List[float]: Полиномиальные коэф-ты, начиная с 0
    """
    a_matrix = np.vstack([[v ** p for p in range(power)] for v in x_val])
    return LA.lstsq(a_matrix, np.array(y_val, dtype=float), rcond=None)[0]

def my_z(p, t, t_krit=190, p_krit=4.6):
    return 1-0.427*p/p_krit*(t/t_krit)**(-3.688)
def my_z2(p, t, t_krit=190, p_krit=4.6):
    """Численный расчет сверхсжимаемости через
     разложение в ряд тейлора уравнение пенга-робинсона
    с точностью до 2 слогаемого

    Args:
        p (float): Давление, МПА
        t (float): Температура, К
        t_krit (float, optional): Критич. Температура, К {default = 190}
        p_krit (float, optional): Критич. Давление, МПа {default = 4.6}

    Returns:
        float: Значение сверхсжимаемости Z
    """
    Pk = p_krit  * (10 ** 6)
    Tk = t_krit
    P = p * (10 ** 6)
    R = 8.31
    a = 0.42748 * R * R * (Tk ** 2.5) / Pk
    b = 0.0864 * R * Tk / Pk
    A = a * P / R / R / (t ** 2.5)
    B = b * P / R / t
    A3 = 1
    A2 = -1
    A1 = A - B * B - B
    A0 = -A * B
    x1 = 1
    B2 = 0
    B1 = 0
    B0 = 0
    disc = 0
    for _ in range(2):
        B2 = (6 * A3 * x1 + 2 * A2) / 2
        B1 = 3 * A3 * x1 * x1 + 2 * A2 * x1 + A1
        B0 = A3 * x1 * x1 * x1 + A2 * x1 * x1 + A1 * x1 + A0
        disc = B1 * B1 - 4 * B2 * B0
        x1 = (-B1 + (disc ** 0.5)) / 2 / B2 + x1
    return x1

def dh(comp_degree, z, t_in, R, k, kpd):    
    """удельное изменение энтальпии

    Args:
        comp_degree (float): Степень сжатия, д.ед
        z (float): сверхсжимаемлость, д.ед
        t_in (float): Давление начала процесса сжатия, К
        R (float): постоянная больцмана поделеная на молярную массу       
        k (float): Коэф-т политропы, д.ед
        kpd (float): политропный кпд, д.ед
    Returns:
        dh (float): необходимое для сжатия измение энтальпии, дж/кг
    """    
    kv = kpd * k / (k-1)
    # kv =  k / (k-1)
    return kv * z * R * t_in * (comp_degree ** (1 / kv ) - 1)               

def plot(p, t, R, t_krit=190, p_krit=4.6):
    """плотность газа, кг/м3

    Args:
        p (float): Давление, МПа
        t (float): Температура, К
        R (float): Постоянная Больцмана поделеная на молярную массу.
        t_krit (float, optional): Критич. Температура, К {default = 190}
        p_krit (float, optional): Критич. Давление, МПа {default = 4.6}
    Returns:
        float: плотность газа при указанные давлении и температуры, дж/кг
    """
    return p * (10 ** 6) / my_z(p,t, t_krit, p_krit) / R / t
def p_z(plot, R, t_in):
    return plot * R * t_in / (10**6)
def ob_raskh(Q, p_in, t_in, R, plot_std=0.692):
    """Обьемный расход, м3/мин

    Args:
        Q (float): Комерческий расход, млн. м3/сут
        p_in (float): Текущее давление, МПа
        t_in (float): Текущяя температура, К
        R (float): Постоянная Больцмана поделеная на молярную массу.
        plot_std (float, optional): стандартная плотность, кг/м3. Defaults to 0.698.

    Returns:
        float: Возвращяет обьемный расход, при указанных условиях, м3/мин
    """
    q_m3_min = Q * (10 ** 6) / 24.0 / 60.0
    return q_m3_min * plot_std / plot(p_in, t_in, R)
def q_in(volume_rate:float, p_in:float, t_in:float, R:float, plot_std:float=.692)->float:
    """Комерческий расход, млн.м3/суь

    Args:
        volume_rate (float): Объемный расход, м3/мин
        p_in (float): Текущее давление, МПа
        t_in (float): Текущяя температура, К
        R (float): Постоянная Больцмана поделеная на молярную массу.
        plot_std (float, optional): стандартная плотность, кг/м3. Defaults to 0.698.

    Returns:
        float: Возврощяет комперческий расход относительно обьемного, млн. м3/сут
    """    
    q_mln = volume_rate/ ((10 ** 6) / 24.0 / 60.0)
    return q_mln / plot_std * plot(p_in, t_in, R)

