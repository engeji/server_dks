"""[summary]
"""
from .spch_init import SpchInit
from .formulas import my_z
from typing import Tuple, List
import math
class Spch(SpchInit):
    def __init__(self,spch:SpchInit):
        self.spch = spch
    def __getattr__(self, attr):
        return getattr(self.spch, attr)
    def koef_raskh_by_percent_x(self, percent_x:float)->float:
        koef_min, koef_max = (
            self.min_k_raskh,
            self.max_k_raskh)
        return koef_min + (koef_max - koef_min) * (percent_x / 100)

    def percent_x_by_k_raskh(self, k_raskh):
        koef_min, koef_max = (
            self.min_k_raskh,
            self.max_k_raskh)
        return (k_raskh - koef_min) / (koef_max - koef_min) * 100
    def freq_by_percent(self, volume_rate:float, percent_x:float)->float:
        koef_raskh = self.koef_raskh_by_percent_x(percent_x)
        u_val = 4 * volume_rate / (math.pi * (self.d_val ** 2)) / koef_raskh
        # u = freq * self.d_val * math.pi / 60.0
        return u_val / (self.d_val * math.pi)
    def volume_rate_by_freq_and_koef_raskh(self, freq:float, koef_raskh:float)->float:
        u_val = self.vel(freq)
        # koef_raskh = 4 * ob_raskh(q_in, p_in, t_in, r_val, plot_std=plot_std) / 60. / (
        #     math.pi * (self.d_val ** 2) * self.vel(freq))
        return koef_raskh / 4. * (math.pi * (self.d_val **2) * u_val)
    def get_freq_by_vel(self, vel:float)->float:
        return vel / math.pi / self.d_val
    def calc_y(self, volume_rate:float, k_raskh:float, z_in:float, r_val:float, t_in:float, k_val:float)->float:
        percent_x = self.percent_x_by_k_raskh(k_raskh)
        freq = self.freq_by_percent(volume_rate, percent_x)
        koef_nap = self.calc_k_nap(k_raskh)
        u_val = self.vel(freq) / 60.
        cur_dh = koef_nap * u_val * u_val
        kpd = self.calc_k_kpd(k_raskh)
        m_t =  (k_val - 1) / (k_val * kpd)
        point_y = (cur_dh * m_t / (z_in * r_val * t_in) + 1) ** (1 / m_t)
        return point_y
    def calc_xy(self, freq:float, k_raskh:float, z_val:float,
        r_val:float, t_in:float, k_val:float)->Tuple[float]:
        """Расчет точки на ГДХ

        Args:
            freq (float): Частота, об/мин
            k_raskh (float): Коэф-т расхода, б.м
            z_val (float): Сверхсжимаемость, д.ед
            r_val (float): Газовая постоянная,
            t_in (float): Температура на входе, К
            k_val (float): Коэф-т политропы, б.м.

        Returns:
            Tuple[float]: точка на ГДХ (
                [0]: Обьёмный расход, м3/мин,
                [1]: степень сжатия, д.ед
            )
        """
        point_x = self.calc_x(freq, k_raskh)
        # u_val = self.vel(freq)
        # koef_nap = self.calc_k_nap(k_raskh)
        # cur_dh = koef_nap * u_val * u_val / 60. /60.
        # kpd = self.calc_k_kpd(k_raskh)
        # m_t =  (k_val - 1) / (k_val * kpd)
        point_y = self.calc_y(point_x, k_raskh, z_val, r_val, t_in, k_val)
        return point_x, point_y
    def calc_x(self, freq, k_raskh:float)->float:
        u_val = self.vel(freq)
        point_x = k_raskh * math.pi * (self.d_val ** 2) * u_val  / 4
        return point_x
    def get_freq_bound(self, volume_rate:float)-> List[float]:
        """Возвращяет Tuple(max, min) частот для заданных условий всасывания

        Args:
            volume_rate (float): Обьемный расход, м3/сут
        Returns:
            List[float]: List(max, min) частот для заданных условий всасывания
        """
        return [
            4 * volume_rate / ((math.pi ** 2) * (self.d_val ** 3) * psi)
        for psi in (self.max_k_raskh, self.min_k_raskh)]
    @property
    def p_in(self):
        return  self.ptitle / 10. /  self.stepen
    def get_volume_rate_by_freq_and_perc(self, freq:float, percent_x:float)->float:
        k_raskh = self.koef_raskh_by_percent_x(percent_x)
        return k_raskh * math.pi * (self.d_val ** 2) * self.vel(freq) / 4.
    def get_N1_p_in(self, freq:float, koef_raskh:float, z_in:float, r_val:float, t_in:float)->float:
        u_val = self.vel(freq) / 60
        koef_nap = self.calc_k_nap(koef_raskh)
        kpd = self.calc_k_kpd(koef_raskh)
        return (u_val ** 3) * koef_nap * koef_raskh * math.pi * (self.d_val**2) / (
            4 * z_in * r_val * t_in * kpd
        ) * 10 ** 3
