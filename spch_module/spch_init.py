"""Модуль для класса СПЧ
"""
import math
import re
from itertools import groupby
from types import SimpleNamespace
from typing import Tuple, List

import numpy as np
from scipy.optimize import root 
from .formulas import dh, my_z, ob_raskh, calc_c00
from .limit import DEFAULT_LIMIT, Limit
from .mode import Mode

class SpchInit:
    """Класс для Сменной проточной части (СПЧ)
    """
    def _init_from_txt(self, text, title):
        self.name = title
        _, head, data = re.split(r'HEAD\n|\nDATA\n', text)
        [
            self.mgth,
            self.stepen,
            self.r_val,
            self.t_val,
            self.ptitle,
            self.ppred,
            self.fnom,
            z_val
        ] = [
                float(next(filter(lambda x: x[0]==letter, [line.split('\t')
            for line in head.split('\n')]))[1])
        for letter in 'mght eps R T p_title Ppred fnom Z'.split()]
        self.d_val = .8
        dic_header = {head:ind for ind, head in enumerate(data.split('\n')[0].split('\t'))}
        all_points = [
            SimpleNamespace(
                q = float(line.split('\t')[dic_header['q']]),
                kpd = float(line.split('\t')[dic_header['kpd']]),
                freq = float(line.split('\t')[dic_header['f']]) * self.fnom,
                comp = float(line.split('\t')[dic_header['comp']]),
            )
        for line in data.split('\n')[1:]]
        self._freq, self._x_raskh, self._y_kpd, self._y_nap = np.array([(
            x.freq,
            self.koef_raskh_by_volume_rate(x.q, x.freq),
            x.kpd,
            self.koef_nap_by_comp(x.comp, x.freq, self.t_val, self.r_val, x.kpd, z_val),
        ) for x in all_points], dtype=object).T

    def __init__(self, sheet=None, text=None, title=None):
        self._c00_kpd = self._c00_nap = ()
        self.r_val=self.t_val=self.p_val=self.stepen=.0
        self.d_val=self.ppred=self.mgth=self.ptitle=self.fnom=.0
        self.name = ''
        if text is not None:
            self._init_from_txt(text, title)
        elif sheet is not None:
            self._init_from_xl(sheet)
        self._min_k_raskh, self._max_k_raskh = (
            f(self._x_raskh) for f in (min, max)
        )

    def _init_from_xl(self, sheet):
        """инициализация

        Args:
            sheet (xlrd.sheet.Sheet): Лист из базы данных
        """
        self.name = sheet.name
        lines = [[coll.value for coll in list(row)] for row in list(sheet.get_rows())]
        self.ind_end_points = [line[0] for line in lines].index('//')
        all_points = []
        for ind in range(self.ind_end_points):
            if lines[ind][0] != '/':
                all_points += [SimpleNamespace(
                    Q = lines[ind][0],
                    PinMPa = lines[ind][1],
                    kpd = float(lines[ind][4]),
                    freq = float(lines[ind][5]))]

        all_attribs = {
            lines[ind][0]:lines[ind][1]
        for ind in range(self.ind_end_points+1, len(lines))}
        self.t_val = all_attribs['T']
        self.r_val = all_attribs['R']
        self.p_val = all_attribs['P']
        self.d_val = all_attribs['d']

        for letter in ('stepen', 'ppred', 'mgth', 'ptitle', 'fnom'):
            self.__setattr__(letter, all_attribs[letter])
        self.ptitle = float(self.ptitle)
        self._x_raskh, self._y_nap, self._y_kpd, self._freq = np.array([[
            self.koef_raskh(x.Q, x.PinMPa, x.freq, self.t_val, self.r_val, DEFAULT_LIMIT.plot_std),
            self.koef_nap(x.PinMPa,
                self.p_val, x.freq, self.t_val, self.r_val, x.kpd),
            x.kpd,
            x.freq
        ] for x in all_points], dtype=object).T
    def calc_k_nap(self, k_raskh:float, power:int=5)->float:
        """Расчет коеф-та напора по тренду

        Args:
            k_raskh (float): коэф-т расхода, д.ед
            power (int, optional): Степень полинома. Defaults to 5.

        Returns:
            float: Коэф-т напора в зависимости от коэф-та расхода
        """
        if len(self._c00_nap) != power:
            self._c00_nap = calc_c00(self._x_raskh, self._y_nap, power)
        return sum(c00 * (k_raskh ** n) for n, c00 in enumerate(self._c00_nap))

    def calc_k_kpd(self, k_raskh:float, power:int=5)->float:
        """Расчет политропного кпд по тренду

        Args:
            k_raskh (float): коэф-т расхода, д.ед
            power (int, optional): Степень полинома. Defaults to 5.

        Returns:
            float: политропный кпд в зависиммости от коэф-та расхода
        """
        if len(self._c00_kpd) != power:
            self._c00_kpd = calc_c00(self._x_raskh, self._y_kpd, power)
        return sum(c00 * (k_raskh ** n) for n, c00 in enumerate(self._c00_kpd))

    def vel(self, freq:float)->float:
        """Линейная скорость вращения центробежного колеса

        Args:
            freqVal (float): Частота, об/мин

        Returns:
            float: Возврощяет скорость газа, при вращении центробежного колеса, м/мин
        """
        return freq * self.d_val * math.pi

    def koef_raskh_by_volume_rate(self, volume_rate:float, freq:float)->float:
        """Коэффициент расхода из обьемного расхода

        Args:
            volume_rate (float): обьемный расход, при заданных условиях, м3/мин
            freq (float): Частота, об/мин

        Returns:
            float: float: Возврощяет коеффициент расхода,
                при заданных условиях и текущей температуре, д.ед
        """
        return 4 * volume_rate  / (math.pi * (self.d_val ** 2) * self.vel(freq))

    def koef_raskh(self, q_in, p_in, freq, t_in, r_val, plot_std):
        """Коэффициент расхода

        Args:
            q_in (float): комерческий расход, млн.м3/сут
            p_in (float): Давление входа, МПа
            freq (float): Частота, об/мин
            t_in (float): Температура входа, К
            r_val (float, optional): постоянная больцмана поделеная на молярную массу
            plot_std (float, optional): Плотность при стандартных условиях, кг/м3

        Returns:
            float: Возврощяет коеффициент расхода, при заданных условиях и текущей температуре, д.ед
        """
        return 4 * ob_raskh(q_in, p_in, t_in, r_val, plot_std=plot_std) / (
            math.pi * (self.d_val ** 2) * self.vel(freq))

    def koef_nap(self, p_in:float, p_out:float, freq:int,
        t_in:float, r_val:float, kpd:float)->float:
        """Коэф-т напора в зависиморсти от условий всасывания

        Args:
            p_in (float): Давление входа, МПа
            p_out (float): Давление выхода, МПа
            freq (int): Частота, об/мин
            t_in (float): Температура входа, К
            r_val (float): постоянная больцмана поделеная на молярную массу
            kpd (float): Политропнйы КПД, д.ед

        Returns:
            float: Возврощяет коеффициент напора, при заданных условиях и текущей температуре, д.ед
        """
        z_val = my_z(p_in, t_in)
        dh_val = dh(p_out/p_in, z_val, t_in, r_val, DEFAULT_LIMIT.k_val, kpd)
        v_val = self.vel(freq) / 60.
        return dh_val / (v_val ** 2)
    def koef_nap_by_comp(self, comp:float, freq:int,
        t_in:float, r_val:float, kpd:float, z_val:float)->float:
        """Коэф-т напора в зависиморсти от степени сжатия и z

        Args:
            comp (float): Степень сжатия, д.ед
            freq (int): Частота, об/мин
            t_in (float): Температура входа, К
            r_val (float): постоянная больцмана поделеная на молярную массу
            kpd (float): Политропнйы КПД, д.ед
            z (float): Сверхсжимаемость, д.ед

        Returns:
            float: Возврощяет коеффициент напора, при параметрах, д.ед
        """
        dh_val = dh(comp, z_val, t_in, r_val, DEFAULT_LIMIT.k_val, kpd)
        v_val = self.vel(freq) /60.
        return dh_val / (v_val ** 2)

    @property
    def min_k_raskh(self):
        """Минимальный коэф-т расхода
        """
        #FIXME: setborder_avarage
        return self._min_k_raskh

    @property
    def max_k_raskh(self):
        """Максимальный коэф-т расхода
        """
        return self._max_k_raskh

    
    def get_no_dim_fact_points(self):
        """Возвращяет python'like json структуру точек для гдх
        """
        fact_points = groupby(zip(self._freq, self._x_raskh,
            self._y_kpd, self._y_nap), lambda p: p[0])
        return{
            'no_dim':{
                'datasets':[{
                    'data':[{
                        'x': p[1],
                        'y': p[3],
                        'type': 'no_dim'
                    } for p in points],
                    'my_type': 'nodim_nap'
                }for freq, points in fact_points]
            }
        }

    def __repr__(self):
        return f'ГПА{self.mgth:.0f}-{self.ptitle:.0f} {self.stepen}'
    @property
    def short_name(self):
        return f'{self.mgth:.0f}/{self.ptitle:.0f}-{self.stepen}'
    def __format__(self, fmt):
        res = f'{self.short_name}'
        return f'{res:{fmt}}'