"""Модуль класса входных данных
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import reduce
from typing import Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import RootResults, root_scalar

from .formulas import dh, my_z, ob_raskh, plot
from .gdh import get_gdh_curvs
from .header import BaseCollection, BaseStruct, Header, get_format_by_key
from .limit import Limit
from .mode import Mode
from .spch import Spch
from .summary import CompSummary, StageSummary
from .weight import DEFAULT_BORDER, Border, Weight
from .stage import Stage



class Comp(BaseCollection[Stage]):
    """Класс компановки ДКС
    """
    def __init__(self, stages:Union[Stage, Iterable[Stage]]):
        """Конструктор компановки ДКС

        Args:
            stages (Union[Stage, Iterable[Stage]]): Ступень(и) ДКС
        """
        super().__init__(stages)
    @property
    def w_cnt_calc(self)->List[int]:
        return  [
            stage.w_cnt
        for stage in self]
    @w_cnt_calc.setter
    def w_cnt_calc(self, value):
        for w_val, stage in zip(value, self):
            stage.w_cnt_current = w_val
    def calc_comp_summary(self, mode: Mode, freqs:Iterable[float], border_list:Iterable[Border]=DEFAULT_BORDER)->CompSummary:
        """Расчет работы компановки

        Args:
            mode (Mode): Режим работы
            freqs (Iterable[float]): Частота(ы), об/мин
                (Количесвто элементов списка должно совпадать с количесвтом ступеней)

        Returns:
            CompSummary: возврощяет показатель работы компановки
        """
        assert len(freqs) == len(self
            ),"(Количесвто элементов списка частот должно совпадать с количесвтом ступеней)"
        _p_in = mode.p_input
        _t_in = mode.t_in
        _res = []
        for stage, freq in zip(self, freqs):
            _res.append(
                stage.calc_stage_summary_in(
                    Mode(_t_in, mode.q_in[stage.idx], _p_in), freq
                )
            )
            _p_in = _p_in * _res[-1].comp_degree - stage.lim.dp_avo
            _t_in = stage.lim.t_avo
        return CompSummary(_res, mode, border_list)
    def calc_comp_summary_out(self, mode:Mode, freqs:Iterable[float], border_list:Iterable[Border]=DEFAULT_BORDER)->CompSummary:
        cur_mode = mode.clone()
        res = []
        for st, freq in list(zip(self, freqs))[-1::-1]:
            stage:Stage = st
            cur_mode.t_in = stage.lim.t_avo if stage.idx >= 1 else mode.t_in
            res.append(stage.calc_stage_summary_out(cur_mode, freq))
            cur_mode.p_input = res[-1].p_in + stage.lim.dp_avo
        res.reverse()
        return CompSummary(res, mode, border_list)
    def get_freq_bound_min_max(self, mode:Mode, all_freqs:List[float])->List[Tuple[float,float]]:
        assert len(all_freqs) == len(self
            ),"(Количесвто элементов списка частот должно совпадать с количесвтом ступеней)"
        res = [self[0].get_freq_min_max(mode)]
        p_in = mode.p_input
        t_in = mode.t_in
        for idx_stage, stage in list(enumerate(self))[1:]:
            prev_summ = self[idx_stage-1].calc_stage_summary_in(
                mode=Mode( t_in, mode.q_in[stage.idx], p_in), freq=all_freqs[idx_stage-1])
            p_in = p_in * prev_summ.comp_degree - stage.lim.dp_avo
            t_in = stage.lim.t_avo
            res.append(stage.get_freq_min_max(Mode(t_in, mode.q_in[stage.idx], p_in)))
        return res

    def show_plt(self, mode:Mode, f_max:float, f_min:float, summ:StageSummary):
        stage_1:Stage = self._list_items[0]
        f_dim_max = f_max / stage_1.type_spch.fnom
        f_dim_min = f_min / stage_1.type_spch.fnom
        stage_1.show_plt(mode.t_in, f_dim_max, f_dim_min, summ)
    
    def get_freq_bound_min_max_out(self, mode: Mode, list_border:List[Border], all_freqs:List[float])->List[Tuple[StageSummary,StageSummary]]:
        summs = self[-1].get_freq_min_max_out(mode, list_border[-1])
        res = [summs]
        cur_mode = mode.clone()
        for stage_idx, st in list(enumerate(self))[-2::-1]:
            prev_summ = self[stage_idx+1].calc_stage_summary_out(cur_mode, all_freqs[stage_idx+1])
            cur_mode.p_input = prev_summ.p_in + st.lim.dp_avo
            res.append(self[stage_idx].get_freq_min_max_out(cur_mode, list_border[stage_idx]))
        res.reverse()
        return res
