"""[summary]
"""
from dataclasses import dataclass
from itertools import chain
from typing import Iterable, Tuple, List, Union
from xml.dom import ValidationErr
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root, root_scalar

from .formulas import my_z, ob_raskh, plot, p_z, dh
from .gdh import get_gdh_curvs
from .header import BaseStruct
from .limit import Limit
from .mode import Mode
from .spch import Spch
from .summary import StageSummary
from .weight import Border


@dataclass
class Stage(BaseStruct):
    """Класс ступени ДКС
    """
    type_spch:Spch
    w_cnt:int
    lim:Limit
    idx:int=0
    _w_cnt_current:int=None
    @property
    def w_cnt_current(self):
        return self.w_cnt if self._w_cnt_current is None else self._w_cnt_current
    @w_cnt_current.setter
    def w_cnt_current(self, value):
        self._w_cnt_current = value
    def get_stage_summ(self, volume_rate, comp_degree, t_in, freq, p_out:float)->StageSummary:
        k_raskh = self.type_spch.koef_raskh_by_volume_rate(volume_rate, freq)
        kpd_pol = self.type_spch.calc_k_kpd(k_raskh)
        koef_nap = self.type_spch.calc_k_nap(k_raskh)
        dh_val = koef_nap * ((self.type_spch.vel(freq)/60.)**2)
        p_in = p_out/comp_degree
        mas_rate = volume_rate / 60. * plot(p_in, t_in, self.lim.r_val)
        kpd = self.type_spch.calc_k_kpd(k_raskh)
        mght = dh_val * mas_rate / kpd / (10**3)
        percent_x = (k_raskh - self.type_spch.min_k_raskh) / (self.type_spch.max_k_raskh - self.type_spch.min_k_raskh) * 100
        t_out = t_in * (comp_degree ** (self.lim.k_val - 1 ) / (self.lim.k_val * kpd_pol))-273.15
        return StageSummary(
            type_spch=self.type_spch, mght=mght,freq=freq, kpd=kpd_pol,
            p_in=p_in, comp_degree=comp_degree, w_cnt=self.w_cnt_current, p_out_req=p_out,
            volume_rate=volume_rate, percent_x=percent_x, t_out=t_out
        )
    def calc_stage_summary_in(self, mode:Mode, freq:float)->StageSummary:
            """Расчет одной ступени

            Args:
                mode (float): Режим работы всей ступени
                freq (float): Частота, об/мин

            Returns:
                StageSummary: Показатель работы ступени
            """
            q_one = mode.q_in[self.idx] / self.w_cnt_current
            k_raskh = self.type_spch.koef_raskh(
                q_in=q_one, p_in=mode.p_input, freq=freq,t_in=mode.t_in,
                r_val=self.lim.r_val, plot_std=self.lim.plot_std)
            z_in = my_z(mode.p_input, mode.t_in)
            volume_rate, comp_degree = self.type_spch.calc_xy(
                freq=freq, k_raskh=k_raskh, z_val=z_in, r_val=self.lim.r_val,
                t_in=mode.t_in,k_val=self.lim.k_val)
            return self.get_stage_summ(
                volume_rate=volume_rate, comp_degree=comp_degree, t_in=mode.t_in, freq=freq,
                p_out=comp_degree * mode.p_input
            )
    def calc_stage_summary_out(self, mode:Mode, freq:float, method:str='brentq', is_in_border:bool=False)->Union[None, StageSummary]:
        cur_mode = mode.clone()
        cur_mode.t_in = self.lim.t_avo if self.idx >= 1 else mode.t_in
        def func(p_in):
            cur_mode.p_input = p_in
            return self.calc_stage_summary_in(cur_mode, freq).p_out_req - mode.p_input
        bracket = [
            self.get_p_in_by_freq_and_perc_on_mode(mode, freq, 0),
            self.get_p_in_by_freq_and_perc_on_mode(mode, freq, 100),
        ]
        p_in_0 = sum(bracket) / 2.
        try:
            sol = root_scalar(
                f=func, method=method, x0=p_in_0, x1=p_in_0 * 1.0001, bracket=bracket
            )
            cur_mode.p_input = sol.root
            return self.calc_stage_summary_in(cur_mode, freq)
        except ValueError:
            if is_in_border:
                sol = root_scalar(
                    f=func, x0=p_in_0, x1=p_in_0 * 1.0001
                )
                cur_mode.p_input = sol.root
                return self.calc_stage_summary_in(cur_mode, freq)
            else:
                return None
    def get_freq_min_max(self, mode:Mode)->Tuple[float,float]:
        volume_rate = ob_raskh(mode.q_in[self.idx], mode.p_input, mode.t_in, self.lim.r_val, self.lim.plot_std) / self.w_cnt_current
        return self.type_spch.get_freq_bound(volume_rate)
    def get_freq_min_max_out(self, mode:Mode, border:Border)->Tuple[StageSummary,StageSummary]:
        assert any(list(map(lambda x: x.key == 'mght', border))), 'В граничных условиях нет мощности'
        curr_mode = mode.clone()
        curr_mode.t_in = mode.t_in if self.idx == 0 else self.lim.t_avo
        dispatcher_border = {
            'mght' : (self.get_perc_by_mght, self.get_mght_by_perc),
            'comp_degree' : (self.get_perc_by_comp, self.get_comp_by_perc),
            'freq_dim' : (self.get_perc_by_freq_dim, self.get_freq_dim_by_perc),
        }
        all_summs = []
        for weight in filter(lambda x: x.key != 'percent_x', border):
            def calc(val):
                curr_summ:StageSummary = dispatcher_border[weight.key][0](curr_mode, val)
                if curr_summ.percent_x < border['percent_x'].min_val:
                    return dispatcher_border[weight.key][1](curr_mode, border['percent_x'].min_val )
                elif curr_summ.percent_x > border['percent_x'].max_val:
                    return dispatcher_border[weight.key][1](curr_mode, border['percent_x'].max_val )
                else:
                    return curr_summ
            all_summs.append((
                calc(weight.min_val),
                calc(weight.max_val),
            ))
        res = [
            max(arr_summ, key=lambda summ: summ.freq) if idx == 0 else min(arr_summ, key=lambda summ: summ.freq)
        for idx, arr_summ in enumerate(zip(*all_summs))]
        return [
            summ if border.get_obj_val(summ) <.1 else None
        for summ in res]
        
    def get_freq_dim_by_perc(self, mode:Mode, percent_x:float)->Union[None,StageSummary]:
        koef_raskh = self.type_spch.koef_raskh_by_percent_x(percent_x)
        self.type_spch.get_volume_rate_by_freq_and_perc
        kpd = self.type_spch.calc_k_kpd(koef_raskh)
        koef_nap = self.type_spch.calc_k_nap(koef_raskh)
        q_in = mode.q_in[self.idx] / self.w_cnt_current
        mas_rate =  q_in * self.lim.plot_std * (10**6) / 24 / 60 / 60
        def func(freq):
            cur_mode = mode.clone()
            cur_mode.p_input = self.get_p_in_by_freq_and_perc_on_mode(mode, freq, percent_x)
            summ = self.calc_stage_summary_in(cur_mode, freq)
            return summ.p_out_req - mode.p_input
        x0_val = self.type_spch.fnom
        bracket = [.3*x0_val,1.3*x0_val]
        try:
            sol = root_scalar(f=func, x0=x0_val, x1=x0_val * 1.001, bracket=bracket, method='brentq')
            cur_mode = mode.clone()
            cur_mode.p_input = self.get_p_in_by_freq_and_perc_on_mode(mode, sol.root, percent_x)
            summ = self.calc_stage_summary_in(cur_mode, sol.root)
            return summ
        except ValueError:
            return None

    def get_comp_by_perc(self, mode:Mode, percent_x:float)->StageSummary:
        koef_raskh = self.type_spch.koef_raskh_by_percent_x(percent_x)
        kpd = self.type_spch.calc_k_kpd(koef_raskh)
        koef_nap = self.type_spch.calc_k_nap(koef_raskh)
        q_in = mode.q_in[self.idx] / self.w_cnt_current
        mas_rate =  q_in * self.lim.plot_std * (10**6) / 24 / 60 / 60
        def comp_by_freq(freq):
            p_in = self.get_p_in_by_freq_and_perc_on_mode(mode, freq, percent_x)
            comp_degree = mode.p_input / p_in
            return comp_degree, p_in
        def func(freq):
            u_val = self.type_spch.vel(freq) / 60
            cur_dh = koef_nap * (u_val**2)
            comp_by_freq_val, p_in =  comp_by_freq(freq)
            z_in = my_z(p_in, mode.t_in)
            m_t =  (self.lim.k_val - 1) / (self.lim.k_val * kpd)
            cur_comp_degree = (cur_dh * m_t / (z_in * self.lim.r_val * mode.t_in) + 1) ** (1 / m_t)
            return cur_comp_degree - comp_by_freq_val

        x0_val = self.type_spch.fnom*.5
        sol = root_scalar(f=func, x0=x0_val, x1=x0_val*2, method='secant')
        freq = sol.root
        curr_mode = mode.clone()
        curr_mode.p_input = self.get_p_in_by_freq_and_perc_on_mode(mode, freq, percent_x)
        summ = self.calc_stage_summary_in(curr_mode, freq)
        return summ
    def get_perc_by_freq_dim(self, mode:Mode, freq_dim:float)->Union[None,StageSummary]:
        return self.calc_stage_summary_out(mode, freq_dim * self.type_spch.fnom, is_in_border=True)
    def get_mght_by_perc(self, mode:Mode, percent_x:float)->StageSummary:
        koef_raskh = self.type_spch.koef_raskh_by_percent_x(percent_x)
        kpd = self.type_spch.calc_k_kpd(koef_raskh)
        koef_nap = self.type_spch.calc_k_nap(koef_raskh)
        q_in = mode.q_in[self.idx] / self.w_cnt_current
        mas_rate =  q_in * self.lim.plot_std * (10**6) / 24 / 60 / 60
        def mght_by_freq(freq):
            p_in = self.get_p_in_by_freq_and_perc_on_mode(mode, freq, percent_x)
            comp_degree = mode.p_input / p_in
            z_in = my_z(p_in, mode.t_in)
            cur_dh = dh(comp_degree, z_in, mode.t_in, self.lim.r_val, self.lim.k_val, kpd)
            mght = cur_dh * mas_rate / kpd / (10**3)
            return mght
        def func(freq):
            u_val = self.type_spch.vel(freq) / 60
            cur_dh = koef_nap * (u_val**2)
            return cur_dh * mas_rate / kpd / (10**3) - mght_by_freq(freq)

        x0_val = self.type_spch.fnom*.5
        sol = root_scalar(f=func, x0=x0_val, x1=x0_val*2, method='secant')
        freq = sol.root
        curr_mode = mode.clone()
        curr_mode.p_input = self.get_p_in_by_freq_and_perc_on_mode(mode, freq, percent_x)
        summ = self.calc_stage_summary_in(curr_mode, freq)
        return summ
    def get_perc_by_mght(self, mode:Mode, mght:float)->StageSummary:
        q_in = mode.q_in[self.idx] / self.w_cnt_current
        mas_rate =  q_in * self.lim.plot_std * (10**6) / 24 / 60 / 60
        def fun_p_in(kpd, z_in):
        # mght = cur_dh * mas_rate / kpd / (10**3)
            cur_dh = mght / mas_rate * kpd * (10**3)
            m_t =  (self.lim.k_val - 1) / (self.lim.k_val * kpd)
            comp_degree = (cur_dh * m_t / (z_in * self.lim.r_val * mode.t_in) + 1) ** (1 / m_t)
            return mode.p_input / comp_degree
        def func(x):
            freq, kpd, p_in = x
            z_in = my_z(p_in, mode.t_in)
            cur_p_in = fun_p_in(kpd, z_in)
            koef_raskh = self.type_spch.koef_raskh(q_in, cur_p_in, freq, mode.t_in, self.lim.r_val, self.lim.plot_std)
            cur_kpd = self.type_spch.calc_k_kpd(koef_raskh)
            koef_nap = self.type_spch.calc_k_nap(koef_raskh)
            u2_vel = (self.type_spch.vel(freq)/60)**2
            return [
                koef_nap * u2_vel - mght / mas_rate * kpd * (10**3),
                cur_kpd - kpd,
                cur_p_in - p_in
            ]
        x0_val = [self.type_spch.fnom, .8, self.type_spch.p_in]
        sol = root(fun=func, x0=x0_val)
        freq, kpd, p_in = sol.x
        curr_mode = mode.clone()
        curr_mode.p_input = fun_p_in(kpd, my_z(p_in, mode.t_in))
        summ = self.calc_stage_summary_in(curr_mode, freq)
        return summ
    def get_perc_by_comp(self, mode:Mode, comp_degree:float)->StageSummary:
        p_in = mode.p_input / comp_degree
        q_in = mode.q_in[self.idx] / self.w_cnt_current
        z_in = my_z(p_in, mode.t_in)
        def func(freq):
            koef_raskh = self.type_spch.koef_raskh(q_in, p_in, freq, mode.t_in, self.lim.r_val, self.lim.plot_std)
            koef_nap = self.type_spch.calc_k_nap(koef_raskh)
            kpd = self.type_spch.calc_k_kpd(koef_raskh)
            u2_val = (self.type_spch.vel(freq)/ 60) ** 2 
            cur_dh = koef_nap * u2_val
            m_t =  (self.lim.k_val - 1) / (self.lim.k_val * kpd)
            cur_comp_degree = (cur_dh * m_t / (z_in * self.lim.r_val * mode.t_in) + 1) ** (1 / m_t)
            return comp_degree - cur_comp_degree
        x0_val = self.type_spch.fnom
        sol = root_scalar(f=func, x0=x0_val, x1=x0_val*1.01)
        curr_mode = mode.clone()
        curr_mode.p_input = p_in
        summ = self.calc_stage_summary_in(curr_mode, sol.root)
        return summ
    def show_plt(self, t_in:float, f_max:float, f_min:float, summ:Iterable[StageSummary]=None, isLine:bool=False, z_in=None):
        list_all_curve = get_gdh_curvs(
            sp=self.type_spch, temper=t_in, k=self.lim.k_val, R=self.lim.r_val,
            plot_st=self.lim.plot_std, freqs=np.linspace(f_max,f_min,9), z_in=z_in
        )['gdh']['datasets']
        list_freq_curve = list(filter(lambda dic: dic['my_type']=='freq', list_all_curve))
        curves = list(map(lambda dic: dic['data'], list_freq_curve))
        for curv in curves:
            x_val,y_val, label = list(zip(*[
                [dic['x'], dic['y'], dic['label']]
            for dic in curv]))
            plt.plot(x_val,y_val)
            plt.annotate(label[0], (x_val[0],y_val[0]))
        if summ:
            if isLine:
                x_arr,y_arr = zip(*[
                    [cur_summ.comp_degree, cur_summ.volume_rate]
                for cur_summ in summ])
                plt.plot(x_arr, y_arr)
            else:
                for cur_summ in summ:
                    plt.scatter(cur_summ.volume_rate, cur_summ.comp_degree)
        plt.show()
    @property
    def prime(self):
        return self.type_spch.name + str(self.idx)
    @property
    def second(self):
        return self.w_cnt_current
    def get_volime_rate(self, mode:Mode, p_in:float)->float:
        return ob_raskh(mode.q_in[self.idx] / self.w_cnt_current, p_in, mode.t_in, self.lim.r_val, plot_std=self.lim.plot_std)
    def get_p_in_by_freq_and_perc_on_mode(self, mode:Mode, freq:float, percent_x:float)->float:
        volume_rate = self.type_spch.get_volume_rate_by_freq_and_perc(freq, percent_x)
        plot = mode.q_in[self.idx] *  (10**6) / 24 / 60 * self.lim.plot_std  / volume_rate
        p_z_val = p_z(plot, self.lim.r_val, mode.t_in)
        x0_val = p_z_val * .9
        sol = root_scalar(f=lambda p_in: p_in / my_z(p_in, mode.t_in) - p_z_val, x0=x0_val, x1=x0_val * 1.001)
        return sol.root
