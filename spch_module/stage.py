"""[summary]
"""
from dataclasses import dataclass
from typing import Iterable, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root, root_scalar

from .formulas import my_z, ob_raskh, plot
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
            type_spch=self.type_spch,
            mght=mght,
            freq=freq,
            kpd=kpd_pol,
            p_in=p_in,
            comp_degree=comp_degree,
            w_cnt=self.w_cnt_current,
            p_out_req=p_out,
            volume_rate=volume_rate,
            percent_x=percent_x,
            t_out=t_out
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
                q_in=q_one,
                p_in=mode.p_input,
                freq=freq,
                t_in=mode.t_in,
                r_val=self.lim.r_val,
                plot_std=self.lim.plot_std)
            z_in = my_z(mode.p_input, mode.t_in)
            volume_rate, comp_degree = self.type_spch.calc_xy(
                freq=freq,
                k_raskh=k_raskh,
                z_val=z_in,
                r_val=self.lim.r_val,
                t_in=mode.t_in,
                k_val=self.lim.k_val)
            return self.get_stage_summ(
                volume_rate=volume_rate,
                comp_degree=comp_degree,
                t_in=mode.t_in,
                freq=freq,
                p_out=comp_degree * mode.p_input
            )
    def calc_stage_summary_out(self, mode:Mode, freq:float, method:str='secant')->StageSummary:
        cur_mode = mode.clone()
        cur_mode.t_in = self.lim.t_avo if self.idx >= 1 else mode.t_in
        def func(p_in):
            cur_mode.p_input = p_in
            return self.calc_stage_summary_in(cur_mode, freq).p_out_req - mode.p_input
        p_in_0 = self.type_spch.ptitle / self.type_spch.stepen / 10
        sol = root_scalar(
            f=func,
            method=method,
            x0=p_in_0,
            x1=p_in_0 * 1.0001,
        )
        cur_mode.p_input = sol.root
        return self.calc_stage_summary_in(cur_mode, freq)


    def get_freq_min_max(self, mode:Mode)->Tuple[float,float]:
        volume_rate = ob_raskh(mode.q_in[self.idx], mode.p_input, mode.t_in, self.lim.r_val, self.lim.plot_std) / self.w_cnt_current
        return self.type_spch.get_freq_bound(volume_rate)
    def get_freq_min_max_out(self, mode:Mode, border:Border)->Tuple[StageSummary,StageSummary]:
        assert any(list(map(lambda x: x.key == 'mght', border))), 'В граничных условиях нет мощности'
        cur_mode = mode.clone()
        cur_mode.t_in = self.lim.t_avo if self.idx >= 1 else mode.t_in
        weigth_percent = border['percent_x']
        def func(x, is_max, key):
            p_in, freq = x
            cur_mode.p_input = p_in
            summ = self.calc_stage_summary_in(cur_mode, freq)
            percent_val = weigth_percent.get_obj_val_no_abs(summ)
            return_val = summ[key] - (border[key].max_val if is_max else border[key].min_val)
            return_press = summ.p_out_req - mode.p_input
            return [ return_val, return_press]

        p_in_0 = self.type_spch.ptitle / self.type_spch.stepen / 10
        freq_nom = self.type_spch.fnom
        x0_val = [p_in_0, freq_nom]
        
        all_sols = [
            [root(
                fun=func,
                x0=x0_val,
                args=(bound, weigth.key),
                # method='broyden1'
            )
            for weigth in filter(lambda x: x.key != 'percent_x', border)]
            # for weigth in filter(lambda x: True, border)]
        for bound in [False, True]]
        print(all_sols)
        sols_min, sols_max = [
            self.calc_stage_summary_out(mode, fun(sol_arr, key=lambda x: x.x[1]).x[1])
        for sol_arr, fun in zip(all_sols, [max, min])] 

        min_percent_x, max_percent_x  = [
            self.calc_stage_summary_out(mode, root(
                fun=func,
                x0=x0_val,
                args=(bound, 'percent_x')
            ).x[1])
        for bound in [False, True]]
        # return min_percent_x, max_percent_x
        return sols_min, sols_max
        if border['percent_x'].max_val <= sols_max.percent_x <= border['percent_x'].min_val:
            if border['percent_x'].max_val <= sols_min.percent_x <= border['percent_x'].min_val:
                return sols_min, sols_max
            else:
                return min_percent_x, sols_max
        else:
            if border['percent_x'].max_val <= sols_min.percent_x <= border['percent_x'].min_val:
                return sols_min, max_percent_x
            else:
                return min_percent_x, max_percent_x



    
    def show_plt(self, t_in:float, f_max:float, f_min:float, summ:Iterable[StageSummary]=None, isLine:bool=False, z_in=None):
        list_all_curve = get_gdh_curvs(
            sp=self.type_spch,
            temper=t_in,
            k=self.lim.k_val,
            R=self.lim.r_val,
            plot_st=self.lim.plot_std,
            freqs=np.linspace(f_max,f_min,9),
            z_in=z_in)['gdh']['datasets']
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
