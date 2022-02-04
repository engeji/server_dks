import math
from itertools import chain
from typing import Iterable, List, Tuple
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import (Bounds, NonlinearConstraint, OptimizeResult,
                            minimize, show_options)

from .mode import Mode
from .comp import Comp
from .summary import CompSummary, StageSummary
from .weight import DEFAULT_BORDER, Border


def is_nan(wrapped_func):
    def wraper(*arg):
        res = wrapped_func(*arg)
        return 0 if np.isnan(res) else res
    return wraper

COUNTER = 0
def get_log(wrapped_func):
    def wraper(*arg):
        res = wrapped_func(*arg)
        COUNTER += 1
        return res
    return wraper

class Solver:
    def __init__(self,comp:Comp,mode:Mode,p_req:float=None, sens:float=.01, list_border:List[Border]=None):
        self.comp:Comp = comp
        self.mode:Mode = mode
        self.p_req = p_req
        self.vec_ob_func = np.vectorize(lambda x1, y1: self.ob_func([x1,y1]))
        self.vec_border_func = np.vectorize(lambda x1, y1: self.get_border_val([x1,y1]))
        self.vec_p_out_req_func = np.vectorize(lambda x1, y1: self.get_summ([x1,y1])[-1].p_out_req)
        self.vec_all_mght = np.vectorize(lambda x1, y1: self.get_all_mght([x1,y1]))
        self.sens_press = sens
        self.border_list = [DEFAULT_BORDER] * len(self.comp) if list_border is None else list_border
        self.init_plot()
    def init_plot(self):
        freq_min_1st, freq_max_1st = self.get_bounds[0]
        freq_x_for_plot = np.linspace(freq_min_1st, freq_max_1st, 10)

        freq_y_for_plot = [
            self.func_constr([freq,0])[1][0]
        for freq in freq_x_for_plot]

        freq_y_for_plot2 = [
            self.func_constr([freq,0])[1][1]
        for freq in freq_x_for_plot]

        self.X, self.Y = np.meshgrid(
            np.arange(
                min(freq_x_for_plot) * .9,
                max(freq_x_for_plot) * 1.1,
                200
            ),
            np.arange(
                min(freq_y_for_plot) * .9,
                max(freq_y_for_plot2) * 1.1,
                200
            )
        )
        self.plot_ob = PlotObFunc()

        self.plot_ob.add_line([freq_x_for_plot, freq_y_for_plot])
        self.plot_ob.add_line([freq_x_for_plot, freq_y_for_plot2])
        self.plot_ob.add_line([[freq_x_for_plot[0]]*2, [freq_y_for_plot[0], freq_y_for_plot2[0]]])
        self.plot_ob.add_line([[freq_x_for_plot[-1]]*2, [freq_y_for_plot[-1], freq_y_for_plot2[-1]]])

        for title, arr_curve  in self.get_curves():
            self.plot_ob.add_curve(title, arr_curve)

    def get_summ(self, x)-> CompSummary:
        return self.comp.calc_comp_summary(self.mode, x, self.border_list)
    @is_nan
    def get_all_mght(self, x)->float:
        return self.get_summ(x).all_second
    @is_nan
    def get_border_val(self, x, comp_summ:CompSummary=None)->float:
        curr_comp_summ = self.get_summ(x) if comp_summ is None else comp_summ
        return curr_comp_summ.all_prime
    def ob_func_by_components(self,x):
        comp_summ = self.get_summ(x)
        borders_sum = self.get_border_val(x, comp_summ)
        dp_val = abs(comp_summ[-1].p_out_req - self.p_req)/self.sens_press
        return [
            dp_val**2,
            borders_sum,
            self.get_all_mght(x),
        ]
    @is_nan
    def ob_func(self, x):
        return sum(self.ob_func_by_components(x))
    def func_constr(self, x:List[float])->List[List[float]]:
        assert len(x) == len(self.comp), 'Вектор не соотв. компановке'
        return self.comp.get_freq_bound_min_max(self.mode, x)
    def get_curves(self)->List[tuple]:
            return [
                (
                    'Целевая функция',
                    (self.X, self.Y, self.vec_ob_func(self.X,self.Y)),
                ),
                (
                    'Граничные условия',
                    (self.X, self.Y, self.vec_border_func(self.X,self.Y)),
                    
                ),
                (
                    'Давление выхода 2 ступени',
                    (self.X, self.Y, self.vec_p_out_req_func(self.X,self.Y)),
                ),
                (
                    'Суммарная мощность',
                    (self.X, self.Y, self.vec_all_mght(self.X,self.Y)),
                ),
        ]
    def show_plt(self):
        return self.plot_ob.show_plt()
    def show_plt_old(self):
        fig, ax = plt.subplots(2,2)
        freq_min_1st, freq_max_1st = self.get_bounds[0]
        freq_x_for_plot = np.linspace(freq_min_1st, freq_max_1st, 10)
        freq_y_for_plot = [
            self.func_constr([freq,0])[1][0]
        for freq in freq_x_for_plot]
        freq_y_for_plot2 = [
            self.func_constr([freq,0])[1][1]
        for freq in reversed(freq_x_for_plot)]
        X, Y = np.meshgrid(
            np.arange(
                min(freq_x_for_plot) * .9,
                max(freq_x_for_plot) * 1.1,
                200
            ),
                np.arange(
                min(freq_y_for_plot) * .9,
                max(freq_y_for_plot2) * 1.1,
                200
            )
        )


        degries = [1, 10,100,1000]
        cnt_deg = [5,5,1,5]
        lev = list(chain.from_iterable([
            np.linspace(1*deg, 9*deg, cnt)
        for cnt, deg in zip(cnt_deg,degries)]))
        point_min, comp_sum = self.optimize()
        plt.figtext(0.2, 0.02, repr(comp_sum)+f'func={point_min.fun}')
        curves = [
            self.vec_ob_func(X,Y),
            self.vec_border_func(X,Y),
            self.vec_p_out_req_func(X,Y),
            self.vec_all_mght(X,Y)
        ]
        # return curves[0]
        for idx, (_, ax_ij) in enumerate(np.ndenumerate(ax)):
            ax_ij.plot(
                [*freq_x_for_plot, *list(reversed(freq_x_for_plot)), freq_x_for_plot[0]],
                [*freq_y_for_plot, *freq_y_for_plot2, freq_y_for_plot[0]]
            )
            CS = ax_ij.contour(X, Y, curves[idx], lev)
            ax_ij.scatter(*point_min.x)
            ax_ij.clabel(CS)
            ax_ij.title.set_text({
                0:'Целевая функция',
                1:'Граничные условия',
                2:'Давление выхода 2 ступени',
                3:'Суммарная мощность',
            }[idx])
        plt.show()
        # return show_options('minimize','TNC',False)
    @property
    def get_bounds(self)->List[List[float]]:
        return [
            self.comp[0].get_freq_min_max(self.mode),
            # [self.comp[0].get_freq_min_max(self.mode)[0], 3000],
            *[
                [-np.inf, np.inf]
            for idx_stage in range(len(self.comp)-1)]
        ]
    @property
    def get_constr(self)->List[NonlinearConstraint]:
        return [
            NonlinearConstraint(
                lambda x: (
                    self.func_constr(x)[idx][1] - x[idx]) / (self.func_constr(x)[idx][1] - self.func_constr(x)[idx][0]),
                0,
                1
            )
        for idx in range(len(self.comp))]
    @property
    def get_x0(self)->List[float]:
        res = [sum(self.comp[0].get_freq_min_max(self.mode))/2]
        for idx, stage in list(enumerate(self.comp))[1:]:
            res.append(
                sum(self.comp.get_freq_bound_min_max(
                    self.mode,
                    list(map(lambda stage2: res[-1] if stage2 == stage else stage2.type_spch.fnom, self.comp))
                )[idx])/2)
        return res
    def optimize(self, is_chage_w_cnt:bool=True,get_all:bool=False, border_list:List[Border]=None)->Tuple[OptimizeResult, CompSummary]:
        assert len(self.border_list) == len(self.comp), 'Количество ступеней не равно количеству ограничений ступеней'
        w_cnt_arr = [
            range(1,stage.w_cnt+1)
        for stage in self.comp]

        comp_res_arr, res_arr = [], []
        permisions = product(*w_cnt_arr) if is_chage_w_cnt else [[
                stage.w_cnt
            for stage in self.comp]]

        for w_cnt in permisions:
            self.comp.w_cnt_calc = w_cnt
            res = minimize(
                self.ob_func,
                self.get_x0,
                method= 'trust-constr',#'SLSQP'  'trust-constr' 'TNC'
                constraints=self.get_constr,
                bounds=self.get_bounds,
                # hess=lambda x: np.zeros((len(self.comp),len(self.comp)))
            )
            comp_summ = self.get_summ(res.x)
            comp_res_arr.append(comp_summ)
            res_arr.append(res)
        if get_all:
            return comp_res_arr
        minimum = min(comp_res_arr)
        print(f'COUNTER is {COUNTER}')
        return (res_arr[comp_res_arr.index(minimum)], minimum)

class PlotObFunc:
    def __init__(self):
        self.dic_curve = {}
        self.list_line = []
        self.levels = [
            (1, 50),
            (10, 90),
            (100, 5),
            (1000, 5)
        ]
    def add_curve(self, title:str, curve:np.ndarray):
        self.dic_curve[title] = curve
    def add_line(self, line:List[List[float]]):
        self.list_line.append(line)
    def show_plt(self, point:Tuple[float, float]=None):
        cnt_curve_on_flat = math.ceil(math.sqrt(len(self.dic_curve)))
        fig, ax = plt.subplots(cnt_curve_on_flat ,cnt_curve_on_flat)
        lev = list(chain.from_iterable([
            np.linspace(1*deg, 9*deg, cnt)
        for deg, cnt in self.levels]))
        
        for curve_title, (_, ax_ij) in zip(self.dic_curve, np.ndenumerate(ax)):
            for line in self.list_line:
                ax_ij.plot(*line)
            CS_value = ax_ij.contour(*self.dic_curve[curve_title], lev)
            if point:
                ax_ij.scatter(point)
            ax_ij.clabel(CS_value)
            ax_ij.title.set_text(curve_title)
        plt.show()
        return self.dic_curve