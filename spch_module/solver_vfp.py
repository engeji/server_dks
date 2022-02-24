"""[summary]
"""
from typing import List, Tuple

import numpy as np
from scipy.optimize import NonlinearConstraint, Bounds

from .comp import Comp
from .mode import Mode
from .solver import PlotObFunc, Solver, is_nan
from .summary import CompSummary
from .weight import Border


class Solver_vfp(Solver):
    def __init__(self, comp:Comp, mode:Mode, list_border:List[Border]):

        self.vec_mght_1st = np.vectorize(lambda x1, y1: self.get_mght_1st([x1,y1]))
        self.vec_p_in_2st = np.vectorize(lambda x1, y1: self.get_p_in_2st([x1,y1]))

        self.vec_percent_1st = np.vectorize(lambda x1, y1: self.get_percent_1st([x1,y1]))
        self.vec_percent_2st = np.vectorize(lambda x1, y1: self.get_percent_2st([x1,y1]))
        self.vec_comp_deg_1st = np.vectorize(lambda x1, y1: self.get_comp_deg_1st([x1,y1]))

        super().__init__(comp=comp, mode=mode, list_border=list_border)
        self.vec_ob_func = np.vectorize(lambda x1, y1: self.ob_func([x1,y1]))
        self.vec_p_out_req_func = np.vectorize(lambda x1, y1: self.get_summ([x1,y1])[0].p_in)
    def get_summ(self, x)-> CompSummary:
        return self.comp.calc_comp_summary_out(self.mode, x, self.border_list, is_in_border=False)
    def ob_func_by_components(self,x):
        comp_summ = self.get_summ(x)
        borders_sum = self.get_border_val(x, comp_summ)
        p_in = comp_summ[0].p_in
        return [
            p_in
        ]
    def init_plot(self):
        freq_y_plot = np.linspace(*self.get_bounds[-1], 50)

        freq_x_plot = [
            self.func_constr([0,freq])[0][0]
        for freq in freq_y_plot]

        freq_x_plot2 = [
            self.func_constr([0,freq])[0][1]
        for freq in freq_y_plot]
        
        self.X, self.Y = np.meshgrid(
            np.arange(
                freq_x_plot[0] - 200,
                freq_x_plot2[-1] + 200,
                200
            ),
            np.arange(
                freq_y_plot[0] - 200,
                freq_y_plot[-1] + 200,
                200
            )
        )
        self.plot_ob = PlotObFunc()
        for title, arr_curve  in self.get_curves():
            self.plot_ob.add_curve(title, arr_curve)

        self.plot_ob.add_line([freq_x_plot, freq_y_plot])
        self.plot_ob.add_line([freq_x_plot2, freq_y_plot])
        self.plot_ob.add_line([[freq_x_plot[0],freq_x_plot2[0]], [freq_y_plot[0], freq_y_plot[0]]])
        self.plot_ob.add_line([[freq_x_plot[-1],freq_x_plot2[-1]], [freq_y_plot[-1], freq_y_plot[-1]]])
    @is_nan
    def get_mght_1st(self, x)->float:
        return self.get_summ(x)[0].mght
    @is_nan
    def get_percent_1st(self, x)->float:
        return self.get_summ(x)[0].percent_x
    @is_nan
    def get_percent_2st(self, x)->float:
        return self.get_summ(x)[1].percent_x
    @is_nan
    def get_p_in_2st(self, x)->float:
        return self.get_summ(x)[1].p_in
    @is_nan
    def get_comp_deg_1st(self, x)->float:
        return self.get_summ(x)[0].comp_degree
    def normolize(self, Z_cur):
        map_min = np.min(Z_cur)
        map_max = np.max(Z_cur)
        return Z_cur * 10.
        return (map_min + Z_cur) / (map_max - map_min) * 100.
    def get_curves(self)->List[tuple]:
            return [
                (
                    'Давление входа в 1 ступень',
                    (self.X, self.Y, self.vec_ob_func(self.X,self.Y)),
                ),
                (
                    'Percent 2 ступень',
                    (self.X, self.Y, self.vec_percent_2st(self.X,self.Y)),
                ),
                (
                    'Percent 1 ступень',
                    (self.X, self.Y, self.vec_percent_1st(self.X,self.Y)),
                ),
                (
                    'Степень сжатия 1 ступень',
                    (self.X, self.Y, self.vec_comp_deg_1st(self.X,self.Y) * 10),
                ),
            ]
    @property
    def get_bounds(self)->List[List[float]]:
        summs = self.comp[-1].get_freq_min_max_out(self.mode, self.border_list[-1])
        return [
            *[
                [-np.inf, np.inf]
            for idx_stage in range(len(self.comp)-1)],
            [
                summ.freq
            for summ in summs],
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
    def func_constr(self, x:List[float])->List[Tuple[float, float]]:
        assert len(x) == len(self.comp), 'Вектор не соотв. компановке'
        return [
            [
                summ.freq
            for summ in summ_min_max]
        for summ_min_max in self.comp.get_freq_bound_min_max_out(self.mode, self.border_list, x)]

