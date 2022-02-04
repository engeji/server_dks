"""[summary]
"""
import numpy as np

from ..facilities import get_comp_by_name
from ..mode import Mode
from ..solver import Solver
from ..weight import DEFAULT_BORDER, Border, Weight
from ..header import MyList
from .conftest import main_comp as comp
from .conftest import main_mode as mode
from ..summary import CompSummary

def test_min(capsys,comp, mode):
    p_out_req = 8.
    delta_freq = 20.
    solver = Solver(comp, mode, p_out_req)
    res, _ = solver.optimize()
    list_res = {}
    for proc in np.linspace(-.5,.5,17):
        list_border = [
            Border(
                Weight(
                    (res.x[0] + delta_freq) / comp[0].type_spch.fnom  + proc,
                    (res.x[0] - delta_freq) / comp[0].type_spch.fnom  + proc,
                    DEFAULT_BORDER[0].weight,'freq_dim',DEFAULT_BORDER[0].sens
                )
            ),
            Border(
                Weight(
                    2,
                    .5,
                    DEFAULT_BORDER[0].weight,'freq_dim',DEFAULT_BORDER[0].sens
                )
            ),
        ]
        solver = Solver(comp, mode, p_out_req)
        result, comp_summ = solver.optimize(is_chage_w_cnt=False,border_list=list_border)
        result_freq = ' '.join([
                f'{stage_summ.freq:.0f}'
        for stage_summ in comp_summ])
        print(''.join([
            f' fun is {result.fun:.4e}',
            f' proc is {proc:.3f}',
            f' res_freq is {result_freq}',
            f' summ_mght is {comp_summ.all_second:.0f}',
            f' p_out is {comp_summ[-1].p_out_req:.3f}'
        ]))
        list_res[result.fun] = proc
    assert list_res[min(*[
        key
    for key in list_res])] == 0

def test_no_min(comp, mode):
    mode.q_in = [30]

    p_out_req = 8.
    solv =Solver(comp, mode, p_out_req)
    res,comp_summ  = solv.optimize()
    print(res)
    print(solv.ob_func_by_components(res.x))
    print(comp_summ)
    print([f'{bord.get_obj_val(stage_summ)}\n' for bord, stage_summ in zip(solv.border_list,comp_summ)])
    assert abs(comp_summ[-1].p_out_req - p_out_req) < solv.sens_press

def test_2_step():
    mode = Mode(283, 15, 2.65)
    comp = get_comp_by_name([
        'ГПА-ц3-16С-45-1.7(ККМ)',
        'ГПА Ц3 16с76-1.7М'
    ],[1,1])
    p_req = 8
    solv1 = Solver(comp, mode, p_req)
    mode2 = Mode(mode.t_in, [mode.q_in[0], mode.q_in[0]/.75], mode.p_input)
    (res1,comp_summ), (res2,comp_summ2) = (
        Solver(comp, mode, p_req).optimize(), Solver(comp, mode2, p_req).optimize()
    )
    print(comp_summ, mode, comp_summ2, mode2)
    print(type(mode.q_in), type(mode2.q_in))
    assert comp_summ[0].mght < comp_summ2[0].mght
def test_w_cnt_calc():
    mode =  Mode(283, 27, 2.65)
    p_req = 8
    comp = get_comp_by_name([
            'ГПА-ц3-16С-45-1.7(ККМ)',
            'ГПА Ц3 16с76-1.7М'
        ],[3,3])
    solver = Solver(comp, mode, p_req, sens = .01)
    res, comp_min = solver.optimize()
    # print(comp_min)
    print(f'min is {comp_min}')
    assert comp_min[0].w_cnt == 1 and comp_min[1].w_cnt == 2
    from itertools import groupby
    groupby()