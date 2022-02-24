import numpy as np

from ..facilities import get_comp_by_name, get_summ_collections
from ..mode import Mode
from ..solver import Solver
from ..solver_vfp import Solver_vfp
from ..weight import DEFAULT_BORDER, Border, Weight
from ..comp import Comp
import pytest
def test_p_in_calc():
    mode = Mode(283, [20, 20.55], 8)
    comp = get_comp_by_name([
            'ГПА-ц3-16С-45-1.7(ККМ)',
            'ГПА Ц3 16с76-1.7М'
        ],[1,1])
    stage = comp[0]
    curr_mode_0 = mode.clone()
    curr_mode_50 = mode.clone()
    curr_mode_100 = mode.clone()
    fnom = 6000
    curr_mode_0.p_input = stage.get_p_in_by_freq_and_perc_on_mode(mode, fnom, 0)
    curr_mode_50.p_input = stage.get_p_in_by_freq_and_perc_on_mode(mode, fnom, 50)
    curr_mode_100.p_input = stage.get_p_in_by_freq_and_perc_on_mode(mode, fnom, 100)
    summs = [
        stage.calc_stage_summary_in(curr_mode, fnom)
    for curr_mode in (curr_mode_0, curr_mode_50, curr_mode_100)]
    print(get_summ_collections([mode]*3, summs))
    assert abs(summs[0].percent_x - 0) < .01
    assert abs(summs[1].percent_x - 50) < .01
    assert abs(summs[2].percent_x - 100) < .01

def test_stage_calc():
    mode = Mode(283, [20, 20.55], 8)
    comp = get_comp_by_name([
            'ГПА-ц3-16С-45-1.7(ККМ)',
            'ГПА Ц3 16с76-1.7М'
        ],[1,1])
    stage = comp[0]
    summ = stage.calc_stage_summary_out(mode, 8480)
    print(get_summ_collections([mode], [summ]))
    print(summ['freq_dim'])

@pytest.mark.parametrize('func, array, param_name', [
    ('get_perc_by_mght', np.linspace(2000,20000,10), 'mght'),
    ('get_mght_by_perc', np.linspace(2,20,10), 'percent_x'),
    ('get_perc_by_comp', np.linspace(1,2,10), 'comp_degree'),
])
def test_mght_and_perc(func, array, param_name):
    mode = Mode(293, [20,20.55], 8.)
    comp = get_comp_by_name(
        [
            'ГПА-ц3-16С-45-1.7(ККМ)',
            'ГПА Ц3 16с76-1.7М'],
        [1,1])
    stage = comp[0]
    summs = [
        (param, getattr(stage, func)( mode, param))
    for param in array]

    print(get_summ_collections([mode]*len(summs), summs))

    assert all([
        all([
            abs(summ.p_out_req - mode.p_input) < .01,
            abs(summ[param_name] - param) < .01,
        ])
    for param, summ in summs])


def test_comp_calc():
    pass
@pytest.mark.parametrize('mode, comp',[
    (Mode(283, 40, 4.5), get_comp_by_name(['ГПА-ц3-16С-45-1.7(ККМ)'], [1])),
    (Mode(283, 20, 8), get_comp_by_name(['ГПА-ц3-16С-45-1.7(ККМ)'], [1])),
    (Mode(283, 20, 8), get_comp_by_name(['ГПА Ц3 16с76-1.7М'], [1]))
])
def test_freq_stage_min_max_bound(mode:Mode, comp:Comp):

    stage1 = comp[0]
    border = Border([
        Weight(
            max_val=16000,
            min_val=7000,
            weight=100,
            key='mght',
            sens=.1
        ),
        Weight(
            max_val=1.1,
            min_val=.7,
            weight=100,
            key='freq_dim',
            sens=.1
        ),
        Weight(
            max_val=1.9,
            min_val=1.1,
            weight=100,
            key='comp_degree',
            sens=.1
        ),
    ])
    
    res = stage1.get_freq_min_max_out(mode=mode, border=border)
    print(res)
    print(get_summ_collections([mode] * len(res), res))
    assert all([
        abs(summ.p_out_req - mode.p_input) <= .1
    for summ in res if not summ is None])
    
def test_freq_comp_bounds():
    comp = get_comp_by_name(
        [
            'ГПА-ц3-16С-45-1.7(ККМ)',
            'ГПА Ц3 16с76-1.7М'],
        [1,1])
    border = Border([
        Weight(
            max_val=16000,
            min_val=7000,
            weight=100,
            key='mght',
            sens=.1
        ),
        Weight(
            max_val=1.1,
            min_val=.7,
            weight=100,
            key='freq_dim',
            sens=.1
        ),
        Weight(
            max_val=1.9,
            min_val=1.1,
            weight=100,
            key='comp_degree',
            sens=.1
        ),
    ])
    mode = Mode(283, 20, 8.)
    stage_bound = [
        summ.freq
    for summ in comp[-1].get_freq_min_max_out(mode, border)]

    summ_last_stage_max = comp[-1].calc_stage_summary_out(mode, stage_bound[0])
    summ_last_stage_min = comp[-1].calc_stage_summary_out(mode, stage_bound[1])

    print('last stage bound is')
    print(get_summ_collections([mode]*2,
        [
            summ_last_stage_max,
            summ_last_stage_min
        ]))
    min_bound = comp.get_freq_bound_min_max_out(mode, [border]*2, [None, stage_bound[0]])
    max_bound = comp.get_freq_bound_min_max_out(mode, [border]*2, [None, stage_bound[1]])

    all_freq_min = [summ[0].freq for summ in min_bound]
    all_freq_max = [summ[1].freq for summ in max_bound]

    summ_min = comp.calc_comp_summary_out(mode, all_freq_min, [border]*2)
    summ_max = comp.calc_comp_summary_out(mode, all_freq_max, [border]*2)

    print(get_summ_collections([mode]*2,  [summ_max,summ_min]))
    assert abs(summ_max[-1].p_out_req - mode.p_input) <.01
    assert abs(summ_min[-1].p_out_req - mode.p_input) <.01


def test_show_plot():
    from ..limit import DEFAULT_LIMIT
    mode = Mode(283, [20, 20.55], 8)
    comp = get_comp_by_name([
            'ГПА-ц3-16С-45-1.7(ККМ)',
            'ГПА Ц3 16с76-1.7М'
        ],[1,1])
    border = Border([
            Weight(16000, 7000, 100, 'mght', .1),
            Weight(1.1, .7, 100, 'freq_dim', .1),
    ])
    # border['percent_x'].max_val = 11
    # border['percent_x'].min_val = 80
    solver = Solver_vfp(comp, mode, [border]*2)
    solver.show_plt()