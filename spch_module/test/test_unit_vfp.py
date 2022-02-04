from ..facilities import get_comp_by_name
from ..mode import Mode
from ..solver import Solver
from ..solver_vfp import Solver_vfp
from ..weight import DEFAULT_BORDER, Border, Weight
from ..facilities import get_summ_collections
import numpy as np
def test_stage_calc():
    mode = Mode(283, [20, 20.55], 8)
    comp = get_comp_by_name([
            'ГПА-ц3-16С-45-1.7(ККМ)',
            'ГПА Ц3 16с76-1.7М'
        ],[1,1])
    stage = comp[-1]
    summ = stage.calc_stage_summary_out(mode, 3640)
    print(summ)
    print(summ['freq_dim'])
    

def test_comp_calc():
    pass
def test_freq_stage_min_max_bound():
    mode = Mode(293, [20,20.55], 8.)
    comp = get_comp_by_name(
        [
            'ГПА-ц3-16С-45-1.7(ККМ)',
            'ГПА Ц3 16с76-1.7М'],
        [1,1])
    stage1 = comp[-1]
    border = Border([
        Weight(
            max_val=16000,
            min_val=-.1,
            weight=100,
            key='mght',
            sens=.1
        ),
        Weight(
            max_val=1.1,
            min_val=.1,
            weight=100,
            key='freq_dim',
            sens=.1
        ),
        Weight(
            max_val=1.9,
            min_val=1.,
            weight=100,
            key='comp_degree',
            sens=.1
        ),
    ])
    
    res = stage1.get_freq_min_max_out(mode=mode, border=border)
    sols = [
            get_summ_collections([mode], [sol])
        for sol in res]
    # for sol_arr in zip(*res)]
    print(*sols, sep='\n###############\n')
    # summs1 = [
    #     stage1.calc_stage_summary_in(Mode(stage1.lim.t_avo, mode.q_in, x.x[0]), x.x[1])
    # for x in sols_min]

    # print(get_summ_collections([mode]*len(summs1), summs1))
    # print("#############################")
    # summs1 = [
    #     stage1.calc_stage_summary_in(Mode(stage1.lim.t_avo, mode.q_in, x.x[0]), x.x[1])
    # for x in sols_max]

    # print(get_summ_collections([mode]*len(summs1), summs1))
    # assert all([
    #     abs(summ.p_out_req - mode.p_input) < .1
    # for summ in summs])
    # assert abs(summs[0]['mght'] - border['mght'].min_val) < .1
    # assert abs(summs[1]['mght'] - border['mght'].max_val) < .1
    
def test_freq_comp_bounds():
    comp = get_comp_by_name(
        [
            'ГПА-ц3-16С-45-1.7(ККМ)',
            'ГПА Ц3 16с76-1.7М'],
        [1,1])
    border = Border(
        DEFAULT_BORDER['mght'],
    )
    mode = Mode(283, [20,20], 8.)
    stage_bound = list(zip(*comp[-1].get_freq_min_max_out(mode, border)))[-1]
    print(f'last stage bound is {stage_bound}')
    min_bound = comp.get_freq_bound_min_max_out(mode, [border]*2, [None, stage_bound[0]])
    max_bound = comp.get_freq_bound_min_max_out(mode, [border]*2, [None, stage_bound[1]])

    all_freq_min = list(zip(*min_bound))[0]
    all_freq_max = list(zip(*max_bound))[1]

    summ_min = comp.calc_comp_summary_out(mode, all_freq_min, [border]*2)
    summ_max = comp.calc_comp_summary_out(mode, all_freq_max, [border]*2)

    print(summ_min)
    print(summ_max)

    assert all([
        abs(summ.mght - border['mght'].max_val) <.1
    for summ in summ_max])

    assert all([
        abs(summ.mght - border['mght'].min_val) <.1
    for summ in summ_min])

def test_show_plot():
    from ..limit import DEFAULT_LIMIT
    mode = Mode(DEFAULT_LIMIT.t_avo, [20, 20.55], 8)
    comp = get_comp_by_name([
            'ГПА-ц3-16С-45-1.7(ККМ)',
            'ГПА Ц3 16с76-1.7М'
        ],[1,1])
    border = Border([
            Weight(16000, 7000, 100, 'mght', .1),
            Weight(1.1, .7, 100, 'freq_dim', .1),
    ])
    # solver = Solver_vfp(comp, mode, [border]*2)
    from itertools import chain
    summs = list(chain(*[
        stage.get_freq_min_max_out(mode, border)
    for stage in comp]))


    print(get_summ_collections([mode]*4, summs))
    print('########################################')
    summs2 = [
            comp[0].calc_stage_summary_out(mode, freq)
        for freq in np.linspace(3710,5830, 9)]
    print(get_summ_collections(
        modes=[mode]*10,
        sums=summs2
    ))
    from ..formulas import q_in, my_z
    z_avg = sum([
        my_z(summ.p_in, mode.t_in)
    for summ in summs]) / len(summs)
    # comp[0].show_plt(mode.t_in, 1.1, .7, summs2,z_in=z_avg)
    print(*[
        q_in(summ.volume_rate, summ.p_in, mode.t_in, comp[0].lim.r_val)
    for summ in summs2], sep='\n')
    # solver.show_plt()
