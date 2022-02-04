import numpy as np

from .. import ALL_SPCH_LIST
from ..facilities import get_comp_by_name, get_spch_by_name
from ..mode import Mode
from ..summary import CompSummaryCollection
from ..solver import Solver
from ..weight import DEFAULT_BORDER
from .conftest import list_comp, mode_coll
from ..header import BaseStruct, MyList

def test_show_plt(list_comp, mode_coll, capsys):
    # idx = 14
    # comp = list_comp[idx]
    comp = get_comp_by_name([
        'ГПА-ц3-16С-45-1.7(ККМ)',
        'ГПА Ц3 16с76-1.7М'
    ],[1,2])
    mode = Mode(283, 27, 2.65)
    solver = Solver(comp,mode, 8.)
    print(f'show is {solver.show_plt()}')
    # print(solver.optimize())

def test_border(list_comp, mode_coll, capsys):
    idx = 10
    comp = list_comp[idx]
    mode = mode_coll[idx]
    x = [3042, 4869]
    summ = comp.calc_comp_summary(mode, x)
    with capsys.disabled():
        for stage_summ in summ:
            print(f'{stage_summ}')
            for weight in DEFAULT_BORDER:
                key = weight.key
                value = stage_summ[weight.key]
                res = weight.get_obj_val(stage_summ)
                print(f"   key {key} value {value:.2f} res {res:.2f}")
            print(f"border summ {DEFAULT_BORDER.get_obj_val(stage_summ)}")
    
def test_ALL_SPCH_LIST(capsys):
    with capsys.disabled():
        print(*[ sp.name for sp in ALL_SPCH_LIST], sep='\n')
def test_comp_summary_collection(capsys):
    comp = get_comp_by_name([
        'ГПА-ц3-16С-45-1.7(ККМ)',
        'ГПА Ц3 16с76-1.7М'
    ],[2,2])
    modes = [
        Mode(283, [q, q+1], 2.65)
    for q in range(15,18)]
    p_req = 8
    summaries = [
        Solver(comp, mode,p_req).optimize()[1]
    for mode in modes]
    comp_coll = CompSummaryCollection(summaries)
    with capsys.disabled():
        print(comp_coll)

