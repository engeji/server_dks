from .. __init__ import ALL_SPCH_LIST
from .. limit import DEFAULT_LIMIT
from ..comp import Stage
from ..mode import Mode
import numpy as np
def test_dimens():
    for sp in ALL_SPCH_LIST[:1]:
        k_raskh_avg = (sp.max_k_raskh - sp.min_k_raskh) / 2.
        x, y = sp.calc_xy(sp.fnom, k_raskh_avg, .9, sp.r_val, sp.t_val, DEFAULT_LIMIT.k_val)
        volume_rate = sp.volume_rate_by_freq_and_koef_raskh(sp.fnom, k_raskh_avg)
        assert abs(x - volume_rate) < .01
        comp_degree = sp.calc_y(volume_rate, k_raskh_avg, .9, sp.r_val, sp.t_val, DEFAULT_LIMIT.k_val)
        assert abs(comp_degree - y) < .01
        percent_x = sp.percent_x_by_k_raskh(k_raskh_avg)
        freq = sp.freq_by_percent(volume_rate,percent_x)
        assert abs(k_raskh_avg -  sp.koef_raskh_by_percent_x(percent_x)) < .01
        assert abs(freq - sp.fnom) < .01
        stage = Stage(sp,1,DEFAULT_LIMIT,0)
        mode = Mode(285., 20, sp.ptitle)
        print(mode)
        assert False
        stage.calc_stage_summary_out()

        
        
