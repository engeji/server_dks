import math
from itertools import chain

import numpy as np

from .formulas import my_z
from .spch import Spch

N_KPD = 1000
N_MGHT = 5

def get_gdh_curvs(
    sp:Spch, temper=None, n=10, n_mgh=50, k=1.31, R=None,
    freqs=np.linspace(1.1, 0.7, 9), plot_st=0.698,
    z_in=None):
    t_current =  sp.t_val if temper is None else temper
    R_curr = sp.r_val if R is None else R
    z_avg = my_z( sp.ptitle / sp.stepen / 10.0, t_current) if z_in is None else z_in
    all_k_raskh = np.linspace(sp.min_k_raskh, sp.max_k_raskh, n)
    sp.calc_xy
    res_freq = []
    for f in freqs:
        freq = int(sp.fnom * f)
        res_freq.append([])
        for ind_k, k_raskh in enumerate(all_k_raskh):
            point_x, point_y = sp.calc_xy(freq, k_raskh, z_avg, R_curr, t_current, k)
            res_freq[-1].append({
                'x':point_x,
                'y':point_y,
                'label': ("n/nном {0:.2f}".format(f) if f == 1 else "{0:.2f}".format(f)) if ind_k == 0 else "",
                'type':'freq'
            })    
    all_kpd = {
        ind_k:{
            'kpd': sp.calc_k_kpd(k_raskh) * 100,
            'round_kpd':math.floor(sp.calc_k_kpd(k_raskh) * 100),
            'k_raskh': k_raskh,
            'div': (sp.calc_k_kpd(k_raskh) * 100) % 1,         
        }
    for ind_k, k_raskh in enumerate(np.linspace(sp.min_k_raskh, sp.max_k_raskh, N_KPD))}

    round_kpd_set = set(map(lambda x: x['round_kpd'], all_kpd.values()))

    kpd_for_curve = list(map(lambda round_kpd:
        next(filter(lambda key: all_kpd[key]['round_kpd'] == round_kpd, all_kpd.keys())),
        round_kpd_set
    ))
    max_kpd = max(map(lambda p: p['kpd'], all_kpd.values()))
    indecies_for_curve = set((0,*kpd_for_curve, N_KPD-1, next(filter(lambda key: all_kpd[key]['kpd'] == max_kpd, all_kpd.keys()))))
    for ind in indecies_for_curve:      
        res_freq.append([])  
        for ind_f, f in enumerate(freqs):
            freq = int(sp.fnom * f)
            point_x, point_y = sp.calc_xy(freq, all_kpd[ind]['k_raskh'], z_avg, R_curr, t_current, k)
            res_freq[-1].append({
                'x':point_x,
                'y':point_y,
                'label':f"{all_kpd[ind]['round_kpd'] / 100:.2f}" if ind_f == 0 else "",
                'type':'kpd'
            })
    max_mght, min_mght = (
        (f * math.pi * sp.d_val / 60) ** 3 * math.pi * sp.d_val * k_raskh  * nap * plot_st / ( 4 * kpd) / 1000
    for f, k_raskh, nap, kpd in (
        (
            max(freqs) * sp.fnom,
            sp.max_k_raskh,
            sp.calc_k_nap(sp.max_k_raskh),
            sp.calc_k_kpd(sp.max_k_raskh),
        ),
        (
            min(freqs) * sp.fnom,
            sp.min_k_raskh,
            sp.calc_k_nap(sp.min_k_raskh),
            sp.calc_k_kpd(sp.min_k_raskh),
        ))
    )

    for mght in np.linspace(min_mght, max_mght, N_MGHT):
        temp = []
        for k_raskh in np.linspace(sp.min_k_raskh, sp.max_k_raskh, n_mgh):
            kpd = sp.calc_k_kpd(k_raskh)
            k_nap = sp.calc_k_nap(k_raskh)
            u3 = 4 * kpd / (plot_st * k_nap * k_raskh * math.pi * (sp.d_val ** 2) * mght * 1000)
            u = u3 ** (1 / 3)
            freq = u * 60 / (math.pi * sp.d_val / sp.fnom)
            point_x, point_y = sp.calc_xy(freq, k_raskh, z_avg, R_curr, t_current, k)
            if max(freqs) > freq / sp.fnom > min(freqs):
                temp.append({
                    'x': point_x,
                    'y': point_y,
                    'label': f"{mght:.0f}" if k_raskh == sp.max_k_raskh else "",
                    'type': 'mght'
                })
        if len(temp) > 0:
            res_freq.append(temp)
    

    no_dim = sp.get_no_dim_fact_points()
    return {
        'gdh': {
            'datasets': [{
                'data': points,
                'my_type': points[0]['type']
            }for points in res_freq]},
        'z_avg': f'{z_avg:0.2f}',
        **no_dim,
    }
