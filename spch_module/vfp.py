"""[summary]
"""
# from __future__ import annotations
from typing import Iterable, Union, Dict

from .comp import Comp

class VfpTable:
    res:Dict[float, Comp] = {}
    def __init__(self,q:Union[range, Iterable[float]], p_out:Union[range, Iterable[float]]):
        if isinstance(q, range):
            self.q = list(q)
        else:
            self.q = q

        if isinstance(p_out, range):
            self.p_out = list(p_out)
        else:
            self.p_out = p_out
    def add_comp(self, comp:Comp, p_in_req:float):
        pass
        
