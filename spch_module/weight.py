"""[summary]
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Union, TYPE_CHECKING

from .header import BaseCollection, BaseStruct
if TYPE_CHECKING:
    from .summary import  StageSummary


LIST_BORDERS = [
    {'key':'freq_dim', 'weight':100, 'max_val':1.1, 'min_val':.7, 'sens': .01},
    {'key':'mght', 'weight':100, 'max_val':16000, 'min_val':7000, 'sens': 100},
]

@dataclass
class Weight(BaseStruct):
    max_val:float
    min_val:float
    weight:float
    key:str
    sens:float

    def treshhold_func(self, bord:float, val:float)->float:
        return abs((bord - val)/self.sens)

    def get_obj_val(self, summ:StageSummary)->float:
        val = summ[self.key]
        if self.min_val > val:  
            return self.treshhold_func(self.min_val, val)
        elif self.max_val < val:
            return self.treshhold_func(self.max_val, val)
        return .0

    def treshhold_func_no_abs(self, bord:float, val:float)->float:
        return (bord - val)/self.sens

    def get_obj_val_no_abs(self, summ:StageSummary)->float:
        val = summ[self.key]
        if self.min_val > val:  
            return self.treshhold_func_no_abs(self.min_val, val)
        elif self.max_val < val:
            return self.treshhold_func_no_abs(self.max_val, val)
        return .0

class Border(BaseCollection[Weight]):
    def __init__(self, list_items:Union[Weight, Iterable[Weight]]):
        def_weight = Weight(
            max_val=100,
            min_val=0,
            weight=100,
            key='percent_x',
            sens=1.
        )
        if isinstance(list_items,Weight):
            super().__init__([def_weight, list_items])
        else:
            super().__init__(list_items+[def_weight])

    def get_obj_val(self, stage_summ: StageSummary)->float:
        return sum([
                weight.get_obj_val(stage_summ) * weight.weight
            for weight in self]) / self.get_sum_weight
    @property
    def get_sum_weight(self)->float:
        return sum([
            weight.weight
        for weight in self])
    def __getitem__(self, index:Union[str,int])->Weight:
        if isinstance(index,str):
            return next(filter(lambda x: x.key == index, self._list_items)) 
        else:
            return self._list_items[index]

DEFAULT_BORDER = Border(
    [
        Weight(**dic_bord)
    for dic_bord in LIST_BORDERS]
)
