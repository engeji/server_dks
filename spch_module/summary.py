"""[summary]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List

from .header import BaseCollection, BaseStruct, get_format_by_key, MyList
from .weight import DEFAULT_BORDER
if TYPE_CHECKING:
    from .mode import Mode
    from .spch import Spch
    from .weight import Border



@dataclass
class StageSummary(BaseStruct):
    """Класс показателей работы ступени
    """
    type_spch:Spch
    mght:float
    freq:float
    kpd:float
    p_in:float
    comp_degree:float
    w_cnt:int
    p_out_req:float
    volume_rate:float
    percent_x:float
    t_out:float
    border:Border=DEFAULT_BORDER
    @property
    def freq_dim(self):
        return self.freq / self.type_spch.fnom
    @property
    def gasoline_rate(self):
        return 0
    @property
    def prime(self)->float:
        return self.border.get_obj_val(self) - .01
    @property
    def second(self)->float:
        return self.mght*self.w_cnt/(10**3)
class CompSummary(BaseCollection[StageSummary]):
    """Iterable-класс показателей работы компановки
    """
    def __init__(self, res:Iterable[StageSummary], mode:Mode, border_list:Iterable[Border]):
        """Конструктор класса результатов работы компановки

        Args:
            res (Iterable[StageSummary]): Показатель(и) работы ступени
            mode (Mode): Режим работы
        """
        super().__init__(res)
        assert len(res) == len(border_list), 'Количество граничных условий не соотвествует количеству ступеней'
        self.mode = mode
        for stage_summ , bord in zip(self, border_list):
            stage_summ.border = bord
    @property
    def border_list(self)-> List[Border]:
        return [
            stage_summ.border
        for stage_summ in self]

class CompSummaryCollection(BaseCollection[BaseStruct]):
    """Iterable-класс режимов работы компановки
    """
    def __init__(self,summaries:List[CompSummary]):
        lines = [
            BaseStruct(
                **{
                    key:comp_summ.mode[key]
                for key in comp_summ.mode.get_keys},
                **{
                    key:comp_summ[:,key]
                for key in comp_summ[0].get_keys}
            )
            for comp_summ in summaries]
        super().__init__(lines)
        
    