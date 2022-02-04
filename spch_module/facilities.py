"""Модуль для вспомогательных расчетов, формирования компановок и т.д.
"""
from typing import Iterable, Union

from . import ALL_SPCH_LIST
from .comp import Comp
from .limit import DEFAULT_LIMIT, Limit
from .spch import Spch
from .stage import Stage
from .weight import DEFAULT_BORDER, Border
from .mode import Mode
from .summary import CompSummary, CompSummaryCollection, StageSummary

def get_spch_by_name(name:str)->Spch:
    """Возвращяет обьект класса Spch по названию СПЧ

    Args:
        name (str): Название СПЧ

    Returns:
        Spch: Объект класса Spch
    """
    return next(filter(lambda sp: sp.name == name, ALL_SPCH_LIST))

def get_comp_by_name(
    names:Union[str, Iterable[str]],
    w_cnt:Union[int, Iterable[int]],
    lim:Limit=DEFAULT_LIMIT)->Comp:
    """Создает компановку ДКС

    Args:
        names (Union[str, Iterable[str]]): название(я) спч на ступенях ДКС
        w_cnt (Union[int, Iterable[int]]): Максимальное количество ГПА
        lim (Limit, optional): ПВТ-свойства. Defaults to DEFAULT_LIMIT.

    Returns:
        Comp: Компановка ДКС
    """
    if isinstance(w_cnt, Iterable):
        assert isinstance(names, Iterable)
        stages = [
            Stage(
                type_spch=get_spch_by_name(name),
                w_cnt=w_cnt[idx],
                lim=lim,
                idx=idx
            )
        for idx, name in enumerate(names)]
    else:
        assert isinstance(w_cnt, int)
        stages = Stage(
            type_spch=get_spch_by_name(names),
            w_cnt=w_cnt,
            lim=lim
        )
    return Comp(stages)

def get_summ_collections(
    modes:Iterable[Mode],
    sums:Union[Iterable[StageSummary],Iterable[CompSummary]],
    list_border:Iterable[Border]=[DEFAULT_BORDER])->CompSummaryCollection:
    # assert not isinstance(modes, Iterable), 'Нужно несколько режимов'
    # assert not isinstance(sum, Iterable), 'Нужно несколько режимов'
    # assert not len(modes) == len(sums), 'Количество режимов и расчетов не совподает'
    if isinstance(sums[0], StageSummary):
        return CompSummaryCollection([
            CompSummary([summ], mode, list_border)
        for summ, mode, in zip(sums,modes)])
    elif isinstance(sums[0], CompSummary):
        return CompSummaryCollection(sums)
    else:
        assert True, 'ТЫ чо пёс'
