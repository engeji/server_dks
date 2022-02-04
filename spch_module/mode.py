"""[summary]
"""
from dataclasses import dataclass
from typing import Iterable, Union, List
from collections import namedtuple

from .header import BaseCollection, BaseStruct, MyList
from .limit import Limit

class Mode(BaseStruct):
    """
    Класс режимов работы ДКС

    >>> Mode(273, 15, 2.67)
    Mode(t_in=273, q_in=15, p_input=2.67)
    >>> Mode(273, [15,20], 2.67)
    Mode(t_in=273, q_in=15+20, p_input=2.67)

    Args:
        BaseStruct ([type]): [description]

    Returns:
        [type]: [description]
    """
    def __init__(self, t_in, q_in, p_input):
        self.t_in = t_in
        self._q_in = MyList(q_in)
        self.p_input = p_input
    def __repr__(self):
        # Mode(t_in=273, q_in=[15], p_in=2.67)
        return f"Mode(t_in={self.t_in}, q_in={self.q_in}, p_input={self.p_input})"
    @property
    def get_keys(self):
        return ['q_in', 't_in', 'p_input',]
    @property
    def q_in(self):
        return self._q_in
    @q_in.setter
    def q_in(self, value):
        self._q_in = MyList(value)
    def clone(self):
        return Mode(self.t_in, self._q_in, self.p_input)
class ModeCollection(BaseCollection[Mode]):
    """Iterable-класс режимов работы

    >>> ModeCollection(Mode(273, 15, 2.67))
    <BLANKLINE>
    Комер. расх. |Т.вх |Давл. (треб) 
     млн. м3/сут |  К  |     МПа     
        15.00    | 273 |    2.67     
    >>> ModeCollection([
    ... Mode(273, 15, 2.67),
    ... Mode(273, [15,20], 2.67)
    ... ])
    <BLANKLINE>
    Комер. расх. |Т.вх |Давл. (треб) 
     млн. м3/сут |  К  |     МПа     
        15.00    | 273 |    2.67     
     15.00+20.00 | 273 |    2.67     
    """
    def __init__(self, modes:Union[Mode, Iterable[Mode]]):
        """Конструктор класса массива режимов работы

        Args:
            modes (Union[Mode, Iterable[Mode]]): Режим(ы) работы
        """
        super().__init__(modes)