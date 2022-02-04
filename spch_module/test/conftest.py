"""[summary]
"""
from collections import namedtuple
from typing import List

import pandas
import pytest
from ..comp import Comp
from ..facilities import get_comp_by_name
from ..limit import GET_DEFAULT_LIMIT
from ..mode import Mode, ModeCollection
from ..header import MyList
from ..facilities import get_comp_by_name
PATH_EXCEL = 'data_for_test\\for_tst_2step.xlsx'

f_excel = pandas.ExcelFile(PATH_EXCEL)
df: pandas.DataFrame = f_excel.parse(
    sheet_name=f_excel.sheet_names[0]
)
prove_cls = namedtuple('prove_tup', df.columns)

list_prove:List[prove_cls] = [
    prove_cls(**{
        column:row[1][column]
    for column in df.columns})
for row in df.iterrows()]

@pytest.fixture
def mode_coll()->ModeCollection:
    return ModeCollection([
        Mode(
            GET_DEFAULT_LIMIT("t_in"),
            prove.q_in,
            float(prove.p_in.split('+')[0])
        )
    for prove in list_prove])

@pytest.fixture
def list_comp()->List[Comp]:
    res = []
    for prove in list_prove:
        res.append(
            get_comp_by_name(
                prove.type_spch.split('+'),
                [
                    float(wcnt)
                for wcnt in prove.w_cnt.split('+')],
            )
        )
    return res

@pytest.fixture
def main_comp()-> Comp:
    return get_comp_by_name([
        'ГПА-ц3-16С-45-1.7(ККМ)',
        'ГПА Ц3 16с76-1.7М'
    ],[1,1])

@pytest.fixture
def main_mode()-> Mode:
    return Mode(283, 15, 2.65)
