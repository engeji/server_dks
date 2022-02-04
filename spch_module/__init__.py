"""Пакет для моделирования ДКС
"""
import math
import os
import sys
from collections import namedtuple
from itertools import groupby
from types import SimpleNamespace
from typing import List, Union, Iterable

import numpy as np
import xlrd

from .spch import Spch
from .spch_init import SpchInit
from .comp import Comp, Stage
from .limit import Limit, DEFAULT_LIMIT

PATH_BASE = r'spch_module\base'
PATH_BASE_FILES = PATH_BASE + r'\text_files'
wb = xlrd.open_workbook(PATH_BASE + r'\dbqp.xls')
all_data = [{'name':lis, 'lis': wb.sheet_by_name(lis)} for lis in wb.sheet_names()]
ALL_SPCH_LIST:List[Spch] = list(filter(lambda x : float(x.mgth) == 16.0, [
    Spch((SpchInit(item['lis'])))
for item in all_data]))

for f in os.listdir(PATH_BASE_FILES):
    with open(f'{PATH_BASE_FILES}\\{f}', 'r') as my_file:
        lines = my_file.read()
    ALL_SPCH_LIST.append(Spch(SpchInit(None, lines, '.'.join(f.split('.')[:-1]))))

