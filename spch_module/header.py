"""Модуль для класса заголовков для переопределения repr в виде таблицы
"""
from typing import Iterable, Union, List, TypeVar, Generic


HEADERS_LIST = [
    {"fmt":".2f", "key":"q_in", "title":"Комер. расх., млн. м3/сут"},
    {"fmt":".0f", "key":"t_in", "title":"Т.вх, К"},
    {"fmt":".2f", "key":"p_input", "title": "Давл. (треб), МПа"},
    {"fmt":".2f", "key":"p_out_req", "title": "Давл. вых(треб), МПа"},
    {"fmt":".2f", "key":"p_in", "title": "Давл. вх, МПа"},
    {"fmt":".2f", "key":"p_out", "title": "Давл. вых, МПа"},
    {"fmt":""   , "key":"type_spch", "title": "Тип СПЧ,"},
    {"fmt":".0f", "key":"freq", "title": "Частота, об/мин"},
    {"fmt":".0f", "key":"mght", "title": "Мощность, кВт"},
    {"fmt":""   , "key":"isWork", "title": "Режим,"},
    {"fmt":".2f", "key":"p_out_res", "title": "Давл. вых.(расч), МПа"},
    {"fmt":".2f", "key":"comp_degree", "title":"Ст. сжатия, д. ед."},
    {"fmt":".0f", "key":"w_cnt", "title": "ГПА, шт"},
    {"fmt":".0f", "key":"gasoline_rate", "title":"Расход топлива, тыс м3/сут"},
    {"fmt":".0f", "key":"t_out", "title": "Т.вых, С"},
    {"fmt":".0f", "key":"percent_x", "title":"Помп. удал, д. ед"},
    {"fmt":".0f", "key":"volume_rate", "title":"Об. расход, м3/мин"},
    {"fmt":".2f", "key":"kpd", "title":"Пол. кпд, д. ед"},
    {"fmt":""   , "key":"key", "title":"Вектор,"},
    {"fmt":".2f", "key":"max_val", "title":"Максимум,"},
    {"fmt":".2f", "key":"min_val", "title":"Минимум,"},
    {"fmt":".0f", "key":"weight", "title":"Вес,"},

    # {"fmt":".2f", "key":"freq_dim", "title":"От. частота, д. ед"},
]

def GET_FILTER_KEYS(list_key:Iterable[str])->Iterable[str]:
    return map(lambda dic2: dic2['key'], filter(lambda dic: dic['key'] in list_key, HEADERS_LIST))

def get_dic_HEADERS_LIST(header:str):
    return next(filter(lambda dic: header in dic.values(), HEADERS_LIST))

def header_titles_from_list(headers:Iterable[str])->Iterable[Iterable[str]]:
    return zip(*[
        get_dic_HEADERS_LIST(head)['title'].split(',')
    for head in headers])

def get_format_by_key(key:str)->str:
    return next(filter(lambda dic: dic['key']==key, HEADERS_LIST))['fmt']

class Header:
    def __init__(self, header_titles:Iterable[str]):
        self._header, self._dimen = header_titles_from_list(header_titles)
        self._max_len = [len(item) for item in self._header]
        self._data:Iterable[Iterable[str]] = []

    def add_data(self, data_line:Iterable[str]):
        self._max_len = [
            len(str(value)) if self._max_len[ind] < len(str(value)) else self._max_len[ind]
        for ind, value in enumerate(data_line)]
        self._data.append(data_line)
    def __repr__(self):
        self._line_split = '|'.join([f'{"":{"="}^{val}}' for ind, val in enumerate(self._max_len)])
        self._line1 = '|'.join([f'{val:^{self._max_len[ind]+1}}'
            for ind, val in enumerate(self._header)])
        self._line2 = '|'.join([f'{val:^{self._max_len[ind]+1}}'
            for ind, val in enumerate(self._dimen)])
        return f'\n'.join([
            '\n'.join([
                "",
                self._line1,
                self._line2
            ]),
            *[
                "|".join([
                    f"{item:^{self._max_len[ind]+1}}"
                for ind, item in enumerate(line)])
            for line in self._data]
        ])

T = TypeVar('T')
class BaseStruct:
    def __init__(self, **kwargs):
        self.__dict__ = kwargs
    def __repr__(self):
        res = ','.join([
            f"{key}:{self[key]}"
        for key in self.__dict__])
        return f"BaseStruct({res})"
    @property
    def get_keys(self):
        return GET_FILTER_KEYS(self.__dict__.keys())
    @property
    def get_values(self):
        return [
            self.__dict__[key]
        for key in self.get_keys]
    def __getitem__(self, key):
        return self.__getattribute__(key)
    def __eq__(self, other):
        return all((
            self.__getattribute__("prime") == other.prime,
            self.__getattribute__("second") == other.second
        ))
    def __ne__(self, other):
        return not self.__eq__(other)
    

class MyList:
    def __init__(self, val:Union[T,List[T]]):
        # if isinstance(items, Iterable):
            # super().__init__(items[0].get_keys)
        if isinstance(val, list):
            self._items = val
        elif isinstance(val, MyList):
            self._items = val._items
        else:
            self._items = [val]
    def __format__(self, fmt):
        return '+'.join([
            f'{item:{fmt}}'
        for item in self._items])
    def __getitem__(self,idx)->float:
        if len(self) <= idx:
            return self._items[-1]
        return self._items[idx]
    def __repr__(self):
        return repr(self._items)
    def __len__(self):
        return len(self._items)

class BaseCollection(Header, Generic[T]):
    def __init__(self, items:Union[T, Iterable[T]]):
        self._idx = 0
        self._list_items:List[T] = []
        if isinstance(items, Iterable):
            super().__init__(items[0].get_keys)
            self._list_items = list(items)
            for item in items:
                self.add_data([
                    f'{item[key]:{get_format_by_key(key)}}'
                for key in item.get_keys])
        else:
            super().__init__(items.get_keys)
            self._list_items = [items]
            self.add_data([
                f'{items[key]:{get_format_by_key(key)}}'
            for key in items.get_keys])
    # def __iter__(self):
    #     return self
    def __next__(self)->T:
        try:
            item = self._list_items[self._idx]
        except IndexError as idx_err:
            self._idx = 0
            raise StopIteration() from idx_err
        self._idx +=1
        return item
    def __iter__(self):
        return self
    def __len__(self)->int:
        return len(self._data)
    def __getitem__(self, indecies)->T:
        if isinstance(indecies,tuple):
            dim1, dim2 = indecies       
            return MyList([
                item[dim2]
            for item in self._list_items[dim1]])
        else:
            return self._list_items[indecies]
    @property
    def all_second(self):
        return sum(map(lambda x: x.second, self))
    @property
    def all_prime(self):
        val = sum(map(lambda x: x.prime, self))
        return val if val >= 0 else 0
    def __gt__(self, other):
        if self.all_prime + other.all_prime <= 0:
            return self.all_second > other.all_second
        elif self.all_prime <= 0:
            return False
        elif other.all_prime <= 0:
            return True
        return self.all_second > other.all_second
    def __lt__(self, other):
        return not self.__gt__(other)