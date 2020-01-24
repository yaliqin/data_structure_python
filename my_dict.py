from my_hash import HashTable

class DictADT(HashTable):

    def __setitem__(self, key, value):
        self.add(key,value)

    def __getitem__(self, key):
        if key not in self:
            raise KeyError()
        else:
            return self.get(key)

    def _item_slot(self):
        for slot in self._table:
            if slot not in (HashTable.UNUSED, HashTable.EMPTY):
                yield slot

    def items(self):
        for slot in self._item_slot():
            yield(slot.key, slot.value)

    def keys(self):
        for slot in self._item_slot():
            yield slot.key

    def values(self):
        for slot in self._item_slot():
            yield slot.value


def test_dict_act():
    import random
    d = DictADT()

    d['a'] = 1
    assert d['a'] == 1
    d.remove('a')

    l=list(range(30))
    random.shuffle(l)

    for i in l:
        d.add(i,i)

    for i in range(30):
        assert d.get(i) == i

    print(l)
    print((d.keys()))
    print(sorted((d.keys())))


if __name__ == "__main__":
    test_dict_act()