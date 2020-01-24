from my_hash import HashTable

class SetADT(HashTable):
    def add(self, key):
        return super(SetADT, self).add(key, value=True)
        self.add(key, value = True)

    def __and__(self, other):
        new_set = SetADT()
        for element_a in self:
            if element_a in other:
                new_set.add(element_a)
        for element_b in other:
            if element_b in self:
                new_set.add(element_b)
        return new_set

    def __sub__(self, other):
        new_set = SetADT()
        for element_a in self:
            if element_a not in other:
                new_set.add(element_a)
        for element_b in other:
            if element_b not in self:
                new_set.add(element_b)
        return new_set

    def __or__(self, other):
        new_set = SetADT()
        for element_a in self:
            new_set.add(element_a)
        for element_b in other:
            new_set.add(element_b)
        return new_set

    def remove(self,key):
        super(SetADT,self).remove(key)

    # def pop(self):
    #     self.length = super(SetADT, self).length
    #     self.
    #     self.remove()
    #
def test_set_adt():
    sa = SetADT()
    sa.add(1)
    sa.add(2)
    sa.add(3)

    assert 1 in sa

    sb=SetADT()
    sb.add(3)
    sb.add(4)
    sb.add(5)

    assert sorted(list(sa &sb)) ==[3]


if __name__ == "__main__":
    test_set_adt()