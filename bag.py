class Bag():
    def __init__(self, max_len =10):
        self.max_len = max_len
        self._items = list()

    def add(self,member):
        if len(self)>self.max_len:
            raise Exception('Bag is full')
        self._items.append(member)

    def remove(self,member):
        self._items.remove(member)

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        for item in self._items:
            yield item


def test_bag():
    bag = Bag()
    bag.add(1)
    bag.add(2)
    bag.add(3)

    assert len(bag)==3

    bag.remove(3)
    assert len(bag)==2

    for item in bag:
        print(item)

if __name__ == "__main__":
    test_bag()