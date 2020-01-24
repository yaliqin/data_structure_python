from array_test import Array
class MaxHeap():
    def __init__(self,  max_size):
        self.max_size = max_size
        self._count = 0
        self._elements = Array(max_size)

    def _shiftup(self,index):
        parent_index = int((index-1)/2)
        if self._elements[parent_index]<self._elements[index]:
            self._elements[index], self._elements[parent_index] = self._elements[parent_index],self._elements[index]
            self._shiftup(parent_index)

    def add(self,value):
        if self._count>= self.max_size:
            raise Exception('full')
        last_index = self._count+1
        self._elements[last_index] = value
        self._shiftup(last_index)

    def extract(self):
        if self._count <= 0:
            raise Exception('empty')
        value = self._elements[0]  # 保存 root 值
        self._count -= 1
        self._elements[0] = self._elements[self._count]  # 最右下的节点放到root后siftDown
        self._siftdown(0)  # 维持堆特性
        return value

    def _siftdown(self, ndx):
        left = 2 * ndx + 1
        right = 2 * ndx + 2
        # determine which node contains the larger value
        largest = ndx
        if (left < self._count and  # 有左孩子
                self._elements[left] >= self._elements[largest] and
                self._elements[left] >= self._elements[right]):  # 原书这个地方没写实际上找的未必是largest
            largest = left
        elif right < self._count and self._elements[right] >= self._elements[largest]:
            largest = right
        if largest != ndx:
            self._elements[ndx], self._elements[largest] = self._elements[largest], self._elements[ndx]
            self._siftdown(largest)



def test_maxheap():
    import random
    n = 5
    h = MaxHeap(n)
    for i in range(n):
        h.add(i)
    for i in reversed(range(n)):
        assert i == h.extract()

test_maxheap()