# from array_test import Array

class Array(object):
    def __init__(self, size=32, init=None):
        self._size = size
        self._items = [init] * size

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        self._items[index] = value

    def __len__(self):
        return self._size

    def clear(self, value=None):
        for i in range(len(self._items)):
            self._items[i] = value

    def __iter__(self):
        for item in self._items:
            yield item
#
# class Hash:
#     def __init__(self):
#         self.hash_table = [[None, None] for i in range(11)]
#
#     def hash(self, k, i):
#         h_value = (k+i) % 11
#         if self.hash_table[h_value][0] == k:
#             return h_value
#         if self.hash_table[h_value][0] != None:
#             i += 1
#             h_value = self.hash(k,i)
#             return h_value
#
#     def put(self,k,v):
#         hash_v = self.hash(k,0)
#         self.hash_table[hash_v][0] = k
#         self.hash_table[hash_v][1] = v
#
#
# class Map:
#     def __init__(self):
#         self.capcity = 11
#         self.hash_table = [[None, None] for i in range(self.capacity)]
#         self.num = 0
#         self.load_factor = 0.75
#
#     def hash(self,k,i):
#         h_value = (k+i)%self.capcity
#         if self.hash_table[h_value][0] == k:
#             return h_value
#         if self.hash_table[h_value][0]!= None:
#             i += 1
#             h_value = self.hash(k,i)
#         return h_value
#
#     def resize(self):
#         self.capcity = self.num * 2
#         temp = self.hash_table[:]
#         self.hash_table = [[None, None] for i in range(self.capacity)]
#         for i in temp:
#             if (i[0] != None):
#                 hash_v = self.hash(h[0],0)
#                 self.hash_table[hash_v][0] = i[0]
#                 self.hash_table[hash_v][1] = i[1]
#
#     def put(self, k, v)
#         hash_v = self.hash(k,0):
#         self.hash_table[hash_v][0] = k
#         self.hash_table[hash_v][1] = v
#         self.num += 1
#         if(self.num/len(self.hash_table > self.load_factor)):
#             self.resie()
#
#     def get(self,k):
#         hash_v = self.hash(k, 0)
#         return self.hash_table[hash_v][1]
#
#
# class MyDictionary(object):
#     # 字典类的初始化
#     def __init__(self):
#         self.table_size = 13  # 哈希表的大小
#         self.key_list = [None] * self.table_size  # 用以存储key的列表
#         self.value_list = [None] * self.table_size  # 用以存储value的列表
#
#     # 散列函数，返回散列值
#     # key为需要计算的key
#     def hashfuction(self, key):
#         count_char = 0
#         key_string = str(key)
#         for key_char in key_string:  # 计算key所有字符的ASCII值的和
#             count_char += ord(key_char)  # ord()函数用于求ASCII值
#         length = len(str(count_char))
#         if length > 3:  # 当和的位数大于3时，使用平方取中法，保留中间3位
#             mid_int = 100 * int((str(count_char)[length // 2 - 1])) \
#                       + 10 * int((str(count_char)[length // 2])) \
#                       + 1 * int((str(count_char)[length // 2 + 1]))
#         else:  # 当和的位数小于等于3时，全部保留
#             mid_int = count_char
#
#         return mid_int % self.table_size  # 取余数作为散列值返回
#
#     # 重新散列函数，返回新的散列值
#     # hash_value为旧的散列值
#     def rehash(self, hash_value):
#         return (hash_value + 3) % self.table_size  # 向前间隔为3的线性探测
#
#     # 存放键值对
#     def __setitem__(self, key, value):
#         hash_value = self.hashfuction(key)  # 计算哈希值
#         if None == self.key_list[hash_value]:  # 哈希值处为空位，则可以放置键值对
#             pass
#         elif key == self.key_list[hash_value]:  # 哈希值处不为空，旧键值对与新键值对的key值相同，则作为更新，可以放置键值对
#             pass
#         else:  # 哈希值处不为空，key值也不同，即发生了“冲突”，则利用重新散列函数继续探测，直到找到空位
#             hash_value = self.rehash(hash_value)  # 重新散列
#             while (None != self.key_list[hash_value]) and (key != self.key_list[hash_value]):  # 依然不能插入键值对，重新散列
#                 hash_value = self.rehash(hash_value)  # 重新散列
#         # 放置键值对
#         self.key_list[hash_value] = key
#         self.value_list[hash_value] = value
#
#     # 根据key取得value
#     def __getitem__(self, key):
#         hash_value = self.hashfuction(key)  # 计算哈希值
#         first_hash = hash_value  # 记录最初的哈希值，作为重新散列探测的停止条件
#         if None == self.key_list[hash_value]:  # 哈希值处为空位，则不存在该键值对
#             return None
#         elif key == self.key_list[hash_value]:  # 哈希值处不为空，key值与寻找中的key值相同，则返回相应的value值
#             return self.value_list[hash_value]
#         else:  # 哈希值处不为空，key值也不同，即发生了“冲突”，则利用重新散列函数继续探测，直到找到空位或相同的key值
#             hash_value = self.rehash(hash_value)  # 重新散列
#             while (None != self.key_list[hash_value]) and (key != self.key_list[hash_value]):  # 依然没有找到，重新散列
#                 hash_value = self.rehash(hash_value)  # 重新散列
#                 if hash_value == first_hash:  # 哈希值探测重回起点，判断为无法找到了
#                     return None
#             # 结束了while循环，意味着找到了空位或相同的key值
#             if None == self.key_list[hash_value]:  # 哈希值处为空位，则不存在该键值对
#                 return None
#             else:  # 哈希值处不为空，key值与寻找中的key值相同，则返回相应的value值
#                 return self.value_list[hash_value]
#
#     # 删除键值对
#     def __delitem__(self, key):
#         hash_value = self.hashfuction(key)  # 计算哈希值
#         first_hash = hash_value  # 记录最初的哈希值，作为重新散列探测的停止条件
#         if None == self.key_list[hash_value]:  # 哈希值处为空位，则不存在该键值对，无需删除
#             return
#         elif key == self.key_list[hash_value]:  # 哈希值处不为空，key值与寻找中的key值相同，则删除
#             self.key_list[hash_value] = None
#             self.value_list[hash_value] = None
#             return
#         else:  # 哈希值处不为空，key值也不同，即发生了“冲突”，则利用重新散列函数继续探测，直到找到空位或相同的key值
#             hash_value = self.rehash(hash_value)  # 重新散列
#             while (None != self.key_list[hash_value]) and (key != self.key_list[hash_value]):  # 依然没有找到，重新散列
#                 hash_value = self.rehash(hash_value)  # 重新散列
#                 if hash_value == first_hash:  # 哈希值探测重回起点，判断为无法找到了
#                     return
#             # 结束了while循环，意味着找到了空位或相同的key值
#             if None == self.key_list[hash_value]:  # 哈希值处为空位，则不存在该键值对
#                 return
#             else:  # 哈希值处不为空，key值与寻找中的key值相同，则删除
#                 self.key_list[hash_value] = None
#                 self.value_list[hash_value] = None
#                 return
#
#     # 返回字典的长度
#     def __len__(self):
#         count = 0
#         for key in self.key_list:
#             if key != None:
#                 count += 1
#         return count

class Slot(object):
    def __init__(self, key, value):
        self.key, self.value = key, value

class HashTable(object):
    UNUSED = None
    EMPTY = Slot(None, None) #used but deleted

    def __init__(self):
        self._table = Array(8, init = HashTable.UNUSED)
        self.length = 0

    @property
    def _load_factor(self):
        return self.length / float(len(self._table))

    def __len__(self):
        return self.length

    def __hash(self, key):
        return abs(hash(key)) % len(self._table)

    def _find_key(self,key):
        index = self.__hash(key)
        _len = len(self._table)
        while self._table[index] is not HashTable.UNUSED:
            if self._table[index] is HashTable.EMPTY:
                index = (index * 5 +1) % _len
            elif self._table[index].key == key:
                return index
            else:
                index = (index * 5 + 1)% _len
        return None

    def _slot_can_insert(self,index):
        temp_slot = self._table[index]
        return(temp_slot is HashTable.EMPTY or temp_slot is HashTable.UNUSED)

    def _find_slot_for_insert(self, key):
        index = self.__hash(key)
        _len = len(self._table)
        while not self._slot_can_insert(index):
            index = (index * 5 + 1) % _len
        return index

    def __contains__(self, key):
        index = self._find_key(key)
        return index is not None

    def add(self, key, value):
        if key in self:
            index = self._find_key(key)
            print(index)
            self._table[index] = value
            return False
        else:
            index = self._find_slot_for_insert(key)
            self._table[index] = Slot(key, value)
            self.length += 1
            if self._load_factor >= 0.8:
                self._rehash()
            return True

    def _rehash(self):
        old_table = self._table
        resize = 2 * len(self._table)
        self._table = Array(resize, self.UNUSED)
        self.length = 0

        for slot in old_table:
            if slot is not HashTable.UNUSED and slot is not HashTable.EMPTY:
                index = self._find_slot_for_insert(slot.key)
                self._table[index] = slot #??? why not slot.value
                self.length += 1

    def get(self, key, default = None):
        index = self._find_key(key)
        if index is None:
            return default
        else:
            return self._table[index].value

    def remove(self,key):
        index = self._find_key(key)
        if index is None:
            raise KeyError()
        value = self._table[index].value
        self. length -= 1
        self._table[index] = HashTable.EMPTY
        return value

    def __iter__(self):
        for slot in self._table:
            if slot not in (HashTable.EMPTY, HashTable.UNUSED):
                yield slot.key

def test_hash_table():
    h = HashTable()
    h.add('a',0)
    h.add('b',1)
    h.add('c',2)

    assert len(h)==3
    print(h.get('a'))

    h.remove('a')
    assert h.get('a') is None
    l = list(h)
    assert sorted(list(h)) == ['b', 'c']

    n = 50
    for i in range(n):
        h.add(i, i)

    for i in range(n):
        assert h.get(i) == i


if __name__ == '__main__':
    test_hash_table()





# def main():
#     H = MyDictionary()
#     H["kcat"] = "cat"
#     H["kdog"] = "dog"
#     H["klion"] = "lion"
#     H["ktiger"] = "tiger"
#     H["kbird"] = "bird"
#     H["kcow"] = "cow"
#     H["kgoat"] = "goat"
#     H["pig"] = "pig"
#     H["chicken"] = "chicken"
#     print("字典的长度为%d" % len(H))
#     print("键 %s 的值为为 %s" % ("kcow", H["kcow"]))
#     print("字典的长度为%d" % len(H))
#     print("键 %s 的值为为 %s" % ("kmonkey", H["kmonkey"]))
#     print("字典的长度为%d" % len(H))
#     del H["klion"]
#     print("字典的长度为%d" % len(H))
#     print(H.key_list)
#     print(H.value_list)
#
#
# if __name__ == "__main__":
#     main()
#
#
#
