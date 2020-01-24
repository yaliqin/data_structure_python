class Node():
    def __init__(self, value = None, pre= None, next= None):
        self.value, self.pre, self.next = value, pre, next

class CircularDoublelinedList():
    def __init__(self,max_size = None):
        node = Node()
        node.next, node.pre = node, node
        self.root = node
        self.length = 0
        self.max_size = max_size

    def __len__(self):
        return self.length

    def get_head(self):
        return self.root.next

    def get_tail(self):
        return self.root.pre

    def append(self, value, model):
        if model==1:
            if self.max_size is not None and len(self) >= self.max_size:
                raise Exception('LinkedList is Full')
            node = Node(value=value)
            # if self.length == 0:
            #     tailnode = self.root
            # else:
            #     tailnode = self.get_tail()
            tailnode = self.get_tail() or self.root

            tailnode.next = node
            node.pre = tailnode
            node.next = self.root
            self.root.pre = node
            self.length += 1
        else:
            if self.max_size is not None and self.length == self.max_size:
                raise Exception('full list')
            node = Node(value = value)

            if self.length == 0:
                self.root.next = node
                node.pre = self.root
                self.root.pre = node
                node.next = self.root

            else:
                tail_node = self.get_tail()
                tail_node.next = node
                node.pre = tail_node
                node.next = self.root
                self.root.pre = node

            self.length += 1

    def append_left(self, value):
        if self.max_size is not None and self.length == self.max_size:
            raise FileExistsError('full list')

        node = Node(value)
        # empty list
        if self.root.next is self.root:
            node.next = self.root
            node.pre = self.root
            self.root.next = node
            self.root.pre = node
        else:
            head = self.get_head()
            head.pre = node
            node.next = head
            self.root.next = node
            node.pre = self.root

        self.length += 1

    def remove(self,node):
        if node is self.root:
            return
        else:
            node.pre.next = node.next
            node.next.pre = node.pre
            self.length -= 1
            return node

    def iter_node(self):
        if self.root.next is self.root:
            return
        current_node = self.root.next
        while current_node.next is not self.root:
            yield current_node
            current_node = current_node.next
        yield current_node

    def __iter__(self):
        for node in self.iter_node():
            yield node.value

    def iter_node_reverse(self):
        if self.root.next is self.root:
            return
        current_node = self.get_tail()
        while current_node != self.root:
            yield current_node
            current_node = current_node.pre

    # insert a node with new_value before the node whose value is value
    def insert(self,value, new_value):
        new_node = Node(new_value)
        for node in self.iter_node():
            if node.value == value:
                new_node.next = node
                new_node.pre = node.pre
                node.pre= new_node
        self.length += 1

def test_double_link_list():
    dll = CircularDoublelinedList(10)
    assert dll.length == 0

    dll.append(0,1)
    assert dll.length == 1

    dll.append(1,1)
    dll.append(2,1)

    # for node in dll.iter_node():
    #
    #     print(node.value)
    #
    # for node in dll.iter_node_reverse():
    #     print(node.value)

    #
    # assert list(dll) == [0,1,2]
    #
    # assert [node.value for node in dll.iter_node()]==[0,1,2]
    # assert [node.value for node in dll.iter_node_reverse()]==[2,1,0]

    head = dll.get_head()
    dll.remove(head)

    assert dll.length == 2

    head2 = dll.get_head()
    assert head2.value == 1

    dll.append_left(0)
    assert [node.value for node in dll.iter_node()]==[0,1,2]

    dll.insert(1,5)

    # for node in dll.iter_node():
    #     print(node.value)


# test_double_link_list()