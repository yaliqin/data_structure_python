from double_link_list import CircularDoublelinedList, Node

class MyDoubleQueue():
    def __init__(self, max_size = None):
        self.max_size = max_size
        self.items = CircularDoublelinedList()
        self.length = 0

    def __len__(self):
        return self.length

    # push to head side
    def push_left(self, value):
        if self.max_size!= None and self.length >= self.max_size:
            raise Exception("Double Queue is full")
        new_node = Node(value)
        head_node = self.items.get_head()
        head_node.pre = new_node
        new_node.next = head_node
        new_node.pre = self.items.root
        self.items.root.next = new_node
        self.length += 1
        # print(f"push_left value is {new_node.value}, and length is {self.length}")

    # push to tail side, traditional way
    def push_right(self,value):
        if self.max_size!= None and self.length >= self.max_size:
            raise Exception("Double Queue is full")
        new_node = Node(value)
        tail = self.items.get_tail()
        tail.next = new_node
        new_node.next = self.items.root
        new_node.pre = tail
        self.items.root.pre = new_node
        self.length += 1
        # print(f"push right value is {new_node.value}, and length is {self.length}")


    # remove from head
    def pop_left(self):
        if (self.length <= 0):
            raise exec("Double Queue is Empty")
        head = self.items.get_head()
        self.items.remove(head)
        # print(f"pop left value is {head.value}, and length is {self.length}")
        return head.value

    def pop_right(self):
        if (self.length <= 0):
            raise exec("Double Queue is Empty")
        tail = self.items.get_tail()
        self.items.remove(tail)
        return tail.value

def test_double_queue():
    dd = MyDoubleQueue()
    dd.push_left(2)
    dd.push_left(1)
    dd.push_right(3)
    dd.push_right(4)

    assert dd.__len__()==4

    d1 = dd.pop_left()
    d2 = dd.pop_left()
    d3 = dd.pop_left()
    d4 = dd.pop_left()

    print("pop data")
    print(d1,d2,d3,d4)

# test_double_queue()