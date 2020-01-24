class Node:
    def __init__(self,value = None, next= None ):
        self.value = value
        self.next = next


class LinkedList:
    def __init__(self, max_size = None):
        node = Node()
        self.root = node
        self.max_size = max_size
        self.length = 0
        self.tail = None
        self.head = None

    def len(self):
        return self.length

    def append(self, value):
        if self.length == self.max_size:
            raise Exception("list is full")
        new_node = Node(value)
        # if this is the first node
        if self.tail == None:
            self.head = new_node
            self.tail = new_node
            self.root.next= self.tail
        # there are already some nodes
        else:
            self.tail.next = new_node
        self.tail = new_node
        self.length += 1
        print(self.tail.value)

    def append_left(self,value):
        new_node = Node(value)
        self.root.next = new_node
        new_node.next = self.head
        self.head = new_node
        self.length += 1

    def iter_node(self):
        current_node = self.root.next
        while current_node is not self.tail:
            yield current_node
            current_node = current_node.next
        yield current_node

    def iter(self):
        for node in self.iter_node():
            yield node.value

    def rem(self, value):
        print("remove function")
        previous_node = self.root
        current_node = self.root.next
        while current_node is not self.tail:
            if current_node.value == value:
                print(f"remove {value}")
                previous_node.next = current_node.next
                current_node = current_node.next
                self.length -= 1
            else:
                previous_node = current_node
                current_node = current_node.next
        if self.tail.value == value:
            self.tail = previous_node

    def find(self,value):
        current_node = self.head
        index = 0
        indexs=[]
        while current_node is not self.tail:
            if current_node.value == value:
                indexs.append(index)
            current_node = current_node.next
            index += 1
        if len(indexs) != 0:
            return(indexs)
        else:
            return('Not found')

    def pop_left(self):
        if self.root.next == None:
            raise Exception("No element left")
        head = self.head
        self.root.next = self.head.next
        self.head = head.next
        self.length -= 1
        value = head.value
        if self.tail is head:
            self.tail = None

        del head
        return value


    def clear(self):
        for node in self.iter_node():
            del node
        self.root.next = None
        self.length = 0


def test_linked_list():
    l_list = LinkedList(10)
    l_list.append(2)
    l_list.append(3)
    l_list.append(4)
    l_list.append_left(1)
    l_list.append(5)
    l_list.append(6)
    l_list.append(7)
    # l_list.iter()
    # print("before remove")
    # l_list.rem(3)
    # print("after remove")
    # index = l_list.find(3)
    # print(index)
    # l_list.iter()
    h = l_list.pop_left()
    print(f"first pop value is {h}")
    h = l_list.pop_left()
    print(f"secode pop value is {h}")

    h = l_list.pop_left()
    print(f"third pop value is {h}")

    # l_list.clear()
    # assert l_list.length==0


# test_linked_list()
