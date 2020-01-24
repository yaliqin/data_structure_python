from linked_list import LinkedList


class MyQueue():
    def __init__(self, max_size = 10):
        self.max_size = max_size
        self.__item__ = LinkedList()

    def __len__(self):
        return self.__item__.length

    def push(self, value):
        if(self.__len__()> self.max_size and self.max_size!= None):
            raise FullError("Queue is full")
        else:
            self.__item__.append(value)

    def pop(self):
        if (self.__len__()== 0):
            raise EmptyError(" Queue is empty")
        else:
            return self.__item__.pop_left()

def test_queue():
    q = MyQueue(10)
    q.push(0)
    q.push(1)
    q.push(2)
    q.push(3)
    # print(q.__len__())

    (q.pop())
    (q.pop())
    (q.pop())
    q.pop()
    # assert len(q)== 3
    #
    # assert q.pop() == 0
    # assert q.pop() ==1
    # assert q.pop() ==2

# test_queue()
