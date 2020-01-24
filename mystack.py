from double_queue import MyDoubleQueue

class MyStack():
    def __init__(self,max_size = None):
        node = MyDoubleQueue()
        self.element = node
        self.max_size = max_size
        self.length = 0

    def push(self,value):
        if self.max_size != None and self.length > self.max_size:
            raise Exception("stack is full")
        else:
            self.element.push_left(value)
            self.length += 1
            # print(f"push_left value is {value}, and length is {self.length}")

    def pop(self):
        if self.length > 0:
            value = self.element.pop_left()
            self.length -= 1
            return value

def test_stack():
    s = MyStack()
    s.push(1)
    s.push(2)
    s.push(3)
    s.push(4)
    assert(s.pop()==4)
    assert(s.pop()==3)
    assert(s.pop()==2)
    assert(s.pop()==1)

# test_stack()