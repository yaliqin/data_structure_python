from collections import deque

def fact(n):
    if n == 0:
        return 1
    else:
        return n*fact(n-1)

def print_num_recursive(n):
    if n>0:
        print_num_recursive(n-1)
        print(n)


class Stack():
    def __init__(self):
        self._deque = deque()

    def push(self, value):
        return self._deque.append(value)

    def pop(self):
        return self._deque.pop()

    def is_empty(self):
        return len(self._deque)==0

def print_number_use_stack(n):
    s = Stack()
    while n>0:
        s.push(n)
        n -= 1
    while not s.is_empty():
        print(s.pop())

def move(n, s, d, i):
    if n>= 1:
        move(n-1,s,i, d)
        print(f"move {n} from {s} to {d}")
        move(n-1, i, d,s)

move(5, 'a','b','c')
