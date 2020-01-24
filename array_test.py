class Array():
    def __init__(self, size):
        self.size = size
        self.item = size * [None]

    def set_value(self,index, value):
        if index > self.size -1:
            raise Exception("out of size")
        else:
            self.item[index] = value

    def get_value(self, index):
        return(self.item[index])

    def clear(self):
        for index in range(self.size):
            self.item[index] = None

def test_array():
    arr = Array(10)
    arr.set_value(3,3)
    arr.set_value(0,1)

    assert arr.item[3]==3
    assert arr.get_value(0)==1

    arr.clear()
    # print(arr.item[3])
    assert arr.item[3]== None

# test_array()
