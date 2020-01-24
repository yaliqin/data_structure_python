def binary_search(sorted_array, value):
    begin = 0
    end = len(sorted_array)-1
    while begin <= end:
        middle = int((end - begin)/2)+begin
        # print(middle)
        if value == sorted_array[middle]:
            return middle
        elif value > sorted_array[middle]:
            begin = middle
        else:
            end = middle
    return -1

# def binary_search_recusive(sorted_array, value):
#     begin = 0
#     end = len(sorted_array)-1
#     middle = int((end-begin)/2)
#     if sorted_array[middle+begin]== value:
#         return middle+begin
#     elif sorted_array[middle+begin] < value:
#         binary_search_recusive(sorted_array[middle+1:], value)
#     else:
#         binary_search_recusive(sorted_array[begin:middle], value)
#     return -1

def binary_search_recursive(sorted_array, beg, end, val):
    if beg >= end:
        return -1
    mid = int((beg + end) / 2)  # beg + (end-beg)/2
    if sorted_array[mid] == val:
        return mid
    # input array keep same, change end and begin, return index is real index of the array
    elif sorted_array[mid] > val:
        return binary_search_recursive(sorted_array, beg, mid, val)    # 注意我依然假设 beg, end 区间是左闭右开的
    else:
        return binary_search_recursive(sorted_array, mid+1, end, val)




def test_binary_search():
    sorted_a = [0,1,2,3,4,5,6,7,8]
    index = binary_search(sorted_a, 5)
    print(index)

def test_binary_search_recusive():
    sorted_a = [0,1,2,3,4,5,6,7,8]
    beg = 0
    end = len(sorted_a)-1
    index = binary_search_recursive(sorted_a, beg, end, 5)
    print(index)


if __name__ == "__main__":
    test_binary_search_recusive()