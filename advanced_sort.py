def quick_sort(seq):
    n = len(seq)
    if n<2 :
        return seq
    index = 0
    value = seq[index]
    # for i in range(1,n):

    left_part = [seq[i] for i in range(1,n) if seq[i]<= value]
    right_part = [seq[i] for i in range(1,n) if seq[i] > value]
    return quick_sort(left_part) + [value] + quick_sort(right_part)


def test_quick_sort():
    seq = [2,5,2,1,11,9,4,6]
    seq2 = quick_sort(seq)
    print(seq2)

if __name__ == "__main__":
    test_quick_sort()