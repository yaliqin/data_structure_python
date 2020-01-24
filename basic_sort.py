def bubble_sort(seq):
    n = len(seq)
    for i in range(n-1):
        for j in range(n-1-i):
            if seq[j]>seq[j+1]:
                seq[j],seq[j+1]= seq[j+1],seq[j]


def select_sort(seq):
    n = len(seq)
    for i in range(n-1):
        min_index = i
        for j in range(i+1,n):
            if seq[j] < seq[min_index]:
                min_index = j
        if min_index != i:
            seq[i],seq[min_index] = seq[min_index], seq[i]

def test_select_sort():
    seq =[5,2,1,5,9,2,6]
    select_sort(seq)
    print(seq)

def insert_sort(seq):
    n = len(seq)
    for i in range(1,n):
        value = seq[i]
        index = i
        while index > 0:
            if value < seq[index-1]:
                seq[index] = seq[index-1]
                index -= 1
        seq[index]= value

def test_insert_sort():
    seq =[5,2,1,11,9,4,6]
    select_sort(seq)
    print(seq)


if __name__ == "__main__":
    test_insert_sort()