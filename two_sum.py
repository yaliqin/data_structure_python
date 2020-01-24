class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

    def append(self, value):
        new_node = ListNode(value)
        # if this is the first node
        print(self.val)
        self.next = new_node


class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        ind = 0
        l = ListNode(None)
        a = l
        while l1 != None and l2 != None:
            sum_val = (l1.val + l2.val + ind) % 10
            ind = int((l1.val + l2.val + ind) / 10)
            l1 = l1.next
            l2 = l2.next
            l.next = ListNode(sum_val)
            l = l.next
        while l1 != None:
            sum_val = (l1.val + ind) % 10
            ind = int((l1.val + ind) / 10)
            l.next = ListNode(sum_val)
            l1 = l1.next
            l = l.next
        while l2 != None:
            sum_val = (l2.val + ind) % 10
            ind = int((l2.val + ind) / 10)
            l.next = ListNode(sum_val)
            l2 = l2.next
            l = l.next
        if ind > 0:
            l.next = ListNode(ind)

        return a.next


l1 = ListNode(2)
l1.append(4)
l1.next.append(3)

l2 = ListNode(5)
l2.append(6)
l2.next.append(7)
s = Solution()

a = ListNode(None)
a = s.addTwoNumbers(l1,l2)
print(a.value)