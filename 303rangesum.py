class NumArray:

    def __init__(self, nums):
        self.nums = nums;

    def sumRange(self, i, j) :
        if i == j:
            return (self.nums[i])
        temp = 0
        for index in range(i, j + 1):
            temp += self.nums[index]
        return temp


nums = [-2, 0, 3, -5, 2, -1]
i,j =0,2
obj = NumArray(nums)
param_1 = obj.sumRange(i,j)
print(param_1)


i,j =2, 5
param_1 = obj.sumRange(i,j)
print(param_1)

i,j =0, 5
param_1 = obj.sumRange(i,j)
print(param_1)
