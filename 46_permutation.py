# solution 1:
def permute(nums):

    results = []
    if (len(nums) == 0):
        return result

    temp = []

    def dfs(results, nums, temp):
        if len(temp) == len(nums):
            results.append(temp[:])
            print(f'results:{results}')
            return

        for i in range(len(nums)):
            print('foor loop')
            print(f'i:{i}')
            print(f'temp: {temp}')

            if nums[i] in temp:
                print('data hit')
                continue

            temp.append(nums[i])
            print('now is dfs recursive call')
            dfs(results, nums, temp)
            temp.pop()
            print(f'temp after pop:{temp}')

    dfs(results, nums, temp)

    return results

# solution 2
class Solution(object):
    def permute2(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        self.dfs(nums, res, [])
        return res

    def dfs(self, nums, res, temp):
        if not nums:
            res.append(temp)
            print(f'res:{res}')

        for i in range(len(nums)):
            print('for loop, recursive')
            self.dfs(nums[:i] + nums[i + 1:], res, temp + [nums[i]])


nums =[1,2,3]
# permute(nums)

s = Solution
s.permute(nums)