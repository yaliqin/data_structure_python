# solution 1:

def combinationSum(self, candidates, target):
    res = []
    candidates.sort()
    self.dfs(candidates, target, 0, [], res)
    return res


def dfs(self, nums, target, index, path, res):
    if target < 0:
        return  # backtracking
    if target == 0:
        res.append(path)
        return
    for i in xrange(index, len(nums)):
        self.dfs(nums, target - nums[i], i, path + [nums[i]], res)


# solution 2

def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    results = []
    if (len(candidates) == 0):
        return results

    candidates.sort()
    temp = []

    def dfs(results, candidates, temp, target, index):
        if target == 0:
            results.append(temp[:])
            return

        for i in range(index, len(candidates)):
            if candidates[i] > target:
                break

            temp.append(candidates[i])

            dfs(results, candidates, temp, target - candidates[i], i)
            temp.pop()  # first add the element without caparation

    dfs(results, candidates, temp, target, 0)

    return results

    "