class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        start_idx1 = 0
        # start_idx2 = 0
        end_idx1 = 0
        # end_idx2 = 0
        l1 = 0
        l2 = 0
        d = {}
        ls = len(s)
        for k in range(ls):
            if s[k] not in d:
                end_idx1 += 1
                d[s[k]] = k
            else:
                print(s[k])
                l1 = end_idx1 - start_idx1
                tmp_idx = d[s[k]] + 1
                if tmp_idx > start_idx1:
                    start_idx1 = tmp_idx

                d[s[k]] = k
                end_idx1 += 1

                if l1 > l2:
                    l2 = l1
        l1 = end_idx1 - start_idx1

        if l1 > l2:
            l2 = l1

        return l2

s = Solution()
strings = 'abcabcbb'
s.lengthOfLongestSubstring(strings)