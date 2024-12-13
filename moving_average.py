# numpy is not used in the implementation, only for tests
import numpy as np


class WindowSizeException(Exception):
    pass


def ma(nums, k):
    """Returns a moving average array with window size k.
    This functions is safe for window sizes k > len(nums).
    In that case (and the initial filling phase) it uses the maximum window available,
    which corresponds to the cumulative moving average.
    The core logic could be adapted for use in a streaming API as well."""

    if k <= 0:
        raise WindowSizeException(f"Illegal window size {k}")

    if not nums:
        return []

    # preallocate the array
    result = [0 for _ in range(len(nums))]

    result[0] = nums[0]
    for i in range(1, k):
        # same idea here as below, except nothing leaves the window
        result[i] = (result[i - 1] * i + nums[i]) / (i + 1)
    for i in range(k, len(nums)):
        # idea: we retrieve the moving sum by multiplying previous result with k,
        # then subtract the number leaving the window and add the one entering
        # the window, before dividing everything by k. this simplifies to:
        result[i] = (result[i - 1] * k + nums[i] - nums[i - k]) / k

    return result


def tests():
    def property_test(runs=500):
        def ma_naive(nums, k):
            """reference implementation for random property testing"""
            if not nums:
                return []

            res = []

            for i in range(len(nums)):
                start, end = max(0, i - k + 1), i + 1
                window_sum = sum(nums[start:end])
                window_size = end - start
                res.append(window_sum / window_size)

            return res

        for _ in range(runs):
            ns, k = np.random.randint(-10, 10 + 1, 20).tolist(), np.random.randint(1, 5)
            res, shouldbe = ma(ns, k), ma_naive(ns, k)
            np.testing.assert_almost_equal(res, shouldbe)

    try:
        property_test()

        # unit tests
        ns, k = [1, 1, 1, 1, 1], 2
        res, shouldbe = ma(ns, k), [1.0, 1.0, 1.0, 1.0, 1.0]
        np.testing.assert_almost_equal(res, shouldbe)

        ns, k = [1, 2, 3, 4, 5], 1
        res, shouldbe = ma(ns, k), [1.0, 2.0, 3.0, 4.0, 5.0]
        np.testing.assert_almost_equal(res, shouldbe)

        ns, k = [1, 2, 3, 4, 5], 3
        res, shouldbe = ma(ns, k), [1.0, 1.5, 2.0, 3.0, 4.0]
        np.testing.assert_almost_equal(res, shouldbe)

        ns, k = [2, 4, 6, 8, 10], 2
        res, shouldbe = ma(ns, k), [2.0, 3.0, 5.0, 7.0, 9.0]
        np.testing.assert_almost_equal(res, shouldbe)

        ns, k = [1], 1
        res, shouldbe = ma(ns, k), [1.0]
        np.testing.assert_almost_equal(res, shouldbe)

        ns, k = [1, -1, 1, -1], 2
        res, shouldbe = ma(ns, k), [1.0, 0.0, 0.0, 0.0]
        np.testing.assert_almost_equal(res, shouldbe)

        ns, k = [], 1
        res, shouldbe = ma(ns, k), []
        np.testing.assert_almost_equal(res, shouldbe)
    except AssertionError:
        print(f"failed {ns}, {k}:")
        raise


if __name__ == "__main__":
    tests()
