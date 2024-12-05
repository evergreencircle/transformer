def maximumIntersections(arr, N, m):
    pointCount = {}  # Create a dictionary to store point counts

    print(arr)
    # Traverse the array of segments
    for i in range(N):
        # Increment the count of the left endpoint of the segment
        pointCount[arr[i][0]] = pointCount.get(arr[i][0], 0) + 1
        # print(i, pointCount[arr[i][0]], pointCount)
        # Decrement the count of the right endpoint of the segment + 1
        pointCount[arr[i][1] + 1] = pointCount.get(arr[i][1] + 1, 0) - 1
        # print(i, pointCount[arr[i][1] + 1], pointCount)

    currSum = 0
    # Iterate through the sorted points and their counts

    sorted_res = sorted(pointCount.items())
    print(sorted_res)
    ans = [0] * m
    last_ans = 0
    counter = 1
    for point, val in sorted_res:
        print("iteration", point, counter)
        while counter < point - 1 and counter != m:
            counter += 1
            ans[counter - 1] = last_ans
            print(f"Counter: {counter}", ans)
        if point - 1 == m:
            break
        currSum += val
        ans[point - 1] = currSum
        print("end", ans)
        last_ans = currSum
    # for ind, val in enumerate(ans):

    return ans


# Driver Code
if __name__ == "__main__":
    # arr = [[2,4], [1,3], [2, 5]]
    arr = [[6, 10], [1, 10], [2, 2]]
    N = len(arr)
    m = 6
    m = 10
    result = maximumIntersections(arr, N, m)
    print(result)
