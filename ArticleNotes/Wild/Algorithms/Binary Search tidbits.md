## Overview
Binary search is a powerful algorithmic technique that can be used to efficiently find a specific value in a sorted array or list. It works by repeatedly dividing the search space in half until the desired value is found or it is determined that the value is not in the array.

binary search can be used to solve any problem where there is a monotonic relationship within the set of numbers being searched.

A monotonic relationship is one where the values in a set either consistently increase or consistently decrease. If such a relationship exists within the set of numbers being searched, then binary search can be used to efficiently find a specific value within that set.

Binary search is a powerful algorithm for finding a target value within a sorted array. It works by repeatedly dividing the search space in half until the target value is found or it is determined that the target value is not in the array.

## Basic Code (C++)

```cpp
int find(vector<int> &arr, int target) {
    int left = 0, right = arr.size() - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1;
}
```

This basic implementation of binary search has several corner cases to consider:

1. **Loop condition**: Should we use `left <= right` or `left < right`?
2. **Midpoint calculation**: There are several common ways to calculate the midpoint: `mid = (left+right)/2`, `mid = left+(right-left)/2`, and `mid = (left+right+1)/2`.
3. **Boundary movement**: Both `left` and `right` have two ways of moving. For `left`, we can use `left=mid+1` and `left=mid`.
4. **Final state**: If the target value is not found, `left` and `right` will point to the positions on either side of where the target value should be inserted, and `left > right`.

## Finding Range of a Number

To find the left and right boundary indices of a certain number in an ordered array, we can modify the binary search algorithm slightly:

```cpp
int find_range(vector<int> &arr, int target, bool left_range) {
    int left = 0, right = arr.size() - 1;
    while (left < right) {
        int mid = left_range ? (left + right) / 2 : (left + right + 1) / 2;
        if (arr[mid] == target) {
            left_range ? right = mid : left = mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return left;
}
```

This code finds the left and right boundaries of a target value in the array. If the target value does not exist in the array, additional checks are needed.

## Generalizing Binary Search

Binary search can be generalized to any problem where there is a monotonic relationship within the set of numbers being searched. This means that as long as the function formed by the value in the search range and the target value is monotonous, we can narrow down the range of the target value by comparing the search objects.

Key characteristics of binary search:

1. **Monotonic mapping**: The array value (search range) and its index (search target) present a monotonic mapping relationship.
2. **Halving**: Each iteration of binary search cuts the current data size in half, hence the name "half search".