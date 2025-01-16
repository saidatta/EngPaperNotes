https://halfrost.com/segment_tree/
A segment tree is a binary tree data structure, used for storing intervals or line segments. It was invented by Jon Louis Bentley in 1977. A segment tree allows for quick querying of all intervals containing a certain point.

Here is a brief look at the segment tree characteristics:
- The space complexity of a segment tree that stores 'n' intervals is O(n).
- The time complexity of querying is O(log n + k), where 'k' is the number of intervals that meet the condition.
- Segment tree data structure can be generalized to higher dimensions.

## 1. What is a segment tree
Take a one-dimensional segment tree as an example.

![](https://img.halfrost.com/Blog/ArticleImage/153_1.png)

Let S be a set of one-dimensional line segments. Sort the endpoint coordinates of these line segments from small to large, let it bex1,x2,⋯,xm�1,�2,⋯,��. We call each interval divided by these endpoints a "unit interval" (the position of each endpoint will become a unit interval), which includes from left to right:

(−∞,x1),[x1,x1],(x1,x2),[x2,x2],...,(xm−1,xm),[xm,xm],(xm,+∞)(−∞,�1),[�1,�1],(�1,�2),[�2,�2],...,(��−1,��),[��,��],(��,+∞)

The structure of the line segment tree is a binary tree, each node represents a coordinate interval, and the interval represented by node N is recorded as Int(N), then it must meet the following conditions:

-   Each of its leaf nodes represents each unit interval from left to right.
-   The interval represented by its internal node is the union of the intervals represented by its two children.
-   Within each node (including leaves) there is a data structure that stores line segments. If the coordinate interval of a line segment S contains Int(N) but not Int(parent(N)), then the line segment S will be stored in node N.

![](https://img.halfrost.com/Blog/ArticleImage/153_2.png)

Segment trees are binary trees in which each node represents an interval. Typically, a node will store data for one or more merged intervals so that query operations can be performed.

## 2. Why do we need this data structure

Many problems require us to give results based on queries over the available data ranges or intervals. This can be a tedious and slow process, especially if the queries are numerous and repetitive. Segment trees allow us to efficiently process such queries in logarithmic time complexity.

Segment trees can be used [in the fields of computational geometry and geographic information systems](https://en.wikipedia.org/wiki/Geographic_information_systems) . For example, there may be a large number of points in space at some distance from a central reference point/origin. Suppose we want to find points within a certain distance from the origin. A normal lookup table would require a linear scan (assuming a hashmap) of all possible points or all possible distances. Segment trees allow us to do this in logarithmic time while requiring much less space. Such a problem is called [a flat range search](https://en.wikipedia.org/wiki/Range_searching) . Efficiently addressing such issues is critical, especially when dealing with dynamic data that changes rapidly (e.g., radar systems for air traffic). The following will take the line segment tree to solve the Range Sum Query problem as an example.

![](https://img.halfrost.com/Blog/ArticleImage/153_3.png)

The figure above is the line segment tree used as a range query.

## 3. Construct line segment tree

Suppose the data exists in arr[] of size n.

1.  The root of the segment tree usually represents the entire data interval. Here is arr[0:n-1].
2.  Each leaf of the tree represents a range containing exactly one element. Thus, leaves represent arr[0], arr[1] and so on, up to arr[n-1].
3.  The internal nodes of the tree will represent the merged or unioned results of their children.
4.  Each child node can represent approximately half of the range represented by its parent node. (two points of thought)

Use a size of≈4∗n≈4∗� An array of can easily represent a segment tree of n-element ranges. ( [Stack Overflow](http://stackoverflow.com/q/28470692/2844164) has a good discussion of why. Don't worry if you're not sure yet. More on that later in this article.)

The node with subscript i has two nodes, and the subscripts are(2∗i+1)(2∗�+1)and(2∗i+2)(2∗�+2) 。

![](https://img.halfrost.com/Blog/ArticleImage/153_4.png)

Segment trees seem intuitive and are well suited for recursive construction.

We will use the array tree[] to store the nodes of the segment tree (initialized to all zeros). Subscripts start at 0.

-   The node of the tree is at index 0. So tree[0] is the root of the tree.
-   The children of tree[i] are stored in tree[2 * i + 1] and tree[2 * i + 2].
-   fills arr[] with additional 0 or null values ​​such thatn=2k�=2�(where n is the total length of arr[] and k is a non-negative integer.)
-   The subscript value range of leaf nodes is∈[2k−1,2k+1−2]∈[2�−1,2�+1−2]

![](https://img.halfrost.com/Blog/ArticleImage/153_5.png)

The code to construct the line segment tree is as follows:

Go

```go
// SegmentTree define
type SegmentTree struct {
	data, tree, lazy []int
	left, right      int
	merge            func(i, j int) int
}

// Init define
func (st *SegmentTree) Init(nums []int, oper func(i, j int) int) {
	st.merge = oper
	data, tree, lazy := make([]int, len(nums)), make([]int, 4*len(nums)), make([]int, 4*len(nums))
	for i := 0; i < len(nums); i++ {
		data[i] = nums[i]
	}
	st.data, st.tree, st.lazy = data, tree, lazy
	if len(nums) > 0 {
		st.buildSegmentTree(0, 0, len(nums)-1)
	}
}

// Create a segment tree of [left....right] at the position of treeIndex
func (st *SegmentTree) buildSegmentTree(treeIndex, left, right int) {
	if left == right {
		st.tree[treeIndex] = st.data[left]
		return
	}
	midTreeIndex, leftTreeIndex, rightTreeIndex := left+(right-left)>>1, st.leftChild(treeIndex), st.rightChild(treeIndex)
	st.buildSegmentTree(leftTreeIndex, left, midTreeIndex)
	st.buildSegmentTree(rightTreeIndex, midTreeIndex+1, right)
	st.tree[treeIndex] = st.merge(st.tree[leftTreeIndex], st.tree[rightTreeIndex])
}

func (st *SegmentTree) leftChild(index int) int {
	return 2*index + 1
}

func (st *SegmentTree) rightChild(index int) int {
	return 2*index + 2
}
```

The author turns the operation of line segment tree merging into a function. The merge operation changes according to the meaning of the question. The common ones are addition, taking max, min and so on.

We take arr[] = [18, 17, 13, 19, 15, 11, 20, 12, 33, 25 ] as an example to construct a segment tree:

![](https://img.halfrost.com/Blog/ArticleImage/153_6.png)

After the line segment tree is constructed, the data in the array is:

```c
tree[] = [ 183, 82, 101, 48, 34, 43, 58, 35, 13, 19, 15, 31, 12, 33, 25, 18, 17, 0, 0, 0, 0, 0, 0, 11, 20, 0, 0, 0, 0, 0, 0 ]
```

The segment tree is filled with 0s to 4*n elements.

> LeetCode corresponding topics are [218. The Skyline Problem](https://books.halfrost.com/leetcode/ChapterFour/0200~0299/0218.The-Skyline-Problem/) , [303. Range Sum Query - Immutable](https://books.halfrost.com/leetcode/ChapterFour/0300~0399/0303.Range-Sum-Query-Immutable/) , [307. Range Sum Query - Mutable](https://books.halfrost.com/leetcode/ChapterFour/0300~0399/0307.Range-Sum-Query-Mutable/) , [699. Falling Squares](https://books.halfrost.com/leetcode/ChapterFour/0600~0699/0699.Falling-Squares/)

## 4. Line segment tree query

There are two query methods for the line segment tree, one is direct query, and the other is lazy query.

### 1. Direct inquiry

The method returns the result when the query scope exactly matches the scope represented by the current node. Otherwise, it traverses the segment tree deeper to find a node that exactly matches part of the node.

```go
// Query the value in [left....right] range

// Query define
func (st *SegmentTree) Query(left, right int) int {
	if len(st.data) > 0 {
		return st.queryInTree(0, 0, len(st.data)-1, left, right)
	}
	return 0
}

// In the range of [left...right] in the line segment tree rooted at 
// treeIndex, search for the value of the interval [queryLeft...queryRight]
func (st *SegmentTree) queryInTree(treeIndex, left, right, queryLeft, queryRight int) int {
	if left == queryLeft && right == queryRight {
		return st.tree[treeIndex]
	}
	midTreeIndex, leftTreeIndex, rightTreeIndex := left+(right-left)>>1, st.leftChild(treeIndex), st.rightChild(treeIndex)
	if queryLeft > midTreeIndex {
		return st.queryInTree(rightTreeIndex, midTreeIndex+1, right, queryLeft, queryRight)
	} else if queryRight <= midTreeIndex {
		return st.queryInTree(leftTreeIndex, left, midTreeIndex, queryLeft, queryRight)
	}
	return st.merge(st.queryInTree(leftTreeIndex, left, midTreeIndex, queryLeft, midTreeIndex),
		st.queryInTree(rightTreeIndex, midTreeIndex+1, right, midTreeIndex+1, queryRight))
}
```

![](https://img.halfrost.com/Blog/ArticleImage/153_7.png)

In the above example, the query ranges from the sum of elements in [2, 8]. No line segment can fully represent the [2, 8] range. But it can be observed that the range [2, 2], [3, 4], [5, 7], [8, 8] can be used to form [8, 8]. Quickly verify that the sum of the input elements at [2,8] is 13 + 19 + 15 + 11 + 20 + 12 + 33 = 123. The sum of nodes for [2, 2], [3, 4], [5, 7] and [8, 8] is 13 + 34 + 43 + 33 = 123. The answer is correct.

### 2. Lazy query

Lazy query corresponds to lazy update, and the two are supporting operations. When the interval is updated, all the nodes in the interval are not directly updated, but the value of the increase or decrease of nodes in the interval is stored in the lazy array. Wait until the next query to apply the increase or decrease to the specific node. This is also done to amortize the time complexity, to ensure that the time complexity of query and update is at the O(log n) level, and will not degenerate to the O(n) level.

Steps to lazy query nodes:

1.  First judge whether the current node is a lazy node. It is judged by querying whether lazy[i] is 0. If it is a lazy node, apply its increase or decrease to this node. And update its child nodes. This step is exactly the same as the first step of the update operation.
2.  Recursively query child nodes to find a suitable query node.

```go
// 查询 [left....right] 区间内的值

// QueryLazy define
func (st *SegmentTree) QueryLazy(left, right int) int {
	if len(st.data) > 0 {
		return st.queryLazyInTree(0, 0, len(st.data)-1, left, right)
	}
	return 0
}

func (st *SegmentTree) queryLazyInTree(treeIndex, left, right, queryLeft, queryRight int) int {
	midTreeIndex, leftTreeIndex, rightTreeIndex := left+(right-left)>>1, st.leftChild(treeIndex), st.rightChild(treeIndex)
	if left > queryRight || right < queryLeft { // segment completely outside range
		return 0 // represents a null node
	}
	if st.lazy[treeIndex] != 0 { // this node is lazy
		for i := 0; i < right-left+1; i++ {
			st.tree[treeIndex] = st.merge(st.tree[treeIndex], st.lazy[treeIndex])
			// st.tree[treeIndex] += (right - left + 1) * st.lazy[treeIndex] // normalize current node by removing lazinesss
		}
		if left != right { // update lazy[] for children nodes
			st.lazy[leftTreeIndex] = st.merge(st.lazy[leftTreeIndex], st.lazy[treeIndex])
			st.lazy[rightTreeIndex] = st.merge(st.lazy[rightTreeIndex], st.lazy[treeIndex])
			// st.lazy[leftTreeIndex] += st.lazy[treeIndex]
			// st.lazy[rightTreeIndex] += st.lazy[treeIndex]
		}
		st.lazy[treeIndex] = 0 // current node processed. No longer lazy
	}
	if queryLeft <= left && queryRight >= right { // segment completely inside range
		return st.tree[treeIndex]
	}
	if queryLeft > midTreeIndex {
		return st.queryLazyInTree(rightTreeIndex, midTreeIndex+1, right, queryLeft, queryRight)
	} else if queryRight <= midTreeIndex {
		return st.queryLazyInTree(leftTreeIndex, left, midTreeIndex, queryLeft, queryRight)
	}
	// merge query results
	return st.merge(st.queryLazyInTree(leftTreeIndex, left, midTreeIndex, queryLeft, midTreeIndex),
		st.queryLazyInTree(rightTreeIndex, midTreeIndex+1, right, midTreeIndex+1, queryRight))
}
```

## 5. Update the line segment tree

### 1. Single point update

A single point of update is similar `buildSegTree`. Update the value of the leaf node of the tree corresponding to the updated element. These updated values ​​propagate their influence to the root through the upper nodes of the tree.

Go

```go
// 更新 index 位置的值

// Update define
func (st *SegmentTree) Update(index, val int) {
	if len(st.data) > 0 {
		st.updateInTree(0, 0, len(st.data)-1, index, val)
	}
}

// 以 treeIndex 为根，更新 index 位置上的值为 val
func (st *SegmentTree) updateInTree(treeIndex, left, right, index, val int) {
	if left == right {
		st.tree[treeIndex] = val
		return
	}
	midTreeIndex, leftTreeIndex, rightTreeIndex := left+(right-left)>>1, st.leftChild(treeIndex), st.rightChild(treeIndex)
	if index > midTreeIndex {
		st.updateInTree(rightTreeIndex, midTreeIndex+1, right, index, val)
	} else {
		st.updateInTree(leftTreeIndex, left, midTreeIndex, index, val)
	}
	st.tree[treeIndex] = st.merge(st.tree[leftTreeIndex], st.tree[rightTreeIndex])
}
```

![](https://img.halfrost.com/Blog/ArticleImage/153_8.png)

In this example, elements at index 1, 3, and 6 (in the original input data) are incremented by +3, -1, and +2, respectively. You can see how changes propagate down the tree, all the way to the root.

### 2. Interval update

The line segment tree only updates a single element, which is very efficient, and the time complexity is O(log n). But what if we want to update a range of elements? With the current approach, each element has to be updated independently, each taking some time. Updating each leaf node separately means processing their common ancestor multiple times. Ancestor nodes may be updated multiple times. What if you want to reduce this double counting?

![](https://img.halfrost.com/Blog/ArticleImage/153_11.png)

In the example above, the root node was updated three times, while node number 82 was updated twice. This is because updating a leaf node has an effect on the upper parent node. In the worst case, the query range does not contain frequently updated elements, so it takes a lot of time to update nodes that are rarely visited. Adding an additional lazy array can reduce unnecessary calculations and process nodes on demand.

Use another array lazy[], which is exactly the same size as our segment tree array tree[], representing a lazy node. When accessing or querying this node, lazy[i] holds the number of trees[i] that need to be increased or decreased for this node. When lazy[i] is 0, it means that the tree[i] node is not lazy, and there is no cached update.

Steps to update the nodes in the interval:

1.  First judge whether the current node is a lazy node. It is judged by querying whether lazy[i] is 0. If it is a lazy node, apply its increase or decrease to this node. And update its child nodes.
2.  If the interval represented by the current node is within the update range, apply the current update operation to the current node.
3.  Recursively update child nodes.

The specific code is as follows:

Go

```go

// 更新 [updateLeft....updateRight] 位置的值
// 注意这里的更新值是在原来值的基础上增加或者减少，而不是把这个区间内的值都赋值为 x，区间更新和单点更新不同
// 这里的区间更新关注的是变化，单点更新关注的是定值
// 当然区间更新也可以都更新成定值，如果只区间更新成定值，那么 lazy 更新策略需要变化，merge 策略也需要变化，这里暂不详细讨论

// UpdateLazy define
func (st *SegmentTree) UpdateLazy(updateLeft, updateRight, val int) {
	if len(st.data) > 0 {
		st.updateLazyInTree(0, 0, len(st.data)-1, updateLeft, updateRight, val)
	}
}

func (st *SegmentTree) updateLazyInTree(treeIndex, left, right, updateLeft, updateRight, val int) {
	midTreeIndex, leftTreeIndex, rightTreeIndex := left+(right-left)>>1, st.leftChild(treeIndex), st.rightChild(treeIndex)
	if st.lazy[treeIndex] != 0 { // this node is lazy
		for i := 0; i < right-left+1; i++ {
			st.tree[treeIndex] = st.merge(st.tree[treeIndex], st.lazy[treeIndex])
			//st.tree[treeIndex] += (right - left + 1) * st.lazy[treeIndex] // normalize current node by removing laziness
		}
		if left != right { // update lazy[] for children nodes
			st.lazy[leftTreeIndex] = st.merge(st.lazy[leftTreeIndex], st.lazy[treeIndex])
			st.lazy[rightTreeIndex] = st.merge(st.lazy[rightTreeIndex], st.lazy[treeIndex])
			// st.lazy[leftTreeIndex] += st.lazy[treeIndex]
			// st.lazy[rightTreeIndex] += st.lazy[treeIndex]
		}
		st.lazy[treeIndex] = 0 // current node processed. No longer lazy
	}

	if left > right || left > updateRight || right < updateLeft {
		return // out of range. escape.
	}

	if updateLeft <= left && right <= updateRight { // segment is fully within update range
		for i := 0; i < right-left+1; i++ {
			st.tree[treeIndex] = st.merge(st.tree[treeIndex], val)
			//st.tree[treeIndex] += (right - left + 1) * val // update segment
		}
		if left != right { // update lazy[] for children
			st.lazy[leftTreeIndex] = st.merge(st.lazy[leftTreeIndex], val)
			st.lazy[rightTreeIndex] = st.merge(st.lazy[rightTreeIndex], val)
			// st.lazy[leftTreeIndex] += val
			// st.lazy[rightTreeIndex] += val
		}
		return
	}
	st.updateLazyInTree(leftTreeIndex, left, midTreeIndex, updateLeft, updateRight, val)
	st.updateLazyInTree(rightTreeIndex, midTreeIndex+1, right, updateLeft, updateRight, val)
	// merge updates
	st.tree[treeIndex] = st.merge(st.tree[leftTreeIndex], st.tree[rightTreeIndex])
}

```

> LeetCode corresponds to [218. The Skyline Problem](https://books.halfrost.com/leetcode/ChapterFour/0200~0299/0218.The-Skyline-Problem/) , [699. Falling Squares](https://books.halfrost.com/leetcode/ChapterFour/0600~0699/0699.Falling-Squares/)

## 6. Time complexity analysis

Let's take a look at the build process. We visit each leaf of the segment tree (corresponding to each element in the array arr[]). Therefore, we process approximately 2*n nodes. This makes the build process time complexity O(n). For each recursive update process, half of the interval range is discarded to reach the leaf nodes in the tree. This is similar to binary search, only takes logarithmic time. When a leaf is updated, the immediate ancestors at each level of the tree are updated. This takes time linear in the height of the tree.

![](https://img.halfrost.com/Blog/ArticleImage/153_9.png)

4*n nodes can ensure that the line segment tree is constructed as a complete binary tree, so that the height of the tree is log(4*n + 1) rounded up. The time complexity of reading and updating the segment tree is O(log n).

## 7. Frequently asked questions

### 1. Range Sum Queries

![](https://img.halfrost.com/Blog/ArticleImage/153_10.png)

Range Sum Queries are a subset of [Range Queries questions.](https://en.wikipedia.org/wiki/Range_query_(data_structures)) Given an array or sequence of data elements, read and update queries consisting of ranges of elements need to be processed. Segment Tree and Binary Indexed Tree (aka Fenwick Tree)) can quickly solve this kind of problem.

The Range Sum Query question deals specifically with the sum of elements within a query range. There are many variants of this problem, including [immutable data](https://leetcode.com/problems/range-sum-query-immutable/) , [mutable data](https://leetcode.com/problems/range-sum-query-mutable/) , [multiple updates, single query](https://leetcode.com/problems/range-addition/) and [many](https://leetcode.com/problems/range-sum-query-2d-mutable/) updates, multiple queries .

### 2. Single point update

-   [HDU 1166 Enemy formation](http://acm.hdu.edu.cn/showproblem.php?pid=1166) update: single-point increase and decrease query: interval summation
-   [HDU 1754 I Hate It](http://acm.hdu.edu.cn/showproblem.php?pid=1754) update: single-point replacement query: maximum value in interval
-   [HDU 1394 Minimum Inversion Number](http://acm.hdu.edu.cn/showproblem.php?pid=1394) update: single-point increase and decrease query: interval summation
-   [HDU 2795 Billboard](http://acm.hdu.edu.cn/showproblem.php?pid=2795) query: find the maximum position in the interval (directly perform the update operation in the query)

### 3. Interval update

-   [HDU 1698 Just a Hook](http://acm.hdu.edu.cn/showproblem.php?pid=1698) update: segment replacement (since the total interval is only queried once, so the information of 1 node can be directly output)
-   [POJ 3468 A Simple Problem with Integers](http://poj.org/problem?id=3468) update: Segment increase and decrease query: Interval summation
-   [POJ 2528 Mayor's posters](http://poj.org/problem?id=2528) discretization + update: segment replacement query: simple hash
-   [POJ 3225 Help with Intervals](http://poj.org/problem?id=3225) update: segment replacement, interval XOR query: simple hash

### 4. Range merge

This type of question will ask the longest continuous interval that meets the conditions in the interval, so when PushUp needs to merge the intervals of the left and right sons

-   [POJ 3667 Hotel](http://poj.org/problem?id=3667) update: interval replacement query: query the leftmost endpoint that meets the conditions

### 5. Scan line

This kind of problem needs to sort some operations, and then scan it with a scanning line from left to right. The most typical problem is the rectangle area union, perimeter length union, etc.

-   [HDU 1542 Atlantis](http://acm.hdu.edu.cn/showproblem.php?pid=1542) update: interval increase and decrease query: directly take the value of the root node
-   [HDU 1828 Picture](http://acm.hdu.edu.cn/showproblem.php?pid=1828) update: interval increase and decrease query: directly take the value of the root node

### 6. Counting problems

There is another class of problems in LeetCode that involves counting. [315. Count of Smaller Numbers After Self](https://books.halfrost.com/leetcode/ChapterFour/0300~0399/0315.Count-of-Smaller-Numbers-After-Self/) , [327. Count of Range Sum](https://books.halfrost.com/leetcode/ChapterFour/0300~0399/0327.Count-of-Range-Sum/) , [493. Reverse Pairs](https://books.halfrost.com/leetcode/ChapterFour/0400~0499/0493.Reverse-Pairs/) Such problems can be solved with the following routine. Each node of the line segment tree stores the interval count.

Go

```go
// SegmentCountTree define
type SegmentCountTree struct {
	data, tree  []int
	left, right int
	merge       func(i, j int) int
}

// Init define
func (st *SegmentCountTree) Init(nums []int, oper func(i, j int) int) {
	st.merge = oper

	data, tree := make([]int, len(nums)), make([]int, 4*len(nums))
	for i := 0; i < len(nums); i++ {
		data[i] = nums[i]
	}
	st.data, st.tree = data, tree
}

// 在 treeIndex 的位置创建 [left....right] 区间的线段树
func (st *SegmentCountTree) buildSegmentTree(treeIndex, left, right int) {
	if left == right {
		st.tree[treeIndex] = st.data[left]
		return
	}
	midTreeIndex, leftTreeIndex, rightTreeIndex := left+(right-left)>>1, st.leftChild(treeIndex), st.rightChild(treeIndex)
	st.buildSegmentTree(leftTreeIndex, left, midTreeIndex)
	st.buildSegmentTree(rightTreeIndex, midTreeIndex+1, right)
	st.tree[treeIndex] = st.merge(st.tree[leftTreeIndex], st.tree[rightTreeIndex])
}

func (st *SegmentCountTree) leftChild(index int) int {
	return 2*index + 1
}

func (st *SegmentCountTree) rightChild(index int) int {
	return 2*index + 2
}

// 查询 [left....right] 区间内的值

// Query define
func (st *SegmentCountTree) Query(left, right int) int {
	if len(st.data) > 0 {
		return st.queryInTree(0, 0, len(st.data)-1, left, right)
	}
	return 0
}

// 在以 treeIndex 为根的线段树中 [left...right] 的范围里，搜索区间 [queryLeft...queryRight] 的值，值是计数值
func (st *SegmentCountTree) queryInTree(treeIndex, left, right, queryLeft, queryRight int) int {
	if queryRight < st.data[left] || queryLeft > st.data[right] {
		return 0
	}
	if queryLeft <= st.data[left] && queryRight >= st.data[right] || left == right {
		return st.tree[treeIndex]
	}
	midTreeIndex, leftTreeIndex, rightTreeIndex := left+(right-left)>>1, st.leftChild(treeIndex), st.rightChild(treeIndex)
	return st.queryInTree(rightTreeIndex, midTreeIndex+1, right, queryLeft, queryRight) +
		st.queryInTree(leftTreeIndex, left, midTreeIndex, queryLeft, queryRight)
}

// 更新计数

// UpdateCount define
func (st *SegmentCountTree) UpdateCount(val int) {
	if len(st.data) > 0 {
		st.updateCountInTree(0, 0, len(st.data)-1, val)
	}
}

// 以 treeIndex 为根，更新 [left...right] 区间内的计数
func (st *SegmentCountTree) updateCountInTree(treeIndex, left, right, val int) {
	if val >= st.data[left] && val <= st.data[right] {
		st.tree[treeIndex]++
		if left == right {
			return
		}
		midTreeIndex, leftTreeIndex, rightTreeIndex := left+(right-left)>>1, st.leftChild(treeIndex), st.rightChild(treeIndex)
		st.updateCountInTree(rightTreeIndex, midTreeIndex+1, right, val)
		st.updateCountInTree(leftTreeIndex, left, midTreeIndex, val)
	}
}
```

-----

- Merging in Segment Trees?
In the context of segment trees, the merge operation refers to combining or reconciling information from child nodes to form their parent node's value. The specific operation performed depends on the problem you're trying to solve with the segment tree. 

Segment trees are commonly used to answer range queries in an array. A range query might be something like "what is the sum of the elements from index 2 to index 5?" or "what is the minimum element from index 3 to index 7?". The segment tree stores partial results for these queries in a binary tree structure, which makes answering these queries more efficient than scanning through the array.

To create the segment tree, we split the original array into segments, each represented by a node in the tree. In a leaf node, we store the value from the original array. In a non-leaf node, we store the combined value from its two children. This combined value is calculated using a merge operation.

The merge operation can be different depending on the problem:

1. **Sum**: If the segment tree is built for range sum queries, the merge operation is addition. A parent node is the sum of its two children.

2. **Min**: If the segment tree is built for range minimum queries, the merge operation finds the minimum. A parent node is the minimum of its two children.

3. **Max**: If the segment tree is built for range maximum queries, the merge operation finds the maximum. A parent node is the maximum of its two children.

In code, the merge operation might look something like this:

```go
func (st *SegmentTree) merge(left, right int) int {
	// this is a merge operation for a segment tree built for range sum queries
	return left + right
}
```

Or for a segment tree built for range min queries:

```go
func (st *SegmentTree) merge(left, right int) int {
	if left < right {
		return left
	} else {
		return right
	}
}
```

This approach makes segment trees extremely versatile and applicable to a wide range of problems.