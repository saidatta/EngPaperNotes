#### Introduction
- **Context:** Fibonacci heaps are often mentioned in the context of Dijkstra's algorithm for speeding up the process.
- **Purpose:** Efficient solution for a mergeable heap with an efficient decrease key operation.
- **Relation to Binomial Heaps:** Similar to binomial heaps but with less structural overhead, especially in the decrease key operation.
#### General Structure
- **Composition:** A set of trees with the min property (parent key ≤ child key).
- **Tree Characteristics:** Not necessarily complete or binomial, but must maintain the min property.
- **Node Attributes:** Each node stores information about its parent, leftmost child, left and right siblings, key, degree (number of children), and a mark (explained later).
- **Merging Trees:** Based on degree, similar to binomial trees but without the constraint of identical tree structures.
#### Setup for Analysis
- **Potential Method:** Used for amortized analysis with the potential defined as the number of trees in the heap plus twice the number of marked nodes.
- **Initial Potential:** Zero (no trees or marks initially).
- **Potential Function:** \( \Phi = \text{number of trees} + 2 \times \text{number of marked nodes} \).
- **Maximum Degree:** Analyzed with the assumption of a maximum degree of \( D(n) \), to be shown as \( O(\log n) \).
#### Basic Operations and Their Amortized Costs
- **Creating an Empty Heap:** \( O(1) \) actual cost, no change in potential.
- **Finding the Minimum:** \( O(1) \) actual cost, constant time operation.
- **Union Operation:** Concatenating lists of trees, \( O(1) \) actual cost, no change in potential (similar to lazy union in binomial heaps).
- **Insert Operation:** Creating a singleton heap and adding it to the list of trees, \( O(1) \) actual cost, increase in potential by 1 due to the additional tree.
#### Analysis of Insert Operations
- Performing \( n \) inserts results in a Fibonacci heap that is a list of \( n \) single nodes.
#### Delete Min Operation (To Be Explored Further)
- **Process:** Involves more complex steps compared to the operations mentioned above.
- **Expectation:** Likely to involve changes in both the actual cost and the potential, impacting the amortized cost.

---
#### Delete Minimum Operation
- **Similarity to Binomial Heaps:** The delete minimum operation in Fibonacci heaps is akin to the same operation in binomial heaps with lazy union.
- **Process:**
  - Delete the minimum root.
  - Create an array with an entry for each possible degree.
  - Insert trees into this array based on their degree.
  - When encountering a tree with an existing degree, link the two trees.
  - After processing, transform the array back into a heap.
- **Cost Analysis:** 
  - The actual cost is bound by `O(D(n) + T(h))`, where `D(n)` is the maximum degree and `T(h)` is the initial number of trees.
  - The potential changes are factored by the number of trees after processing, which is bounded by `D(n) + 1`.
  - Amortized cost can be calculated as the actual cost plus the change in potential.
#### Decrease Key Operation
- **Method:** 
  - If decreasing a key violates the heap property, cut the node out and make it a new tree in the heap.
  - Mark the parent node if it’s the first child being cut.
  - If a marked node loses a child, cut this node as well and cascade upwards.
- **Pseudocode:**
  - Decrease the key of node X.
  - If the new key violates the heap property, cut X and recursively call cascade cut on its parent.
  - Update the minimum pointer if necessary.
- **Analysis:** 
  - Actual cost includes a constant factor plus the number of cuts.
  - Change in potential depends on the number of new trees (cuts) and the change in the number of marked nodes.
  - Amortized cost is calculated as `O(1 + K)`, where K is the number of cuts.
#### Analysis Summary
- The key to Fibonacci heaps is the lazy approach to the decrease key operation, significantly reducing the amortized cost.
- The amortized running time for all operations is `O(1)`, except for delete minimum, which depends on the maximum degree `D(n)`.
- The delete minimum operation's efficiency is contingent on maintaining a reasonable maximum degree, which is generally `O(log n)`.

---
#### Overview
- **Purpose**: Efficient solution for mergeable heaps with efficient decrease key operation.
- **Structure**: A set of trees with the Min property (parent key ≤ children keys).
- **Stored Information**: Parent, leftmost child, left and right siblings, key, degree, and mark.

#### Bounding `D(n)`
- **Intuition**: Without cuts, it behaves like binomial heaps. With cuts, the degree changes by at most one. Aim to show size of a node `X` with degree `k` is at least `φ^k`, where `φ` is the golden ratio.
- **Cut Limitation**: Can cut at most one child per node.

#### Lemmas for Bounding `D(n)`
1. **Lemma 1**: For a node `X` of degree `k`, its `i-th` child has a degree of at least `i-2`.
2. **Lemma 2**: `k-th` Fibonacci number `F(k)` is at least the sum of Fibonacci numbers up to `k-1`.
3. **Lemma 3**: `F(k+2)` is at least `φ^k`.
4. **Lemma 4 (Crucial)**: The size of a subtree rooted at a node `X` of degree `k` is at least `F(k+2)`.

#### Proving Lemma 4
- Inductive proof based on the minimum size of a subtree for a given degree.
- Utilizes the properties from previous lemmas.
- Concludes that size of a node `X` with degree `k` is at least `F(k+2)`.

#### Formal Analysis for Bounding `D(n)`
- **Goal**: Prove `D(n)` is bounded by `log_φ(n)`.
- **Proof Approach**: Combine the lemmas to establish a relationship between the degree of a node and the size of its subtree.
- **Conclusion**: `D(n)` is `O(log n)`.

#### Operations in Fibonacci Heaps
1. **Make Heap**: `O(1)` - Create an empty heap.
2. **Find Min**: `O(1)` - Return the minimum key.
3. **Union**: `O(1)` - Concatenate two heap lists.
4. **Insert**: `O(1)` - Add a new element.
5. **Delete Min**: `O(D(n))` - Remove the minimum element, restructure the heap.

#### Decrease Key Operation
- **Method**: Decrease the key and cut the node if it violates the heap property. Mark the parent if it’s the first child cut. Cascade cut upwards if necessary.
- **Analysis**: Actual cost is `O(1 + number of cuts)`. Change in potential considers new trees and marks.

### Summary
- **Fibonacci Heaps**: Offer amortized `O(1)` for all operations except `Delete Min`, which is `O(log n)`.
- **Efficiency**: Achieves faster decrease key operation compared to binomial and binary heaps.
---
