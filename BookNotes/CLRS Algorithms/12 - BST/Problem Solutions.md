#### 12-1 Binary Search Trees with Equal Keys
- **Problem**: Handling insertion of items with identical keys in a binary search tree.
- **a. TREE-INSERT with Identical Keys**: 
  - Performance: O(n^2) for inserting n items with identical keys.
- **Strategies**:
  - **b. Alternating Insertion**: Alternate left and right insertion for identical keys. 
    - Performance: O(n) for inserting n identical keys.
  - **c. List at Nodes**: Keep a list of nodes with equal keys.
    - Performance: O(n) for inserting n identical keys.
  - **d. Random Insertion**: Randomly choose left or right for insertion.
    - Worst-Case Performance: O(n^2).
    - Expected Running Time: O(n log n).

#### 12-2 Radix Trees
- **Sorting Bit Strings with Radix Tree**:
  - Traverse the tree to output sorted strings.
  - Lexicographical ordering similar to dictionary.
  - Performance: Θ(n) for sorting n bit strings.

#### 12-3 Average Node Depth in a Randomly Built Binary Search Tree
- **a. Average Depth**: Prove that average depth in a randomly built binary search tree is O(log n).
- **b. Path Length**: Show that `P(T) = P(TL) + P(TR) + n - 1`.
- **c. Average Total Path Length**: Derive `P(n)`.
- **d. Rewrite P(n)**: Use a different expression for P(n).
- **e. Comparison with Randomized Quicksort**: Link the building of a binary search tree to randomized quicksort, concluding `P(n) = O(n log n)`.
- **f. Quicksort Implementation**: Describe an implementation of quicksort that mirrors binary search tree insertions.

#### 12-4 Number of Different Binary Trees
- **a. Formula for bn**: 
  - Base case: b0 = 1.
  - Recursive formula: bn = Σ bi * b(n-i-1) for i = 0 to n-1.
- **b. Generating Function**: 
  - B(x) = xB(x)^2 + 1, leading to a closed-form expression.
- **c. Catalan Numbers**: 
  - Derive bn as the nth Catalan number using Taylor expansion.
- **d. Asymptotic Estimate**: 
  - Provide an estimate for bn.

---

### Code Examples

#### Random Insertion in TREE-INSERT (Strategy d)
```python
import random

def tree_insert(T, z):
    y = None
    x = T.root
    while x is not None:
        y = x
        if z.key == x.key:
            x = random.choice([x.left, x.right])
        elif z.key < x.key:
            x = x.left
        else:
            x = x.right
    z.parent = y
    if y is None:
        T.root = z
    elif z.key < y.key:
        y.left = z
    else:
        y.right = z
```

#### Radix Tree Sorting Algorithm
```python
def radix_tree_sort(S):
    radix_tree = create_radix_tree(S)
    return inorder_traverse(radix_tree)
```

### Figures and Diagrams
- Illustrations showing different strategies for inserting nodes with equal keys in a binary search tree.
- Diagram of a radix tree storing and sorting bit strings.
- Binary tree structures demonstrating the number of different binary trees for various n.

---

These notes provide solutions to problems related to binary search trees, including handling identical keys during insertion, using radix trees for sorting, analyzing the depth of nodes in randomly built trees, and counting the number of different binary trees. The explanations, code examples, and theoretical analysis offer valuable insights for software engineers.