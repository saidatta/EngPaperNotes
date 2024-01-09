#### 13 Overview
- **Purpose**: Red-black trees provide a balanced form of binary search trees ensuring O(lg n) time complexity for basic operations.
- **Key Features**: Incorporates an extra bit for each node's color (RED or BLACK) to maintain balance.
#### 13.1 Properties of Red-Black Trees
- **Node Structure**: Each node contains attributes - color, key, left, right, and parent.
- **Red-Black Properties**:
  1. Every node is either red or black.
  2. The root is black.
  3. Every leaf (NIL) is black.
  4. If a node is red, then both its children are black.
  5. All paths from a node to descendant leaves contain the same number of black nodes.
- **Use of Sentinel (T.nil)**: A single sentinel represents all NILs to simplify boundary conditions in code.
- **Black-Height (bh)**: The number of black nodes on any simple path from a node down to a leaf.
- **Height Bound**: Height h of a red-black tree with n internal nodes is at most 2 lg(n + 1).
#### Lemma 13.1
- **Statement**: The height of a red-black tree with n internal nodes is at most 2 lg(n + 1).
- **Proof**:
  - Subtrees rooted at any node x contain at least 2^bh(x) - 1 internal nodes.
  - At least half the nodes on any path from the root to a leaf must be black.
  - The height h satisfies n ≥ 2^(h/2) - 1, leading to h ≤ 2 lg(n + 1).
#### Dynamic-Set Operations
- **Search Operations**: SEARCH, MINIMUM, MAXIMUM, SUCCESSOR, PREDECESSOR run in O(lg n) time.
- **Insertion and Deletion**: Require special handling to maintain red-black properties, detailed in later sections.
---
#### Basic Structure of a Red-Black Tree Node in Python
```python
class RedBlackTreeNode:
    def __init__(self, key, color):
        self.key = key
        self.color = color
        self.left = self.right = self.parent = None

class RedBlackTree:
    def __init__(self):
        self.nil = RedBlackTreeNode(None, 'BLACK')
        self.root = self.nil
```
### Figures and Diagrams
- **Red-Black Tree Example**: Visualization of a red-black tree with internal nodes, sentinel nodes, and black-height annotations.
- **Representation Styles**: Comparison of different ways to depict red-black trees (with NILs, with sentinel, without leaves).
#### Key Takeaways for Software Engineers
1. **Understanding Red-Black Properties**: Crucial for implementing and maintaining red-black trees.
2. **Importance of Balancing**: Ensures that operations remain efficient (O(lg n)).
3. **Sentinel Usage**: Simplifies code, especially for boundary conditions.
4. **Height Analysis**: Understanding the proof of Lemma 13.1 is key to appreciating the efficiency of red-black trees.