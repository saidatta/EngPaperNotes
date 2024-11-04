#### Overview
- **Basic Concept**: Binary Search Trees (BSTs) support dynamic set operations like SEARCH, INSERT, DELETE, MINIMUM, MAXIMUM, PREDECESSOR, and SUCCESSOR.
- **Performance**: Operations' time complexity is proportional to the tree's height - Θ(lg n) for balanced trees, Θ(n) for skewed trees.
- **Red-Black Trees**: A variation of BSTs covered in Chapter 13 ensures O(lg n) height.
- **Expected Height**: For a randomly built BST, the expected height is O(lg n).

#### 12.1 What is a Binary Search Tree?
- **Structure**: Each node contains a key, satellite data, and pointers to its left child, right child, and parent.
- **Binary-Search-Tree Property**: For any node x, all keys in the left subtree are ≤ x.key, and all keys in the right subtree are ≥ x.key.
- **Inorder Tree Walk**: Recursively prints all keys in sorted order.

##### Binary Search Tree Operations
- **Inorder Tree Walk Algorithm**:
  ```python
  def inorder_tree_walk(x):
      if x != NIL:
          inorder_tree_walk(x.left)
          print(x.key)
          inorder_tree_walk(x.right)
  ```
- **Performance**: Walking an n-node BST takes Θ(n) time.

#### Theorem 12.1
- **Statement**: INORDER-TREE-WALK(x) takes Θ(n) time for a subtree with root x and n nodes.
- **Proof**: Based on the observation that each node is visited exactly twice (once for each child) and a small constant amount of time is spent per node.

---

### Code Example

#### Python Implementation of a Basic Binary Search Tree
```python
class TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = self.right = self.parent = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        node = TreeNode(key)
        y = None
        x = self.root
        while x:
            y = x
            if node.key < x.key:
                x = x.left
            else:
                x = x.right
        node.parent = y
        if y is None:  # Tree was empty
            self.root = node
        elif node.key < y.key:
            y.left = node
        else:
            y.right = node

    def inorder_walk(self, x):
        if x:
            self.inorder_walk(x.left)
            print(x.key)
            self.inorder_walk(x.right)
```