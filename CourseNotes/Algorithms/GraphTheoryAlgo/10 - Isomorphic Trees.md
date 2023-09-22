
# Graph-Theory Algorithms - Isomorphic Trees/Graphs

## Introduction
- Isomorphism: Two graphs are isomorphic if they're structurally the same.
- An isomorphism exists between graphs if a function can map all the nodes and edges in one graph to the other and vice versa.
- Identifying isomorphic graphs is computationally complex.
- Specialized algorithms exist to identify isomorphisms in trees, which are simpler structures.

## Examples
![[Pasted image 20230706125925.png]]
   - Two trees, `Tree 1` and `Tree 2`. 
   - These trees are **not isomorphic** because they are structurally different.

2. ![[Pasted image 20230706130007.png]]
   - Two trees, `Tree 3` and `Tree 4`.
   - These trees **are isomorphic** because they have the same structure.

## Identifying Isomorphic Trees
- Fast, probabilistic algorithms exist for identifying isomorphic trees. These are often hash-based and can be error-prone due to hash collisions.
- Other deterministic methods involve serializing a tree into a unique encoding. If two trees have the same encoding, they are considered isomorphic.

### Encoding a Tree
- Convert an unrooted tree to a rooted tree.
- Choose the same root node for both trees before serialization to get identical encodings.
- Use the center of the tree as the root node.
- Convert the rooted tree into a sequence of left and right brackets or zeros and ones, which represents the tree's structure.

### AHU Algorithm
- A serialization technique that represents a tree as a unique string, capturing the tree's degree, spectrum, and structure.
- It assigns all leaf nodes a pair of left and right brackets.
- For nodes with outgoing children, it combines their children's labels and wraps them in a new pair of brackets.
- Child labels must be sorted to ensure uniqueness.

### Pseudocode
```python
def are_trees_isomorphic(tree1, tree2):
    center1 = find_center(tree1)
    root1 = root_tree(tree1, center1)
    encode1 = encode(root1)

    for center in find_center(tree2):
        root2 = root_tree(tree2, center)
        encode2 = encode(root2)
        if encode1 == encode2:
            return True
    return False

def encode(node):
    if node is None:
        return ''
    labels = []
    for child in node.children:
        labels.append(encode(child))
    labels.sort()
    return '(' + ''.join(labels) + ')'
```
- The `are_trees_isomorphic` function:
  - Finds the center of both trees.
  - Roots both trees at their centers.
  - Encodes the structure of both trees.
  - If there's a match in the encoded structure, the trees are isomorphic.

- The `encode` function:
  - Returns an empty string for a None node.
  - Iteratively encodes all child nodes.
  - Sorts and concatenates the encoded child nodes and wraps the result in brackets.

# Conclusion
- Determining isomorphism between graphs is complex.
- For trees, there exist efficient algorithms using serialization techniques.
- The AHU algorithm provides a unique encoding for a tree to help determine isomorphism. 

# Example
This example is based on Python language. You can implement it in any language you're comfortable with.
```python
class Node:
    def __init__(self):
        self.children = []

def are_trees_isomorphic(tree1, tree2):
    center1 = find_center(tree1)
    root1 = root_tree(tree1, center1)
    encode1 = encode(root1)

    for center

in find_center(tree2):
        root2 = root_tree(tree2, center)
        encode2 = encode(root2)
        if encode1 == encode2:
            return True
    return False

def encode(node):
    if node is None:
        return ''
    labels = []
    for child in node.children:
        labels.append(encode(child))
    labels.sort()
    return '(' + ''.join(labels) + ')'

def find_center(tree):
    # ... this function will find the center of the tree. Implementation depends on how the tree is represented.

def root_tree(tree, root):
    # ... this function will root the tree at the given node. Implementation depends on how the tree is represented.
```
With the above code, you can check if two trees are isomorphic. You need to add the `find_center` and `root_tree` function implementations. These will depend on how you choose to represent the tree. For instance, you could represent a tree using adjacency lists, adjacency matrices, or any other suitable data structure. 

In addition, the Node class is a simple representation for a node in the tree, which can be expanded upon based on your specific requirements. The Node class represents a node with multiple children, which fits the scenario of a general tree.

Remember that the 'center' of a tree in this context would refer to the middle node(s) of the tree, which could be found using various methods like removing leaves until one or two nodes are left.

Here's a simple example of how to use these functions:
```python
# assuming tree1 and tree2 are defined and populated
if are_trees_isomorphic(tree1, tree2):
    print("The trees are isomorphic.")
else:
    print("The trees are not isomorphic.")
```
This code would print "The trees are isomorphic." if the trees are indeed isomorphic according to the defined `are_trees_isomorphic` function, and "The trees are not isomorphic." otherwise.