### Introduction

-   The dot product (also known as scalar product or inner product) is a fundamental operation in applied mathematics.
-   It is an essential mathematical operation in deep learning.
-   The dot product is straightforward to compute.
-   Common notations: A · B, <A, B>, A<sup>T</sup>B
### Definition
-   The dot product of two vectors A and B is defined as the sum of the products of their corresponding components.
-   Mathematically, A · B = Σ(a_i * b_i) for i = 1 to n, where n is the number of elements in the vectors.
![[Screenshot 2024-10-01 at 9.10.58 AM.png]]
### Example
-   Let A = [1, 0, 2, 5, -2] and B = [2, 8, -6, 1, 0]
-   A · B = (1 * 2) + (0 * 8) + (2 * -6) + (5 * 1) + (-2 * 0) = 2 - 12 + 5 = -5
-   The dot product is a single number.
-   It is only defined for vectors with the same number of elements.
![[Screenshot 2024-10-01 at 9.11.21 AM.png]]
### Dot Product for Matrices
-   The dot product can also be computed for matrices.
-   The matrices must have the same dimensions.
-   The dot product is still a single number.
-   Example: let A and B be 3x3 matrices
    -   A · B = Σ(A_ij * B_ij) for i = 1 to 3 and j = 1 to 3
    - ![[Screenshot 2024-10-01 at 9.13.49 AM.png]]
### Implementing Dot Product in Python

-   Use NumPy and PyTorch for implementing the dot product
-   Example using NumPy:

`import numpy as np A = np.array([1, 0, 2]) B = np.array([2, 8, -6]) dot_product = np.dot(A, B) print(dot_product)`

-   Example using PyTorch:

`import torch A = torch.tensor([1, 0, 2]) B = torch.tensor([2, 8, -6]) dot_product = torch.dot(A, B) print(dot_product)`

### Interpretation of Dot Product
-   The dot product is a single number reflecting the commonalities between two mathematical objects.
-   It can represent correlation or covariance coefficients, convolutions, matrix multiplications, and more.
-   The dot product is used to compute similarity between two vectors, matrices, tensors, signals, or images.
![[Screenshot 2024-10-01 at 9.15.35 AM.png]]
### Key Takeaways
-   The dot product is an essential operation in deep learning.
-   It is the element-wise multiplication and sum between two vectors or matrices.
-   The dot product is always a single number, reflecting similarity between the two inputs.
-   It is only defined for vectors with the same number of elements or matrices with the same shape.