In this lecture, we discuss the transpose operation, an essential linear algebra concept used in deep learning.
## Transpose Operation
Transposing means converting rows into columns and columns into rows.
![[Screenshot 2024-10-01 at 9.08.34 AM.png]]
### Column Vector Transpose
A column vector transposed becomes a row vector. The transpose operation is often indicated using a capital T in the superscript.

`Original column vector: [1] [2] [3]  Transposed row vector: [1, 2, 3]`

### Matrix Transpose
![[Screenshot 2024-10-01 at 9.09.09 AM.png]]
For a matrix, transposing involves preserving the column order into the rows. The first column becomes the first row, the second column becomes the second row, and so on.

`Original matrix: [1, 2] [3, 4]  Transposed matrix: [1, 3] [2, 4]`

Double transposing (transposing a transposed matrix) returns the original matrix.

## Transpose in NumPy and PyTorch

### NumPy

Create a row vector and transpose it:

`import numpy as np  nv = np.array([[1, 2, 3]]) print(nv)  nv_t = nv.T print(nv_t)`

Create a matrix and transpose it:

`nm = np.array([[1, 2, 3], [4, 5, 6]]) print(nm)  nm_t = nm.T print(nm_t)`

### PyTorch

Create a row vector and transpose it:

`import torch  tv = torch.tensor([[1, 2, 3]], dtype=torch.float) print(tv)  tv_t = tv.T print(tv_t)`

Create a matrix and transpose it:

`tm = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float) print(tm)  tm_t = tm.T print(tm_t)`

Note that while the syntax for NumPy and PyTorch is similar, the resulting data types are different: NumPy uses `numpy.ndarray`, while PyTorch uses `torch.Tensor`.