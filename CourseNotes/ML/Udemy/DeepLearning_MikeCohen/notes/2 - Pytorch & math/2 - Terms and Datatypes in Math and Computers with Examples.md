Course: Deep Learning Understanding

Overview topics:
-   Linear algebra and data storage important terms
-   Types of numbers and variables

1.  Linear Algebra Terminology:
    
    -   Scalar: A single number Example: 5
        
    -   Vector: A column or a row of numbers Example: Column vector - [1, 2, 3]^T (Transpose) Row vector - [4, 5, 6]
        
    -   Matrix: A 2D spreadsheet of numbers Example: | 1 2 3 | | 4 5 6 | | 7 8 9 |
        
    -   Tensor: A higher-dimensional array of numbers Example: [[[ 1, 2, 3], [ 4, 5, 6], [ 7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]
        
2.  Data Storage in Images:
    
    -   Grayscale Image: Represented as a matrix Example: 3x3 grayscale image with pixel intensities between 0 and 255 | 255 128 0 | | 64 192 32 | | 0 128 255 |
        
    -   Color Image: Represented as a 3D tensor (with Red, Green, and Blue channels) Example: 2x2 color image represented as a 3D tensor Red Channel: | 255 128 | Green Channel: | 64 192 | | 64 192 | | 128 32 |
        
        yamlCopy code
        
               `Blue Channel:   |   0 128 |                         | 255 128 |`
        
3.  Data Types:
    
    -   In computer science: Refers to the format of data storage Example: Floating-point numbers: 3.14, 2.5, 0.001 Boolean: True, False Strings: "Hello, world!"
        
    -   In statistics: Refers to the category of data Example: Categorical: Male, Female Numerical: Age, Height Ratio: Prices, Temperature
        
4.  Different Data Types in Python:
    
    -   List: [1, 2, 3, 4, 5]
    -   Numpy array: np.array([1, 2, 3, 4, 5])
    -   PyTorch tensor: torch.tensor([1, 2, 3, 4, 5])
5.  Terminology in Numpy and PyTorch:
    
    -   Numpy: Vectors are called "arrays" and matrices/tensors are called "ndarrays" (n-dimensional arrays) Example: Vector - np.array([1, 2, 3]) Matrix - np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
    -   PyTorch: All objects are called "tensors" 
	    - Scalar - `torch.tensor(5)`
	    - Vector - `torch.tensor([1, 2, 3])`
	    - Matrix - `torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])`
	    - Tensor - 
		    - `torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])`

In this summary, you have examples for important terminology in linear algebra, data storage in images, data types in computer science and statistics, different data types in Python, and terminology in Numpy and PyTorch. These examples will help you understand and visualize the concepts when working with deep learning algorithms and frameworks like PyTorch.