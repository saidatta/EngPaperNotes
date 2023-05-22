Topics:
- Using numpy's and pytorch's seed functions
- Multiple seeds in Python and their scope

# Introduction
- Randomness is usually impossible to control or reproduce
- Computers allow us to reproduce the same set of random numbers using seeding
- Seeding enables reproducible randomness, which is crucial for sharing and reproducing deep learning models

# Numpy seeding
- Older method: `numpy.random.seed()`
  - Example:
    ```python
    import numpy as np
    np.random.seed(17)
    print(np.random.randn(5))
    ```
  - This method seeds everything globally

- Recommended method: `numpy.random.RandomState()`
  - Example:
    ```python
    import numpy as np
    rand_seed1 = np.random.RandomState(17)
    rand_seed2 = np.random.RandomState(20230510)
    print(rand_seed1.randn(5))
    print(rand_seed2.randn(5))
    ```
  - This method allows you to create local scope for the seed, which is more flexible

# PyTorch seeding
- Similar to the older numpy method
- Example:
  ```python
  import torch
  torch.manual_seed(17)
  print(torch.randn(5))
  ```
- PyTorch's manual seed is local to PyTorch and doesn't affect numpy's random number generator

# Conclusion
- Seeding allows for reproducible randomness in deep learning models
- Numpy and PyTorch have different seeding methods
- Most of the time, we want different random initializations for models to avoid getting stuck in local minima, but it's important to know how to make models reproducible when needed