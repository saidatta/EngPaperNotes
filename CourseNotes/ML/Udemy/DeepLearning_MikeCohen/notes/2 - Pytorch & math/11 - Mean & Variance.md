## Mean
- The mean is a measure of central tendency that provides information about the concentration of data in a distribution.
- It is also known as the arithmetic mean or average.
- Formula: 
  - Mean (X̄) = (ΣX) / N, where ΣX represents the sum of all data values and N is the number of data values.
- The mean is commonly represented as X̄ or μ (mu).
- It is suitable for roughly normally distributed data.
- Example: 
  - Given the numbers [2, 4, 6, 8, 10], the mean can be calculated as follows:
    - ΣX = 2 + 4 + 6 + 8 + 10 = 30
    - N = 5
    - Mean (X̄) = 30 / 5 = 6

## Variance
- Variance is a measure of dispersion that quantifies the spread of data points around the mean.
- Formula: 
  - Variance (σ^2) = Σ((X - X̄)^2) / (N - 1), where X represents individual data values, X̄ is the mean, and N is the number of data values.
- Variance emphasizes larger values and has various mathematical properties.
- Variance is closely related to standard deviation, which is the square root of the variance.
- Variance is suitable for any distribution.
- Example: 
  - Consider the set of numbers [8, 0, 4, 1, 2].
  - Calculate the mean: X̄ = (8 + 0 + 4 + 1 + 2) / 5 = 3. 
  - Compute the variance:
    - ((8 - 3)^2 + (0 - 3)^2 + (4 - 3)^2 + (1 - 3)^2 + (2 - 3)^2) / (5 - 1) = 16/4 = 4.

## Mean Absolute Difference (MAD)
- An alternative measure of dispersion, calculated as the mean of the absolute differences between data values and the mean.
- MAD is denoted as MAD(X) or MAD(X̄).
- MAD is more robust to outliers and less affected by extreme values compared to variance.
- The formula is similar to variance, but instead of squaring, we take the absolute difference:
  - MAD = Σ|X - X̄| / N

## Standard Deviation
- Standard deviation is the square root of the variance and provides a measure of dispersion.
- Formula: Standard Deviation (σ) = √(Variance).
- Standard deviation is commonly represented as σ (sigma).
- It indicates the average distance of data points from the mean.
- The standard deviation is more interpretable and has the same unit as the data.

## Bias in Variance Calculation
- The formula for variance can be biased or unbiased, depending on whether we divide by N or N - 1.
- Dividing by N - 1 yields an unbiased estimate of the population variance, accounting for the degrees of freedom.
- The default behavior in Python's NumPy library is to use the biased variance (dividing by N).
- The unbiased variance (dividing by N - 1) is preferred for small datasets, but the difference becomes negligible for large datasets.
- The biased variance is denoted as `np.var(x)` in NumPy, while the unbiased variance is obtained using `np.var(x, ddof=1)`.

In deep learning, mean and variance are essential for normalization and regularization techniques.

Examples:
1.  Mean: Collection of numbers: 2, 0, 4, 1, -2, 7 Mean: (2+0+4+1-2+7)/6 = 12/6 = 2
2.  Variance: Collection of numbers: 2, 0, 4, 1, -2, 7 Mean: 2 Variance: ((2-2)² + (0-2)² + (4-2)² + (1-2)² + (-2-2)² + (7-2)²) / (6-1) = (0 + 4 + 4 + 1 + 16 + 25) / 5 = 50/5 = 10
3.  Standard deviation: Variance: 10 Standard deviation: √10 ≈ 3.16    
4.  Mean absolute difference (MAD): Collection of numbers: 2, 0, 4, 1, -2, 7 Mean: 2 MAD: (|2-2| + |0-2| + |4-2| + |1-2| + |-2-2| + |7-2|) / 6 = (0 + 2 + 2 + 1 + 4 + 5) / 6 = 14/6 ≈ 2.33