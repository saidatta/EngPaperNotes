
## Introduction

- Multiple perspectives to approach a problem and arrive at the same answer is beautiful and compelling in mathematics and science.
- This video shows how to solve the linear model fitting problem from a purely algebraic perspective.
- Terminology: Instead of AX=B, X is called the designed matrix or the matrix of independent variables.
- X is a tall matrix and can have a one-sided left inverse if it has a linearly independent set of columns.

## Solving the Linear Model Fitting Problem

- Expand the equation Xβ=Y and isolate β on the left-hand side of the equation by getting rid of X.
- Left multiply both sides of the equation by the left inverse to isolate β.
- The left inverse has the general form: (X^T X)^(-1) X^T
- The solution to the vector of β coefficients is: β = (X^T X)^(-1) X^T Y

## Condition for Left Inverse

- X is a tall matrix with rank n (full column rank) to have a left inverse.
- X on its own is not invertible, and it is not possible to apply the inverse operation to X and X transpose separately.
- If the columns of X are linearly dependent, X does not have a full left inverse. The statistical term for this situation is multi-collinearity.

## Column Space of Matrix X

- Linear least squares modeling is based on the assumption that the response variable Y lies in the column space of X.
- Y typically has more elements than the columns in X, so it's not trivial whether Y is in the column space of X.
- An exact solution is possible when Y is in the column space of X, but typically Y is not in the column space of X.
- You can find a vector Y hat that is in the column space of X that is as close as possible to Y. The difference between Y and Y hat is called epsilon.

## Conclusion

- Linear least square modeling can be solved algebraically by using the left inverse of the designed matrix.
- The condition for the matrix to have a left inverse is for it to be a tall matrix with full column rank.
- An exact solution to the linear least square model is possible when the response variable Y lies in the column space of X.
- Y typically has more elements than the columns in X, so an alternative approach is to find a vector Y hat in the column space of X that is as close as possible to Y.