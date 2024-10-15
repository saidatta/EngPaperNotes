=In this chapter, we will discuss two types of reality, continuous and categorical, and how we can represent them using numbers for deep learning.
## Continuous Reality
Continuous reality refers to numeric data with many distinct values, possibly an infinite number of distinct values. Some examples include:
-   Height
-   Weight
-   Income or salary
-   Exam scores
-   Review score=s

These types of data can be easily represented using numbers.
## Categorical Reality
Categorical reality refers to discrete data with a limited number of distinct values. Some examples include:
-   Picture of a landscape (sea vs. mountain)
-   Identity of a picture (cat vs. dog)
-   Disease diagnosis (present vs. absent)
Representing categorical reality using numbers requires a slightly different approach.
### Dummy Coding
Dummy coding is a method of representing categorical data with only two possible options. It assigns one option to be 0 (false) and the other to be 1 (true). Examples:
-   Pass or fail an exam
-   House sold or still on the market
-   Credit card transaction normal or fraudulent
### One-Hot Encoding
One-hot encoding is similar to dummy coding, but it is used for multiple categories. It creates a matrix where each column corresponds to a dummy-coded categorical variable, and each row corresponds to a different observation.

Example:

Movies: Y1, Y2, Y3 Genres: History, Sci-Fi, Kids

One-hot encoded matrix:

| History | Sci-Fi | Kids |
|---------|--------|------|
| 0       | 1      | 0    |
| 0       | 0      | 1    |
| 1       | 0      | 0    |

Y1 is a Sci-Fi movie, Y2 is a Kids movie, and Y3 is a History movie.

In summary, we discussed continuous and categorical reality and how to represent them using numbers. We also looked at the difference between dummy coding and one-hot encoding. Dummy coding is used for one feature, whereas one-hot encoding combines multiple dummy-coded variables for multiple features.