## Introduction to T-Test
- When running experiments in deep learning, it is necessary to determine if one model architecture or set of meta parameters is significantly better than another.
- A test is performed to determine if data drawn from one distribution is significantly different from data drawn from another distribution or sample.
- The T-test is used to compare the performance of different models or parameters and evaluate if the difference is statistically significant.

## Null and Alternative Hypotheses
- The T-test involves comparing the alternative hypothesis (H_a) with the null hypothesis (H_0).
- The alternative hypothesis suggests that the models' performance is significantly different, while the null hypothesis states that the models perform equally well.
- The goal of the test is to provide evidence against the null hypothesis and support the alternative hypothesis.

## Basic Formula of the T-Test
- The T-test calculates a single value, the T-value, which is governed by a simple equation.
- The formula for the T-value is: T = (mean(X) - mean(Y)) / (S / sqrt(N)).
- Here, mean(X) and mean(Y) are the means of the two data sets being compared, S is the standard deviation, and N represents the number of times the model is repeated.
- The T-value measures the difference between the means normalized by the standard deviations.

## Deriving the P-value from the T-Test
- The T-value needs to be evaluated relative to a distribution of T-values expected under the null hypothesis.
- The t-distribution represents the distribution of T-statistic values under the null hypothesis.
- The observed T-value from the real data is compared to a statistical significance threshold, typically determined by specifying a P-value threshold.
- The P-value represents the probability of observing a T-statistic of a given size purely by chance due to random sampling variability.
	- If the P-value is below the threshold (e.g., 0.05), there is strong evidence against the null hypothesis, indicating a significant difference between the models.

## Implementation of T-Test using SciPy
- The t-test can be implemented using the `scipy.stats.ttest_ind()` function for independent samples.
- The `ttest_ind()` function takes two datasets (e.g., accuracies of two models) as input and returns the T-value and P-value.
- The sign of the T-value is arbitrary and depends on the order of the input datasets.
- Visualization techniques, such as plotting the data points, can provide a visual understanding of the data distribution.
- The `fontsize` parameter can be adjusted to improve the readability of plots.

Understanding and implementing the T-test allows for statistical comparison between different model architectures or parameters. It enables the identification of statistically significant differences and informs decision-making regarding model selection based on performance.

#### Example of a t-test
Let's say you are a researcher studying the effects of a new teaching method on students' test scores. You want to compare the performance of two groups of students: one group that was taught using the new method (Group A) and another group that was taught using the traditional method (Group B). You collect test scores for both groups and want to determine if there is a statistically significant difference between their mean scores.

Group A test scores: [90, 85, 92, 88, 76] Group B test scores: [78, 81, 84, 80, 82]

You can use a t-test to compare the means of these two groups. In this case, we will use an independent samples t-test, which is appropriate when comparing the means of two independent groups. Here are the steps for performing the t-test:

1.  Calculate the mean and standard deviation for each group. Group A: mean = 86.2, standard deviation = 5.89 Group B: mean = 81, standard deviation = 2.24
2.  Calculate the t-statistic using the following formula:
t = (mean_A - mean_B) / sqrt((sd_A^2 / n_A) + (sd_B^2 / n_B))

where mean_A and mean_B are the means of the two groups, sd_A and sd_B are the standard deviations of the two groups, and n_A and n_B are the sample sizes of the two groups.

t = (86.2 - 81) / sqrt((5.89^2 / 5) + (2.24^2 / 5)) t ≈ 2.44

3.  Determine the degrees of freedom (df):

df = n_A + n_B - 2 df = 5 + 5 - 2 = 8

4.  Choose a significance level (commonly 0.05) and find the critical t-value using a t-distribution table or calculator. For a two-tailed test with a 0.05 significance level and 8 degrees of freedom, the critical t-value is approximately 2.306.
    
5.  Compare the calculated t-statistic to the critical t-value:
    

Since 2.44 > 2.306, we reject the null hypothesis, which states that there is no significant difference between the means of the two groups. This means that there is evidence to suggest that the new teaching method has a statistically significant impact on students' test scores compared to the traditional method.