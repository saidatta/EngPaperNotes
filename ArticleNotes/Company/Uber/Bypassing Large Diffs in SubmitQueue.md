https://www.uber.com/blog/bypassing-large-diffs-in-submitqueue/?uclick_id=336a56a2-cc2d-40c9-b908-10ed680e0404
### Abstract

- **SubmitQueue**: A system utilized by Uber to verify changes before merging to the main branch.
    - _Purpose_: Similar to GitHub’s “merge queue” and GitLab’s “merge train” for speculatively running CI in parallel.
    - _Problem Addressed_: Large diffs (changes) take considerable time to build and test. As Uber's codebase grows, these delays increase.
    - _Solution Introduced_: Out-of-order landing of changes, provided all paths in the speculation tree are verified and have the same outcome.
    - _Impact_: 74% improvement in waiting time to land code without compromising the integrity of the main branch.

### Introduction

- **Main Branch Principles at Uber**:
    - Always buildable.
    - Tests should always pass.
    - Any error, either locally or on CI, is probably due to the recent change rather than a pre-existing main branch issue.
    - Continuous deployment from the main branch is enabled.
- **Need for SubmitQueue**: As change rates increase, validating them sequentially becomes inefficient.
    - **Function**: Verify changes in parallel using a conflict analyzer.
        - If two changes are independent (affect disjoint sets of packages), they are processed separately.
        - Conflicting changes use a speculation tree to anticipate possible outcomes.

### Speculation Tree

- **Example**:
    - SubmitQueue receives three changes: C1, C2, and C3.
    - Possible outcomes:
        1. C1 & C2 are rejected.
        2. C1 is rejected, C2 commits.
        3. C1 commits, C2 is rejected.
        4. Both C1 & C2 commit.
    - Four different builds result: B3, B2,3, B1,3, and B1,2,3 (Refer to Figure 1 for visual representation).
- **Issues with Large Diffs**:
    - Large diffs impact a significant number of packages, making them conflict with most subsequent changes.
    - Their verification is time-consuming due to the vast number of packages affected.
    - Other changes must wait, even if they've already been verified (See Figure 2).

### Measures Taken to Combat Large Diffs

- **Initial Policy**: Land large diffs outside US business hours.
- **Next Approach**: Improve CI build time.
- **New Problem**:
    - As Uber's codebase expanded, the number of packages affected by a large diff increased.
    - Growth Metrics:
        - Largest change in a month grew by 2.5x over 2 years (Figure 3).
        - Change rate increased >2x in the same duration.
        - Build and test time increased due to codebase growth, negating previous optimizations (Figure 4).
- **BLRD (Bypassing Large Diffs) Approach**:
    - For a new change B received after a conflicting change A, SubmitQueue tests B on both `main+A` and just `main`.
    - If B's verification finishes before A's (which may take long if A is a large diff), B can technically be merged, but currently, B has to wait for A.
    - Proposal: Allow B to bypass A and get merged first (visualized in Figure 5).

### Equations & Code

- Let's consider an algorithm to implement the BLRD approach:

pythonCopy code

`def can_bypass(A, B):     # Check if both B|main+A and B|main verifications are successful     if verify(B, "main+A") and verify(B, "main"):         return True     return False  def merge_change(change):     if can_bypass(A, change):         merge_to_main(change)     else:         wait_for(A)         merge_to_main(change)`

### Conclusion & Takeaways

- **SubmitQueue** is vital for Uber's efficient CI process, especially with the company's growing codebase.
- **Large Diffs** present a challenge, causing delays in the integration of new changes.
- **Bypassing Large Diffs (BLRD)** is a potential solution to address these delays, allowing non-conflicting changes to be merged out of order if they've been verified successfully.

### Actionable Points

1. Implement a prototype for the BLRD approach in a staging environment.
2. Monitor the results and ensure the main branch remains green.
3. If successful, roll out to the production environment, and measure improvements in wait times.
## 📌 Definitions

- **T**: A target in the repo.
- **T0**: Version of target T on the main branch. The version is represented by a hash combining all input files to T and the hashes from other targets that T depends on.
- **TA**: Version of target T after applying change A on main branch.
- **TAB**: Version of target T after applying changes A and B sequentially on main branch.
- **TBA**: Version of target T after applying changes B and A sequentially on main branch.
- **CT(A|B)**: Set of targets with their hashes changed after applying change A on top of change B.

## 📝 Assumption

- Builds are hermetic: same input & configuration yield the same output.
- Tests are deterministic: not flaky.
- Target hash encodes everything affecting state of T and is unaffected by the sequence of changes applied.

## 🎯 Hypothesis

If B arrives at the SubmitQueue before A, testing needs to cover CT(B) and CT(A|B). If A arrives before B, tests should cover CT(A), CT(B) and CT(B|A). This leads to the following:

### 📜 Theorem 1

CT(A∣B)⊆CT(A)∪CT(B)∪CT(B∣A)CT(A∣B)⊆CT(A)∪CT(B)∪CT(B∣A)

## 📒 Scenarios

### Case 1

If TBA ∈CT(A|B) but TA∉CT(A), and TB ∈CT(B), but TAB ∉CT(B|A) then TAB ∈ CT(B|A).

### Case 2

If TBA ∈CT(A|B) and TA∉ CT(B), then TBA == TA and TA ∈ CT(A).

### Case 3

If TBA ∈CT(A|B) and TAB∉ CT(B|A), then T ∈ CT(A).

## 📄 Summary

If A arrives before B and all targets in CT(B) and CT(B|A) pass verification, B can land before A finishes its verification.

## 🚧 Discussion: Non-Hermetic Builds & Flaky Tests

- Target hash doesn't cover all factors affecting state of T in these cases.
- Non-hermetic builds & flaky tests are issues beyond the current hypothesis.
- Efforts made to address these problems are separate from this discussion.

## Proof: General Cases

- **Multiple Changes Bypassing**: Discusses if changes can bypass other changes in the queue.
    - Bypassing 2 Changes
    - Bypassing More Changes
    - Multiple Bypassings
    - Multiple Changes Bypassing Multiple Changes

### 📜 Theorem 2

CT(AB∣C)⊆CT(A)∪CT(B∣A)∪CT(C)∪CT(C∣AB)CT(AB∣C)⊆CT(A)∪CT(B∣A)∪CT(C)∪CT(C∣AB)

---

## 🖼️ Figures

- **Figure 6**: One change bypassing 2 changes.
- **Figure 7**: Multiple changes bypassing one change.
- **Figure 8**: Multiple changes bypassing multiple changes.

---

## 📔 Notes

- Article explains the theorem and its significance with respect to versioning and applying changes in a sequential or non-sequential manner.
- Multiple scenarios have been considered to address a broad spectrum of situations.
- This system depends on hermetic builds and deterministic tests, but also recognizes the potential pitfalls and issues that may arise outside of these conditions.

## 💡 Additional

- Could be beneficial to visualize the scenarios using flowcharts or diagrams.
- Implementing a version control system that can automatically handle these scenarios based on the hypothesis would be a significant technological advancement.
---
Analysis:

From the text provided, it seems that Uber has incorporated BLRD (Bypass Large Repositories on Demand) in their Go monorepo SubmitQueue to improve the efficiency of the queue. The positive impact of this incorporation is seen in the reduction of the P95 wait time by 74% in June compared to April. The current P95 latency is comparable to January 2022 levels despite increased complexity.

Key Points:

1. **Implementation of BLRD in the SubmitQueue for Uber’s Go monorepo**: Uber integrated BLRD into their monorepo to optimize the P95 wait time. This change seems to be a part of Uber’s initiative to enhance the efficiency of their Go monorepo.

2. **Significant Improvement in P95 wait time**: There was a 74% reduction in the P95 wait time after enabling BLRD. This is a substantial improvement, indicating the effectiveness of the BLRD implementation.

3. **Comparison with Past Performance**: The current performance after BLRD integration is at par with January 2022 levels, which had the advantages of faster builds for large diffs and fewer monthly changes.

4. **Challenges with SubmitQueue**: The SubmitQueue verifies all paths in the speculation tree, which increases exponentially with the number of conflicting changes, making it resource-intensive. The original SubmitQueue paper proposed a probabilistic model, but it is insufficient for BLRD's requirements. 

5. **Future Work - Predictive Model for Build and Test Time**: Uber's next challenge is to design a model to predict the build and test time for a change. This would help in speculating paths for changes that might take longer to verify and therefore allow subsequent changes to bypass and reduce waiting times.

Conclusion:

Uber's Go monorepo has seen significant efficiency improvements with the BLRD implementation in the SubmitQueue. However, as the system evolves, the exponential growth in the speculation tree paths due to conflicting changes is becoming a challenge. Uber's forward direction is to develop a model that can predict build and test times, to speculate paths more effectively and further optimize the SubmitQueue. This continuous innovation reflects Uber's commitment to enhancing system performance and user experience.