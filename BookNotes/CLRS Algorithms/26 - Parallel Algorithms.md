----
#### Overview
- Chapter 26 extends the algorithmic model to parallel algorithms, where multiple instructions execute simultaneously, focusing on fork-join parallel algorithms.
- These detailed notes explore the technical aspects of parallel algorithms, with a focus on fork-join parallelism, the P-FIB procedure, and the analysis of parallel execution using graphs and scheduling theories. They include specific algorithms, scheduling models, and performance measurement methods, providing a comprehensive understanding of parallel algorithm design and analysis.

---
#### Parallel Computers and Multicores
- Computers with multiple processing cores, each accessing a shared memory.
- **Types**:
  - Handheld, laptop, desktop, cloud machines (multicores).
  - Clusters and supercomputers with distributed memory.
---
#### Thread Parallelism
- Uses virtual processors (threads) sharing a common memory.
- **Challenge**: Difficult to dynamically partition work and balance load among threads.
---
#### Task-Parallel Programming
- **Model**: Specifies what computational tasks may run in parallel, without indicating the specific thread or processor.
- **Benefits**:
  - Simplifies programming by abstracting thread management and load balancing.
  - Utilizes a scheduler to distribute tasks.
  - Incorporates work/span analysis for performance reasoning.
---
#### Fork-Join Parallelism
- **Key Features**:
  - **Spawning**: Allows parallel execution of subroutines.
  - **Parallel Loops**: Iterations execute simultaneously.
- **Advantages**:
  - Simplifies parallel algorithm description.
  - Natural for divide-and-conquer algorithms.
  - Theoretical framework for work and span analysis.
---
#### Fork-Join Parallel Algorithms
- **Pseudocode Extensions**: `parallel`, `spawn`, `sync`.
- **Serial Projection**: Removing parallel keywords transforms into a serial algorithm.
- **Supported Environments**: Cilk, Habanero-Java, Java Fork-Join Framework, OpenMP, etc.
---
#### Parallel Fibonacci Calculation
- **Serial Algorithm**:
  ```python
  FIB(n)
  1 if n ≤ 1
  2     return n
  3 else 
  4     x = FIB(n − 1)
  5     y = FIB(n − 2)
  6     return x + y
  ```
- **Running Time**: \( T(n) = T(n - 1) + T(n - 2) + Θ(1) \), solution \( T(n) = Θ(F_n) \).
---
#### Parallel Fibonacci Algorithm
- **Modification**: Execute `FIB(n − 1)` and `FIB(n − 2)` in parallel.
- **Example**: 
  ```python
  PARALLEL-FIB(n)
  1 if n ≤ 1
  2     return n
  3 else 
  4     spawn x = FIB(n − 1)
  5     y = FIB(n − 2)
  6     sync
  7     return x + y
  ```
- **Performance**: Improved efficiency due to parallel execution of recursive calls.

---
#### Work/Span Analysis
- **Work (T1)**: Total time to execute all operations in serial.
- **Span (T∞)**: Time taken by the longest sequence of dependent operations (critical path).
- **Parallelism (T1/T∞)**: Measure of the algorithm's inherent parallelizability.
---
#### Invocation Tree for FIB(6)
- **Illustration**: Shows repeated calls to the same instances, highlighting inefficiency.
- **Figure 26.1 in CLRS**: Demonstrates the tree structure of recursive calls for `FIB(6)`.
- ![[Screenshot 2023-12-21 at 4.48.55 PM.png]]
_Figure 26.1 The invocation tree for FIB(6). Each node in the tree represents a procedure instance_ whose children are the procedure instances it calls during its execution. Since each instance of FIB with the same argument does the same work to produce the same result, the inefficiency of this algorithm for computing the Fibonacci numbers can be seen by the vast number of repeated calls to compute the same thing. The portion of the tree shaded blue appears in task-parallel form in Figure 26.2._

![[Screenshot 2023-12-21 at 4.50.22 PM.png]]
_Figure 26.2 The trace of P-FIB(4) corresponding to the shaded portion of Figure 26.1. Each_ circle represents one strand, with blue circles representing any instructions executed in the part of the procedure (instance) up to the spawn of P-FIB (n − 1) in line 3; orange circles_ representing the instructions executed in the part of the procedure that calls P-FIB (n − 2) in_ line 4 up to the sync in line 5, where it suspends until the spawn of P-FIB (n − 1) returns; and_ white circles representing the instructions executed in the part of the procedure after the sync, where it sums x and y, up to the point where it returns the result. Strands belonging to the same_ procedure are grouped into a rounded rectangle, blue for spawned procedures and tan for called_ procedures. Assuming that each strand takes unit time, the work is 17 time units, since there are 17 strands, and the span is 8 time units, since the critical path—shown with blue edges contains 8 strands._

---
#### Practical Application
- **Use Cases**: Optimizing algorithms for multicore environments.
- **Challenge**: Identifying independent tasks for parallel execution.
---
#### Parallel Keywords in P-FIB Procedure
- **P-FIB Procedure**: Computes Fibonacci numbers using parallel constructs.
- **Parallel Keywords**: `spawn` and `sync`.
- **Serial Projection**: Removing `spawn` and `sync` transforms P-FIB into the serial FIB algorithm.
- **P-FIB Algorithm**:
  ```python
  P-FIB(n)
  1 if n ≤ 1
  2     return n
  3 else 
  4     x = spawn P-FIB(n − 1)
  5     y = P-FIB(n − 2)
  6     sync
  7     return x + y
  ```
---
#### Semantics of Parallel Keywords
- **Spawning**: Initiates parallel execution of a procedure (`spawn` keyword).
- **Synchronization**: Waits for completion of spawned tasks (`sync` keyword).
- **Dynamic Parallel Execution**: Creates a potentially large tree of subcomputations executing in parallel.
---
#### Graph Model for Parallel Execution
- **Trace**: The execution of a parallel computation visualized as a directed acyclic graph (DAG).
- **Vertices**: Represent executed instructions or aggregated chains (strands).
- **Edges**: Represent dependencies between instructions.
- **Example (P-FIB(4))**: Work = 17 time units, Span = 8 time units.
---
#### Performance Measures in Task-Parallel Algorithms
- **Work (T1)**: Total time on one processor; sum of all strand execution times.
- **Span (T∞)**: Time on unlimited processors; length of the critical path in the trace.
- **Parallelism**: \( T1/T∞ \), the average amount of work that can be performed in parallel.
- **Slackness**: \( (T1/T∞)/P \), the ratio of parallelism to the number of processors.
---
#### Scheduling in Parallel Algorithms
- **Greedy Schedulers**: Assign as many strands to processors as possible at each step.
- **Types of Scheduler Steps**:
  - Complete step: Fully utilizes all processor resources.
  - Incomplete step: Some processors remain idle.
---
#### Parallel Fibonacci Algorithm Analysis
- **Running Time**: Depends on work, span, and the number of processors.
- **Speedup**: Measured by \( T1/TP \), indicating the efficiency gained by parallel execution.
- **Upper Bound on Running Time**: \( TP \<= T1/P + T∞ \).
---
#### Parallel Control in Trace Analysis
- **Determining Parallel Control**: Inferring whether a strand was spawned or joined based on trace dependencies.
- **Execution Order**: Implied by the topological sort of the trace.
---
#### Ideal Parallel Computer Assumptions
- **Sequential Consistency**: Memory behaves as if instructions execute one at a time in a global linear order.
- **Performance Assumptions**: Equal computing power for each processor, ignoring scheduling cost.
---
#### Theorem 26.1: Greedy Scheduler Execution Time
- **Context**: On an ideal parallel computer with P processors.
- **Statement**: A greedy scheduler executes a task-parallel computation with work \( T_1 \) and span \( T_\infty \) in time \( TP \leq T_1/P + T_\infty \).
- **Proof Outline**:
  - Complete steps: Each does P work, total work for k complete steps is kP.
  - Incomplete steps: Reduce span by 1, the number of incomplete steps is at most \( T_\infty \).
  - Total time is sum of complete and incomplete steps.

---
#### Corollary 26.2: Greedy Scheduler Efficiency
- **Statement**: Running time \( TP \) of a task-parallel computation by a greedy scheduler on P processors is within a factor of 2 of optimal.
- **Proof Outline**:
  - Let \( T^*_P \) be optimal running time on P processors.
  - By work and span laws, \( T_1 \leq T^*_P P \) and \( T_\infty \leq T^*_P \).
  - Combining with Theorem 26.1, \( TP \leq T_1/P + T_\infty \) is within a factor of 2 of \( T^*_P \).
---
#### Corollary 26.3: Linear Speedup with High Slackness
- **Statement**: For a task-parallel computation with high slackness (\( P \ll T_1/T_\infty \)), a greedy scheduler on P processors achieves near-perfect linear speedup (\( TP \approx T_1/P \)).
- **Proof Outline**:
  - If \( P \ll T_1/T_\infty \), then \( T_\infty \ll T_1/P \).
  - By Theorem 26.1, \( TP \leq T_1/P + T_\infty \approx T_1/P \).
  - The speedup \( T_1/TP \approx P \).

---
#### Analyzing Parallel Algorithms: Work/Span Analysis
- **Work Analysis**: Equivalent to analyzing the running time of the serial projection.
- **Span Analysis**: New aspect introduced by parallelism, generally straightforward.
- **Example (P-FIB Program)**:
  - Work \( T_1(n) \) is \( Θ(φ^n) \).
  - Span \( T_∞(n) \) is \( Θ(n) \).
  - Parallelism \( T_1(n)/T_∞(n) \) grows dramatically with n.
---
#### Parallel Loops
- **Example (P-MAT-VEC Procedure)**:
  - Multiplies a square matrix \( A \) by a vector \( x \).
  - Uses `parallel` keyword for loop parallelization.
  - Work \( T_1(n) \) is \( Θ(n^2) \).
  - Span \( T_∞(n) \) is \( Θ(n) \).
  - Parallelism is \( Θ(n) \).
---
#### Recursive Spawning and Span Analysis
- **Impact on Work**: Increases work but not asymptotically.
- **Impact on Span**: Must be considered; span includes parallel loop control time.
- **Span for Parallel Loops**: \( T_∞(n) = Θ(\log n) + \max \{ \text{iter}_∞(i) : 1 ≤ i ≤ n \} \).

---
#### Race Conditions
- **Determinacy Races**: When two parallel tasks access the same memory location and at least one modifies it.
- **Consequences**: Can lead to nondeterministic behavior even in algorithms intended to be deterministic.
- **Famous Incidents**: Therac-25 and Northeast Blackout of 2003.
---
#### RACE-EXAMPLE Procedure
- **Illustration of Determinacy Race**:
  ```python
  RACE-EXAMPLE()
  1 x = 0
  2 parallel for i = 1 to 2
  3     x = x + 1 // determinacy race
  4 print x
  ```
- **Potential Output**: The output can unpredictably be 1 or 2 due to parallel updates to `x`.
---
#### Sequential Consistency and Races
- **Concept**: Parallel execution as an interleaving of instructions.
- **Example (Figure 26.5)**: Shows how interleaving can lead to unexpected results.
---
#### Mitigating Race Conditions
- **Task-Parallel Programming Tools**: Often include race-detection tools to identify and resolve race conditions.
- **Design Principle**: Ensure that parallel strands are mutually noninterfering (only read, do not modify shared locations).
---
#### P-MAT-VEC-WRONG Procedure
- **Faulty Parallel Implementation**:
  ```python
  P-MAT-VEC-WRONG(A, x, y, n)
  1 parallel for i = 1 to n
  2 parallel for j = 1 to n
  3     yi = yi + aij xj // determinacy race
  ```
- **Issue**: Race conditions when updating `yi` in parallel.
---
#### Index Variables in Parallel Loops
- **No Race Condition**: Different iterations of a parallel loop access different instances of the index variable.
---
#### Chess Programming Anecdote
- **Context**: Development of a parallel chess-playing program.
- **Optimization**: Reduced running time on a small machine, but increased on a larger one.
- **Work/Span Analysis**:
  - Original Version: \( T_1 = 2048 \) seconds, \( T_\infty = 1 \) second.
  - Optimized Version: \( T_1' = 1024 \) seconds, \( T_\infty' = 8 \) seconds.
- **Conclusion**: Optimization improved performance on a small machine (32 processors) but degraded on a larger machine (512 processors).
- **Moral**: Work/span analysis is crucial for understanding an algorithm's scalability.
---
#### Parallelizing Matrix Multiplication Algorithms
- **Goal**: Parallelize three matrix multiplication algorithms (Sections 4.1 and 4.2) using parallel loops or recursive spawning.
- **Analysis**: Work/span analysis to evaluate performance on one processor and scalability on multiple processors.
---
#### P-MATRIX-MULTIPLY Algorithm
- **Approach**: Parallelizes the two outer loops of the MATRIX-MULTIPLY procedure.
- **Pseudocode**:
  ```python
  P-MATRIX-MULTIPLY(A, B, C, n)
  1 parallel for i = 1 to n
  2 parallel for j = 1 to n
  3     for k = 1 to n
  4         cij = cij + aik · bkj
  ```
- **Analysis**:
  - Work (\(T_1(n)\)): \(Θ(n^3)\) (same as MATRIX-MULTIPLY).
  - Span (\(T_∞(n)\)): \(Θ(n)\).
  - Parallelism: \(Θ(n^2)\).

---

#### P-MATRIX-MULTIPLY-RECURSIVE Algorithm
- **Approach**: Parallelizes the divide-and-conquer strategy of MATRIX-MULTIPLY-RECURSIVE.
- **Pseudocode**:
  ```python
  P-MATRIX-MULTIPLY-RECURSIVE(A, B, C, n)
  [Details omitted for brevity; includes recursive spawning and usage of a temporary matrix D]
  ```
- **Analysis**:
  - Work (\(M_1(n)\)): \(Θ(n^3)\) (using the master theorem).
  - Span (\(M_∞(n)\)): \(Θ(\log^2 n)\).
  - Parallelism: \(Θ(n^3/\log^2 n)\).

---

#### Parallelizing Strassen’s Method
- **Approach**: Uses spawning for parallelizing the Strassen's algorithm.
- **Steps**:
  1. Scalar multiplication and partitioning (base case).
  2. Creation of matrices \(S_1, S_2, \ldots, S_{10}\), and initialization of \(P_1, P_2, \ldots, P_7\).
  3. Recursive spawning for computing matrix products \(P_1, P_2, \ldots, P_7\).
  4. Updating result matrix \(C\) using \(P_i\) matrices.
- **Analysis**:
  - Work (\(T_1(n)\)): \(Θ(n^{\log_7})\).
  - Span (\(T_∞(n)\)): \(Θ(\log^2 n)\).
  - Parallelism: \(Θ(n^{\log_7}/\log^2 n)\).

---

### Obsidian Notes for "Algorithms by CLRS (4th Edition)" - 26.3 Parallel Merge Sort

---

#### Parallel Merge Sort Overview
- **Background**: Serial merge sort is efficient with a running time of \(Θ(n \log n)\).
- **Strategy**: Implement using fork-join parallelism to enhance performance.

---

#### P-MERGE-SORT Algorithm
- **Modification**: Spawns the first recursive call of the merge sort process.
- **Pseudocode**:
  ```python
  P-MERGE-SORT(A, p, r)
  1 if p ≥ r
  2     return
  3 q = ⌊(p + r)/2⌋
  4 spawn P-MERGE-SORT(A, p, q)
  5 spawn P-MERGE-SORT(A, q + 1, r)
  6 sync
  7 P-MERGE(A, p, q, r)
  ```
- **Key**: Parallelizes recursive sorting and merges using the P-MERGE procedure.

---

#### Need for a Parallel Merge Procedure
- **Challenge**: Parallelizing only the merge sort (without the merge) offers limited parallelism.
- **Example**: Replacing P-MERGE with a serial merge would result in unimpressive parallelism (\(Θ(\log n)\)).

---

#### P-MERGE Procedure
- **Function**: Merges two sorted subarrays of an array A into another array B in parallel.
- **Approach**: Uses recursive auxiliary procedure P-MERGE-AUX.
- **Concept**: Divides and conquers by finding a pivot that splits subarrays for parallel merging.

---
#### P-MERGE-AUX Procedure
- **Parameters**: Merges A[p1 : r1] and A[p2 : r2] into B[p3 : r3].
- **Process**:
  - Finds pivot as median of the larger subarray.
  - Uses binary search to find split point in the smaller subarray.
  - Recursively merges smaller and larger elements in parallel.
---
#### FIND-SPLIT-POINT Procedure
- **Purpose**: Finds a split point in a sorted subarray relative to a pivot.
- **Method**: Binary search to find where all elements are at most (or at least) the pivot.
- **Complexity**: \(Θ(\log n)\) in work and span.
---
#### Analysis of Parallel Merge Sort
- **Key Insight**: Parallel divide-and-conquer strategy for merging improves overall span.
- **Span Reduction**: Aims to decrease the span of merging to enhance parallel efficiency.
- **Parallelism**: Achieves significant parallelism while maintaining efficient work complexity.
---
### Obsidian Notes for "Algorithms by CLRS (4th Edition)" - 26.3 Parallel Merge Sort (Work/Span Analysis)
---
#### Work/Span Analysis of Parallel Merging (P-MERGE-AUX)
- **Span Analysis** (\( T_∞(n) \)):
  - Worst-case span on n elements.
  - Dominated by \( FIND-SPLIT-POINT \) and recursive calls.
  - Recurrence: \( T_∞(n) = T_∞(3n/4) + Θ(\log n) \), solved as \( T_∞(n) = Θ(\log^2 n) \).

- **Work Analysis** (\( T_1(n) \)):
  - Total work is linear, \( T_1(n) = Θ(n) \).
  - Recurrence: \( T_1(n) = 2T_1(3n/4) + Θ(\log n) \), solved as \( T_1(n) = Θ(n) \).
  - Based on binary search and recursive merging of elements.

---

#### Analysis of P-MERGE
- **Span**: Dominated by \( P-MERGE-AUX \), resulting in \( Θ(\log^2 n) \).
- **Work**: Parallel copy loop adds \( Θ(n) \) work, matching \( P-MERGE-AUX \)'s work.

---

#### Analysis of P-MERGE-SORT
- **Work Recurrence** (\( T1(n) \)):
  - Given by: \( T1(n) = 2T1(n/2) + Θ(n) \).
  - Solved as \( T1(n) = Θ(n \log n) \) (Case 2, Master Theorem, \( k = 0 \)).

- **Span Recurrence** (\( T∞(n) \)):
  - Given by: \( T∞(n) = T∞(n/2) + Θ(\log^2 n) \).
  - Solved as \( T∞(n) = Θ(\log^3 n) \) (Case 2, Master Theorem, \( k = 2 \)).

- **Parallelism**:
  - \( T1(n)/T∞(n) = Θ(n \log n)/Θ(\log^3 n) = Θ(n/\log^2 n) \).
  - Significantly better than P-NAIVE-MERGE-SORT's parallelism of \( Θ(\log n) \).

- **Practical Considerations**:
  - In practice, coarsening the base case for small subarrays (switching to an efficient serial sort like quicksort) can improve performance by reducing constant factors.

---

### Problem Solutions for "Algorithms by CLRS (4th Edition)" - Chapter 26

---

#### Problem 26-1: Implementing Parallel Loops using Recursive Spawning

a. **Rewriting SUM-ARRAYS Using Recursive Spawning**
   - Use recursive spawning similar to `P-MAT-VEC-RECURSIVE`.
   - The procedure divides the array into halves recursively and performs addition in parallel.
   - **Parallelism Analysis**:
     - Work \( T_1(n) = Θ(n) \) (linear work for addition).
     - Span \( T_∞(n) = Θ(\log n) \) (due to recursive halving).
     - Parallelism \( T_1(n)/T_∞(n) = Θ(n/\log n) \).

b. **Grain-Size = 1 for SUM-ARRAYS′**
   - With grain-size = 1, each addition is performed independently.
   - **Parallelism**: Limited to \( Θ(1) \), as each element is added serially.

c. **Span of SUM-ARRAYS′ and Best Grain-Size**
   - **Span Formula**: \( T_∞(n, \text{grain-size}) = Θ(\log(n/\text{grain-size})) \).
   - To maximize parallelism, the grain-size should be large enough to reduce overhead but small enough to maintain parallelism.
   - Optimal grain-size balances parallelism with overhead minimization.

---

#### Problem 26-2: Avoiding a Temporary Matrix in Recursive Matrix Multiplication

a. **Parallelizing Without Temporary Matrices**
   - Use spawns for recursive calls but synchronize before adding results.
   - Avoid races by ensuring that writes to the same cell are not done in parallel.

b. **Work and Span Recurrences**
   - **Work Recurrence**: \( T_1(n) = 8T_1(n/2) + Θ(n^2) \) (No change in work).
   - **Span Recurrence**: Incorporate synchronization overhead.

c. **Parallelism Analysis**
   - Likely lower than \( P-MATRIX-MULTIPLY-RECURSIVE \) due to synchronization.
   - For 1000 × 1000 matrices, parallelism may still be high, but slightly reduced.
   - Trade-off may be worthwhile based on overhead reduction.

---

#### Problem 26-3: Parallel Matrix Algorithms

a. **LU Decomposition**
   - Parallelize by dividing matrix operations.
   - Analyze work, span, and parallelism.

b. **LUP Decomposition**
   - Similar to LU but with additional steps for permutation matrix.
   - Parallelize and analyze as above.

c. **LUP Solve**
   - Focus on parallelizing forward and backward substitution.
   - Analyze work, span, and parallelism.

d. **Matrix Inversion**
   - Use equation (28.14) and parallelize matrix operations.
   - Analyze work, span, and parallelism.

---

#### Problem 26-4: Parallel Reductions and Scan Computations

a. **P-REDUCE Design**
   - Use recursive spawning for associative operation.
   - Work \( T_1(n) = Θ(n) \), Span \( T_∞(n) = Θ(\log n) \).

b. **P-SCAN-1 Analysis**
   - Work, span, and parallelism analysis.

c. **P-SCAN-2 Correctness and Analysis**
   - Prove correctness.
   - Analyze work, span, and parallelism.

d. **Improving P-SCAN-3**
   - Fill in missing expressions.
   - Prove correctness and analyze.

e. **Analysis of P-SCAN-3**
   - Work, span, and parallelism analysis.

f. **Rewriting P-SCAN-3 Without Temporary Array**
   - Describe modifications and analyze.

g. **In-Place P-SCAN-4 Algorithm**
   - Design an in-place scan algorithm.
   - Analyze work, span, and parallelism.

h. **Parallel Parentheses Matching**
   - Use +-scan for parentheses matching.
   - Describe and analyze the algorithm.

---

#### Problem 26-5: Parallelizing a Simple Stencil Calculation

a. **SIMPLE-STENCIL Algorithm**
   - Design parallel divide-and-conquer algorithm.
   - Analyze work, span, and parallelism.

b. **Modification for n/3 × n/3 Subarrays**
   - Redesign algorithm and analyze.

c. **Generalization with Parameter b**
   - Redesign for n/b × n/b subarrays.
   - Analyze work, span, and argue o(n) parallelism.

d. **Optimized Stencil Calculation**
   - Design for Θ(n/ \log n) parallelism.
   - Discuss inherent parallelism

 limitations.

---

#### Problem 26-6: Randomized Parallel Algorithms

a. **Modifying Laws for Expectations**
   - Adapt work law, span law, and scheduler bound for random variables.

b. **Speedup Definition with Random Variables**
   - Discuss the need for E[T1]/E[TP] instead of E[T1/TP].

c. **Parallelism of Randomized Algorithms**
   - Define as E[T1]/E[T∞].

d. **Parallel Randomized Quicksort**
   - Design P-RANDOMIZED-QUICKSORT using spawns.
   - Analyze the algorithm.

e. **Analysis of Parallel Randomized Quicksort**
   - Discuss work, span, and parallelism.

f. **Parallel Randomized Select**
   - Design and analyze a parallel version.

g. **Optimizing Randomized Select for Parallelism**
   - Discuss potential optimizations for better parallelism.