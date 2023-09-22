## Overview
-   Modern CPUs (post 1980s) use techniques like pipelines, superscalar methods, out-of-order instruction handling, speculative execution, and simultaneous multithreading to maximize performance.
-  Some cores even have more than one hardware thread, referred to as simultaneous multithreading (SMT) or hyperthreading (HT), each appearing as an independent CPU to software.
-  Achieving full performance with a CPU having a long pipeline requires a predictable control flow through the program.
-   However, branch prediction can be challenging for certain types of programs, leading to pipeline flushes and performance issues.

## Pipelined CPUs
-   In contrast to the typical microprocessor of the 1980s, modern CPUs execute many instructions simultaneously to optimize the flow of instructions and data.
-   Techniques used include:
    -   Pipelines: Helps in executing multiple instructions at different stages simultaneously.
    -   Superscalar techniques: Enable a single processor to process more than one instruction during a clock cycle.
    -   Out-of-order execution: Allows the CPU to execute instructions not in the original program order but in the order that their operands are ready.
    -   Speculative execution: Permits the CPU to predict the outcome of branches and execute the instructions ahead of time.
-   Some cores even feature hyperthreading or simultaneous multithreading (SMT) where each hardware thread appears as an independent CPU to software.

## Benefits of Pipelining
-   Pipelining in CPUs can greatly improve performance, particularly in programs with highly predictable control flows, such as those executing primarily in tight loops.
-   The CPU can correctly predict the branches in these cases, allowing the pipeline to be kept full and the CPU to execute at full speed.

## Challenges with Pipelining
-   Branch prediction can be difficult in programs with many loops each iterating a random number of times or those with many virtual objects that can reference different real objects with different implementations for frequently invoked member functions.
-   If the CPU can't predict where the next branch might lead, it either stalls waiting for execution to proceed far enough to determine where that branch leads or it guesses using speculative execution.
-   Wrong guesses can lead to pipeline flushes where the CPU must discard any speculatively executed instructions following the wrong branch, reducing overall performance.

## Hyperthreading and Resource Contention

-   In the case of hyperthreading (or SMT), all the hardware threads sharing a core also share that core’s resources, including registers, cache, execution units, and so on.
-   The execution of one hardware thread can often be disrupted by the actions of other hardware threads sharing that core, leading to potential issues with resource contention.
-   Counterintuitive results can arise even with a single active hardware thread due to overlapping capabilities of execution units, which can lead to pipeline stalls and unpredictable performance.

## Future Challenges

-   Pipeline flushes and shared-resource contention are not the only challenges for modern CPUs.
-   The next sections will cover the hazards of referencing memory.