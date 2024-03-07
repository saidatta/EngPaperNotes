
https://juejin.cn/post/7247089302529589303
https://www.baeldung.com/jvm-tiered-compilation
https://www.baeldung.com/jvm-code-cache
### Overview
- The Just-In-Time (JIT) compiler is a key component of the Java Virtual Machine (JVM) that boosts Java application performance by converting bytecode to native machine code at runtime.
- JIT compiler functions include interpreting source/intermediate code, dynamically compiling "hot spots" to machine code, and optimizing this compilation.

### JIT Compiler Principle
1. **Interpreter Stage**: The source code or intermediate code is executed line by line.
2. **Compilation Phase**: Frequently executed blocks or functions are identified as "hot spots" and compiled into machine code stored in the `codeCache`.
3. **Optimization Phase**: The JIT compiler performs optimizations (e.g., eliminating redundant calculations) during compilation.
4. **Execution Phase**: The JIT uses pre-compiled machine code for "hot spots" to improve execution speed.
![[Screenshot 2024-02-11 at 4.59.05 PM.png]]
### Classification
- **Interpreter-Based JIT Compiler**: Interprets and executes code in real-time, dynamically compiling frequently executed code.
- **Static Compiler-Based JIT Compiler**: Optimizes and recompiles pre-compiled code at runtime.
- **Tracing-Based JIT Compiler**: Dynamically analyzes execution paths and compiles frequently executed blocks.
- **Method Inlining-Based JIT Compiler**: Merges small methods into a large one for compilation to reduce method call overhead.

### JVM Compilers
- **Client Compiler (C1)**: Focuses on startup speed with local optimizations, converting bytecode to High-level Intermediate Representation (HIR), then to Low-level Intermediate Representation (LIR), and finally to machine code.
- **Server Compiler (C2 and Graal)**: Emphasizes global optimizations with longer compile times but better performance. Graal, introduced in JDK 9, is written in Java and offers deeper optimization and better performance than C2.

### JIT Compiler Optimization Techniques
- **Method Inlining**: Merges multiple methods to reduce call overhead.
- **Loop Expansion**: Repeats loop code to reduce overhead.
- **Constant Folding**: Pre-calculates constant expressions.
- **Redundant Code Elimination**: Removes duplicate code executions.
- **Data Flow Analysis**: Analyzes data flow for variable optimization.
- **Code Generation Optimization**: Adjusts code generation for efficiency.
- **JIT Escape Analysis**: Analyzes variable scope to optimize memory management.

### Limitations and Disadvantages
- **Compilation Time**: Increases startup and response time.
- **Memory Footprint**: Compiled code increases memory usage.
- **Hot Code Identification**: May need to recompile if "hot spots" change.
- **Concurrency**: Typically single-threaded, underutilizing multi-core CPUs.
- **Cross-Platform Support**: Requires optimization for different platforms.
- **Security**: Dynamic code generation can introduce vulnerabilities.

### Practice and Parameters
- **Compilation Parameters**: Detailed compilation-related JVM parameters, like `-XX:+TieredCompilation` and `-XX:ReservedCodeCacheSize`, control the behavior of JIT compilation.
- **JITwatch Tool**: A tool to analyze compilation logs for better understanding and optimization of JIT compilation behavior.

### Conclusion
- The JIT compiler plays a crucial role in the JVM by dynamically compiling and optimizing code to improve Java application performance. Understanding and utilizing JIT compilation effectively can lead to significant performance gains in Java applications.