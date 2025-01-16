**Context:** These notes build on previously discussed design considerations, focusing on composite types, domain-specific types, overall program structure, and concluding with a Jzero language definition and a Unicon case study. The goal is to deepen understanding of how language design decisions around data types and program organization can shape a language’s usability and complexity.

---

### Composite Types

**Definition:** Composite types are aggregates that hold multiple values, often of differing types. They serve as “glue” for building complex data structures and abstractions in a programming language.

#### Common Composite Types:
- **Arrays:**  
  - Contiguous storage of elements accessed by index.
  - Typically zero-based indexing (like C, Java, Python), but can also be one-based (like older languages such as Fortran or Lua).
  - Size changes: some languages support dynamic resizing (e.g., Python lists), while others require fixed-size arrays.
  
  **Example (Jzero-like syntax):**
  ```java
  int arr[] = new int[10]; 
  // Access:
  arr[0] = 42;
  ```
  
- **Records/Structs/Classes:**
  - Group different types under named fields.
  - Enable user-defined data structures.
  - From simple `struct` in C to fully object-oriented `class` with methods in C++/Java.
  
  **Example (C-style struct):**
  ```c
  struct Point {
      int x;
      int y;
  };

  struct Point p;
  p.x = 10;
  p.y = 20;
  ```

- **Tables/Hash Maps/Dictionaries:**
  - Indexed by keys, which need not be sequential integers.
  - Highly useful for flexible data models and runtime-computed keys.
  
  **Example (pseudo-code):**
  ```plaintext
  table phoneBook = { "Alice": "555-1234", "Bob": "555-5678" }
  phoneBook["Carol"] = "555-9999"
  ```

**Design Considerations:**
- Complexity of implementation (e.g., adding a GC-friendly hash map might be harder than a simple array).
- Domain requirements: do the target applications frequently need associative data structures?
- Performance and memory implications of each composite type.
  
---

### Domain-Specific Types

**Motivation:** If your language is aimed at a particular domain (e.g., scientific computing, graphics, financial modeling), you might include specialized, built-in types tuned for that domain.

**Trade-offs:**
- Domain-specific features can vastly simplify user code for niche tasks.
- However, they increase complexity and reduce generality.
- Consider whether a library-based approach (like Java’s standard libraries) suffices before baking domain-specific types into the language syntax and semantics.

**Examples:**
- Specialized numeric types for big integers or arbitrary precision arithmetic.
- Financial quantities with built-in currency and rounding rules.
- Graphics primitives (e.g., `Point2D`, `Color`, `Mesh` for 3D objects) built into the language rather than just libraries.

---

### Overall Program Structure

**Key Questions:**
1. **Starting Execution:**  
   - C-like languages: execution starts at `main()`.
   - Scripting languages: code executes as it is read (no explicit main).
   - Java-like: a `main()` in a class is the entry point, but multiple `main()` methods may exist across classes.
   
   **Example (C-style main):**
   ```c
   int main() {
       // program starts here
       return 0;
   }
   ```

2. **Compilation and Linking Model:**
   - Single-file, all-in-one compilation vs. separate compilation units linked together.
   - Simplicity might mean forcing all code at runtime (like scripting languages).
   - Complexity arises when you introduce modules, packages, and linking external binaries.

3. **Nesting and Program Organization:**
   - Early C: mostly flat organization – a set of functions and global data.
   - Pascal: heavily nested, with functions inside functions and more hierarchical structure.
   - Modern Trend: Many languages evolve to allow more nesting (classes within classes, lambdas, modules).
   
   **Guidance:**
   - Avoid unnecessary nesting to reduce complexity.
   - Keep program structure as simple as possible unless you have a strong reason to introduce complexity.
   
---

### Completing the Jzero Language Definition

**Recap:**  
Jzero is a simplified, Java-like language described in these notes. It aims to be a small, toy subset of Java, suitable for illustrating compiler construction.

**Key Features:**
- **Program Structure:**  
  - A single class per file.
  - All methods and variables are static.
  - A `main()` method is required and serves as the entry point.

- **Statements:**
  - `if`, `if-else`, `while`, assignments, and void method calls.
  
  **Example:**
  ```java
  if (x < 10) {
      x = x + 1;
  } else {
      x = 0;
  }
  ```
  
- **Expressions:**
  - Arithmetic (`+`, `-`, `*`, `/`), relational (`<`, `>`, `==`, `!=`), and Boolean logic operators.
  - Non-void method calls are allowed as part of expressions.
  
- **Data Types:**
  - Atomic: `bool`, `char`, `int`, `long`
    - In Jzero, `int` and `long` are both 64-bit integers for simplicity.
  - Arrays are supported.
  - Basic built-in classes: `String`, `InputStream`, `PrintStream`
    - `String`: supports concatenation (`+`), `charAt()`, `equals()`, `length()`, `substring(b,e)`, `valueOf()`.
    - `InputStream`: `read()`, `close()`.
    - `PrintStream`: `print()`, `println()`, `close()`.
  
**Note:** Jzero is intentionally minimal. It is not intended as a serious production language but as a teaching tool.

---

### Case Study: Graphics in Unicon

**Motivation:**  
Unicon added 2D and later 3D graphics as built-ins, moving beyond what many languages offer only as libraries. This illustrates the tension between adding domain-specific features directly into the language vs. relying on external libraries.

**2D Graphics in Unicon:**
- Introduced as a minimal extension, adding a `window` type that behaves like a file.
- Minimal syntax changes: 19 new keywords, mostly for event handling.
- I/O on windows parallels text file I/O, making 2D drawing operations feel integrated.
- Control flow remains in the user program, rather than delegating to a library’s event loop.

**3D Graphics in Unicon:**
- Added as a natural extension of 2D.
- Supports camera views, a display list for 3D objects, and attributes like color and font.
- The same design goals: minimal syntax overhead and integration with existing language semantics.

**Lessons Learned:**
- Embedding complex domains (like graphics) into the language can enhance productivity and portability.
- Must balance minimalism and power.
- Could consider adding specialized control structures for domain-specific needs, but must weigh complexity.

---

### Summary and Reflection

**What We Covered:**
- Composite data types (arrays, records, tables) and their design implications.
- Domain-specific types and when they might be worth adding directly into a language.
- Overall program structure and entry points, linking/loading models, and how nesting influences complexity.
- A look at the Jzero toy language specification and Unicon’s integrated graphics support as examples of design trade-offs.

**Practical Advice for Language Designers:**
- Keep it simple. Add complexity only where it delivers substantial value.
- Understand your target domain; if domain-specific types will save users from excessive boilerplate, consider adding them.
- Think carefully about starting points, module organization, and compilation models.
- Model your composite types to balance performance, usability, and implementation complexity.

---

### Discussion Questions

1. **Reserved Words:**
   - More reserved words = richer feature set but potentially more complexity and naming conflicts.
   - Fewer reserved words = simpler lexical analysis, but maybe fewer built-in constructs.

2. **Literal Lexical Complexity:**
   - Even integer literals: consider support for hexadecimal (`0x123`), binary (`0b1011`), underscores in numbers (`1_000_000`), or big integer suffixes.
   - Complexity grows with floating-point, Unicode digits, or domain-specific numeric formats.

3. **Avoiding Semicolons:**
   - Line-based syntax: statements end at newline.
   - Significant indentation (like Python).
   - End-of-line inference (like Go’s automatic semicolon insertion).

4. **Multiple `main()` Methods (Java-Style):**
   - Offers flexibility and multiple entry points (e.g., testing different parts).
   - Can be confusing; must specify which `main` to run.

5. **Pre-opened Resources:**
   - Standard I/O streams are universal; adding standard “pre-opened” network, database, or graphical resources may be less universal and harder to abstract.
   - Could simplify certain domains but reduce portability and increase language complexity.
   
---

**Next Steps:**
- After finalizing your language design, move on to **implementation details**, starting with lexical analysis (tokenizing input source code).
- Continue refining your specification as you encounter difficulties in the compiler or runtime implementations.

---

**Related Links:**
- [[Language Semantics]]
- [[Memory Management in Language Implementation]]
- [[Domain-Specific Languages (DSLs)]]
- [[Compiler Construction]]

---

> [!TIP] **Domain-Specific Data Structures**  
> If your language targets a niche domain, domain-specific composite types can drastically reduce boilerplate. However, always consider whether these additions justify their implementation complexity and maintenance overhead.

---

End of Notes.