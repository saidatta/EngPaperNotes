**Source:** Excerpts from *"Programming Language Design"*
These notes expand on key concepts such as composite data structures, domain-specific data types, and program structure considerations in language design. We’ll explore arrays, records/structs, advanced composite types, domain-oriented data constructs, and how these fit into the overall structure of a language. Additionally, we will examine a case study on Unicon’s graphics facilities and pose design questions that help refine language features.

---
### Table of Contents
- [[#Composite Types|Composite Types]]
    - [[#Arrays|Arrays]]
    - [[#Records Structs Classes|Records, Structs, Classes]]
    - [[#Dictionaries and Tables|Dictionaries and Tables]]
- [[#Domain-Specific Types|Domain-Specific Types]]
- [[#Overall Program Structure|Overall Program Structure]]
    - [[#Flat vs Nested Designs|Flat vs Nested Designs]]
    - [[#Linking and Execution|Linking and Execution]]
- [[#Completing Jzero Language Definition|Completing Jzero Language Definition]]
- [[#Case Study: Graphics in Unicon|Case Study: Graphics in Unicon]]
    - [[#2D Graphics in Unicon|2D Graphics in Unicon]]
    - [[#3D Graphics in Unicon|3D Graphics in Unicon]]
- [[#Summary|Summary]]
- [[#Questions|Questions]]

---

### Composite Types

**Definition:** Composite types are data types that aggregate multiple values. They can be built from atomic types or other composite types. They serve as “glue” to structure data in meaningful ways.

Common examples:
- Arrays
- Records/Structs/Classes
- Tables/Dictionaries
- Domain-specific composites (custom geometry, financial instruments, etc.)

**Design Considerations:**
- Implementation complexity vs. expressiveness.
- Familiarity to users: do you replicate known abstractions (like C structs) or innovate new composite paradigms?

---

#### Arrays

Arrays are ubiquitous in almost all programming languages.

**Design Factors:**
- **Indexing Base:**  
  - Zero-based indexing (`arr[0]`) is common (C, Java) and simplifies compiler logic.  
  - One-based indexing (`arr[1]`) can be more intuitive (older languages like Fortran).
  - Arbitrary ranges (e.g., `arr[-2..2]`) are possible but more complex to implement.

- **Resizing:**  
  - Fixed-size arrays: simple but inflexible.  
  - Dynamically resizable arrays: more complex at runtime, but more convenient for developers.

**Example Syntax (Zero-based):**  
```c
int arr[10];       // Fixed-size array of 10 ints
arr[0] = 42;       // Assign first element
```

**Example Syntax (Dynamic):**  
```javascript
let arr = [];
arr.push(42);       // Automatically resizes
```

---

#### Records, Structs, Classes

**Definition:** A record or struct aggregates heterogenous data fields, each with its own name and type. Classes extend this idea by adding methods and often access control (e.g., Java, C++).

**Trade-Offs:**
- Implementing classes with methods and inheritance increases compiler complexity.
- Minimal struct types (like C `struct`) are easier to implement but offer fewer language-level abstractions.

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

**Example (Class in a higher-level language):**  
```java
class Point {
    int x;
    int y;
    void move(int dx, int dy) {
        x += dx; y += dy;
    }
}
```

---

#### Dictionaries and Tables

**Definition:** A dictionary (hash table, map) is a composite type accessed by keys rather than numeric indices. Keys can be strings or other arbitrary data.

**Why Include?**
- Extremely common in modern programming.
- Facilitates flexible, associative lookups at runtime.
- If omitted, users must rely on libraries or roll their own data structures.

**Example (Java-like):**  
```java
Map<String, Integer> scores = new HashMap<>();
scores.put("Alice", 95);
scores.put("Bob", 87);
int aliceScore = scores.get("Alice");
```

**Design Tip:**  
If your language is domain-specific and heavily uses name-value pairs, consider making dictionaries a built-in type with dedicated syntax.

---

### Domain-Specific Types

**Motivation:**  
If your language targets a specific domain—finance, graphics, data analytics—domain-specific types can vastly simplify programming tasks. Instead of libraries full of boilerplate, you get first-class language constructs.

**Examples:**
- Complex numbers, matrices for scientific computing.
- Game entities, scenes, and resources in a game scripting language.
- Financial instruments (bonds, stocks) in a quantitative trading DSL.

**Downsides:**
- Increased implementation complexity.
- Reduced generality: types may be less useful outside the target domain.

**Guideline:**  
Add domain-specific types if they provide substantial leverage over generic solutions (libraries). Otherwise, keep them out for simplicity.

---

### Overall Program Structure

When designing a language, consider how entire programs are organized:

1. **Entry Point:**  
   - Does the program start at a `main()` function (C, Java)?  
   - Or execute top-level code as it’s read (Python, JavaScript)?

2. **Linking & Compilation Model:**  
   - Do you allow separate compilation and linking?  
   - Or must the entire program be presented at once?

3. **Nesting Constructs:**  
   - Deep nesting (functions within functions, modules within modules) adds complexity.
   - Simpler languages (C) keep nesting shallow (few layers).
   - Pascal allowed deep nesting, but complexity weighed it down in practice.

**Historical Perspective:**
- C: relatively flat structure; main entry point.  
- Pascal: heavily nested.  
C eventually became dominant, partially due to simpler structure (among other factors).

---

#### Flat vs Nested Designs

**Flat Design:**
- Easier to implement.
- Less cognitive overhead for users.
- Encourages modularity via separate files and compilation units.

**Nested Design:**
- May enable scope-limited helpers (e.g., nested functions).
- Increases complexity, both for the user and the compiler/runtime.

**Advice:** Keep nesting minimal unless you have a strong reason.

---

#### Linking and Execution

**Options:**
- Single-file scripts: simple, immediate execution.
- Multi-file compilation and linking: more powerful but requires a toolchain.
- Load code dynamically: adds flexibility, complexity, and often security concerns.

**Domain Consideration:**
- Systems languages (C, C++) rely on linkers and loaders.
- Many scripting languages (Python, Ruby) just run source files directly.

---

### Completing Jzero Language Definition

**Jzero Recap:**
- A tiny subset of Java.
- Single class per program.
- All methods and variables are static.
- Execution starts at `main()` (required).

**Allowed Statements:**
- Assignments, `if`, `while`, `void` method calls.

**Allowed Expressions:**
- Arithmetic, relational, Boolean logic.
- Non-void method invocations.

**Types:**
- Atomic: `bool`, `char`, `int`, `long` (64-bit integers).
- Arrays supported.
- Class types: `String`, `InputStream`, `PrintStream` with limited methods.

**String Methods Example:**
```java
String s = "Hello";
char c = s.charAt(1);       // 'e'
int len = s.length();       // 5
String sub = s.substring(1,4); // "ell"
```

**I/O Classes:**
- `InputStream`: `read()`, `close()`
- `PrintStream`: `print()`, `println()`, `close()`

This minimal Jzero spec lets us write basic programs, although it’s not a full production language.

---

### Case Study: Graphics in Unicon

**Key Idea:**  
Unicon integrates 2D and 3D graphics features as built-in language constructs, not as external libraries. This contrasts with many languages that rely on foreign APIs.

---

#### 2D Graphics in Unicon

- Added late to Icon/Unicon to support rapid prototyping of visualization tools.
- Minimal syntax changes:
  - Introduced &window and other keywords.
  - Treated windows as a subtype of file for consistent I/O operations.
- Result: A single, unified abstraction (a window) that supports both text and graphical output.
- Input events (mouse, keyboard) integrated consistently.

**Design Philosophy:**  
Preserve language simplicity, avoid forcing callback-driven control flow common in typical GUI frameworks. The runtime checks events periodically, maintaining a straightforward program structure.

---

#### 3D Graphics in Unicon

- Built on top of 2D graphics concepts.
- Introduced a display list for 3D scenes.
- Similar abstractions (e.g., color, fonts) extended naturally into 3D.
- Support for cameras, transformations, and interactive 3D elements.

**Challenges:**
- Retaining simplicity while adding complexity of 3D environments.
- Balancing minimalism with functionality that leverages 3D hardware acceleration.

---

### Summary

**Key Takeaways:**
- Composite types shape how programmers build complex data structures.
- Domain-specific types can dramatically simplify code in specialized areas.
- Overall program structure influences simplicity, linking, and runtime behavior.
- Historical examples (C vs. Pascal) and modern additions (Unicon’s graphics) illustrate the trade-offs in language design.
- A thoughtful design phase saves time and effort during implementation.

---

### Questions

1. **Reserved Words:**
   - What are pros and cons of many reserved words?  
   **Pro:** More expressive, self-documenting constructs.  
   **Con:** Increased complexity, potential name clashes, harder to learn.

2. **Lexical Rules Complexity:**
   - Even simple integer literals can cause complexity. Consider:  
     - Hex, octal, binary forms.  
     - Large numeric suffixes (e.g., `L`, `LL`, `U`).  
     - Making them easy to parse in a lexical analyzer is non-trivial.

3. **Eliminating Semicolons:**
   - Use newline-sensitive parsing (like Python).
   - Expression-oriented languages where line breaks separate statements.
   - Automatic semicolon insertion (JavaScript) or block indentation (Haskell).

4. **Multiple main() Methods (Java-Style):**
   - Allows entry points in multiple classes.  
   - Facilitates testing or choosing different start points.  
   - Adds flexibility but might confuse newcomers.

5. **Pre-opened I/O Resources:**
   - In a modern environment, consider pre-opened “standard” network or GUI streams.  
   - Practicality depends on domain:
     - General languages: maybe not.  
     - Domain-specific: pre-opened graphics windows could be useful.
   - Trade-off: convenience vs. complexity and portability.

---

**End of Notes.**