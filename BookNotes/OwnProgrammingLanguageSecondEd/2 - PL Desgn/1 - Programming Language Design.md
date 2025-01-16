**Source:** Snippets from “Programming Language Design” (Chapter 2 excerpt)

**Context:** These notes are geared towards experienced engineers who are looking to design their own programming languages. The notes will discuss surface-level design choices, lexical and syntactical rules, semantics, control flow constructs, data types, and overall language structure.

---
### Table of Contents
- [[#Defining a Language|Defining a Language]]
- [[#Lexical vs Syntax Rules|Lexical vs Syntax Rules]]
- [[#Language Design Document|Language Design Document]]
- [[#Determining Words and Punctuation|Determining Words and Punctuation]]
- [[#Specifying Control Flow|Specifying Control Flow]]
- [[#Data Types|Data Types]]
- [[#Example Constructs in Jzero|Example Constructs in Jzero]]
- [[#Case Study: Graphics in Unicon|Case Study: Graphics in Unicon]] (mentioned but not fully elaborated here)

---

### Defining a Language
Before building a programming language, you must **define** it. This definition involves:

1. **Lexical rules:** Basic rules for forming words, punctuation, and symbols.
2. **Syntax rules:** Higher-level rules to structure these lexical elements into expressions, statements, and larger constructs (functions, classes, modules).

**Goal:** Produce a coherent language specification that is both understandable and implementable.

---
![[Screenshot 2024-12-19 at 3.39.02 PM.png]]
### Lexical vs Syntax Rules

**Lexical rules**:
- Govern how characters form valid tokens (identifiers, numbers, strings, operators).
- Examples:
  - Defining what characters constitute an identifier (`[A-Za-z_][A-Za-z0-9_]*` in many languages).
  - Determining how to form numeric literals (`0xFF`, `3.14f`, etc.).
  
**Syntax rules**:
- Govern how tokens combine to form valid statements, expressions, and program units.
- Usually expressed via grammar rules (Backus–Naur Form, EBNF, etc.).

---

### Language Design Document
**Why create a specification?**  
A language specification (or design doc) guides the implementation. It includes:

- A list of reserved keywords.
- Patterns for identifiers.
- Lexical structures for literals.
- Operator precedence and associativity rules.
- Syntax for control structures, declarations, and modules.

This document evolves over time, but having an initial blueprint reduces confusion during implementation.

---

### Determining Words and Punctuation

**Categories to define:**
1. **Reserved words (keywords)**:
   - E.g., `if`, `else`, `while`, `return`.
   - Provide a complete list in the spec.

2. **Identifiers**:
   - Rules on starting characters, allowed symbols, and case sensitivity.
   - Example pattern:
     ```regex
     Identifier: [A-Za-z_][A-Za-z0-9_]*
     ```
   - Consider locale/Unicode rules if needed.

3. **Literals**:
   - Numeric literals (integers, floats, doubles, hex, binary).
   - String literals (with escape sequences).
   - Boolean literals (`true`, `false`).
   - Example numeric literal in a Java-like language:
     ```plaintext
     float_literal:   [0-9]+\.[0-9]*([eE][+\-]?[0-9]+)?[fF]?
                     | \.[0-9]+([eE][+\-]?[0-9]+)?[fF]?
                     | [0-9]+[eE][+\-]?[0-9]+[fF]?
     ```

4. **Operators & punctuation**:
   - Single-character operators: `+`, `-`, `*`, `/`, `%`.
   - Multi-character operators: `++`, `--`, `==`, `!=`.
   - Punctuation: `;`, `,`, `(`, `)`, `{`, `}`, `[`, `]`.
   - Define precedence and associativity rules carefully.

**Example Operator Precedence Table** (inspired by Java):
- Level 1 (lowest): `=` (assignment)
- Level 2: `||` (logical OR)
- Level 3: `&&` (logical AND)
- Level 4: `==`, `!=` (equality, inequality)
- Level 5: `<`, `>`, `<=`, `>=`
- Level 6: `+`, `-`
- Level 7: `*`, `/`, `%`
- Level 8: `!`, unary `+`, unary `-`
- Level 9: `++`, `--` (postfix/prefix)
- Level 10 (highest): function calls, array indexing

*(Note: This is simplified and not exact. Actual Java operator precedence is more detailed.)*

---

### Specifying Control Flow

**Common Constructs:**
- Conditionals: 
  ```c
  if (condition) statement;
  if (condition) statement1 else statement2;
  ```
- Loops:
  ```c
  while (condition) statement;
  for (initialization; condition; increment) statement;
  ```

**Design Decisions:**
- Are `if` and `while` statements or expressions?
- Support `switch`? `do-while`?
- Introducing domain-specific control structures?

**Maintaining Familiarity:**
- Stick to well-known patterns (`if`, `while`) unless innovation is required.
- Avoid surprising precedence or unusual keywords that confuse experienced programmers.

---

### Data Types

**Categories:**
1. **Atomic (scalar) types**:
   - Primitive numeric types: `int`, `float`, `double`.
   - Booleans: `bool`.
   - Possibly `string` as an atomic type (immutable).
   
   ```c
   int x = 42;
   float y = 3.14f;
   bool flag = true;
   string s = "Hello, World!";
   ```

2. **Composite (container) types**:
   - Arrays, lists, maps, objects, structs/classes.
   - Syntax for declarations and initializations.
   
   ```c
   int arr[10];       // static array
   List<int> numbers; // generic list type
   ```

3. **Domain-specific types**:
   - For specialized languages (e.g., financial DSLs, graphics, simulation).
   - Introduce special literals or operators as needed.

**Questions to answer**:
- How many numeric types to support?
- How to handle strings and their encodings (ASCII, UTF-8)?
- Are arrays or objects first-class, or constructed via libraries?

---

### Example Constructs in Jzero

**Context:** Jzero is a simplified language subset from the text.  
**Example:**
```c
// Control flow in Jzero
if (x < 10) {
    x = x + 1;
} else {
    x = 0;
}

while (x < 100) {
    x = x * 2;
}

// Function call example
result = computeSomething(x, 2.5);
```

- No `switch`, no `do-while`.
- Precedence and associativity similar to Java.
- Simple `if`, `if-else`, `while`, and `for` loops.

---

### Notes on Operator Associativity

- **Left-to-right** (most common): `x + y + z` = `(x + y) + z`
- **Right-to-left**: `x = y = 0` = `x = (y = 0)`

**Tip:** Use established conventions to reduce learning overhead.

---

### Additional Considerations

- **Error Handling:**
  - Produce clear compiler errors if unsupported constructs appear.
  - Example: If `switch` is not supported:
    ```plaintext
    error: 'switch' construct not supported in this language.
    ```
  
- **Simplicity vs. Power:**
  - More types and constructs → more complexity, harder to implement.
  - Fewer constructs → less expressive but easier to learn and maintain.

---

### Next Steps

After defining lexical tokens and basic syntax, proceed to:

- [[#Semantic Rules|Semantic rules]]: Define meaning (type-checking, scoping, runtime behavior).
- Implementing a [[Lexer]] and [[Parser]] for your language.
- Writing more advanced examples that leverage all features.
  
*(These steps are beyond the provided snippet but are natural continuations.)*

---

### Case Study: Graphics in Unicon

*(Brief Mention)*

- Unicon extended Icon with graphics facilities.
- Domain-specific types and operators integrated at the syntax level.
- Consider how domain-specific features inform lexical/syntax rules and semantics.

---

### Summary

**Key Takeaways:**
- Start language design by specifying lexical and syntax rules.
- Choose a familiar set of keywords, operators, and punctuation to minimize surprises.
- Carefully define control flow structures and their syntax to support common patterns (if/else, loops).
- Decide which data types to support (atomic, composite, domain-specific) and how they integrate into your language.
- Document your decisions in a specification to guide implementation and future changes.

---

**Related Links:**
- [[Language Semantics]]
- [[Lexical Analysis]]
- [[Parsing Techniques]]
- [[Operator Precedence and Parsing]]

---

**Callout – Advice**  
> [!TIP] **Principle of Least Surprise**  
> When inventing new features or operators, stick to familiar conventions whenever possible. If you must deviate, document your design rationale and test it with sample code.

---

End of Notes.