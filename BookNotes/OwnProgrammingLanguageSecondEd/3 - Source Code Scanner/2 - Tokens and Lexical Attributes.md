**Context:**  
So far, we have discussed how scanners identify and categorize lexemes. The next step in building a language implementation (such as a compiler) is to create **tokens**, which are richer data structures containing not just a category (an integer code) but also **lexical attributes** such as the lexeme text, line number, column number, and possibly filename or other metadata.

---
### What Are Tokens?

- **Token:** A data structure representing a single recognized lexeme. It typically includes:
  - **Category:** An integer code (e.g., `IDENTIFIER`, `IF_KEYWORD`, `INTLIT`) describing the token type.
  - **Lexeme string:** The original substring of input that the scanner recognized.
  - **Line and column information:** Where in the source code this token occurred.
  - **Additional semantic info:** For numeric literals, the scanner might store the integer/float value. For strings, it could store the processed string without quotes.

**Example (ASCII visualization):**

```ascii
Source: "int x = 42"
Tokens:
  "int"       -> T_INT       (line 1, col 1)
  "x"         -> T_IDENTIFIER(line 1, col 5)
  "="         -> '='         (line 1, col 7)
  "42"        -> T_INTLIT    (line 1, col 9)
```

---

### Why Store Column and Other Details?

- **Error reporting:**  
  If multiple tokens appear on the same line, columns help pinpoint errors.  
  Example: `) ) )` on the same line - which parenthesis caused the error?

- **Integrated Development Environments (IDEs):**  
  Column info allows the IDE to place the cursor exactly at the offending token for user convenience.

**Decision Point:**  
Storing columns or extra info is optional. It increases complexity but improves user experience and debugging capabilities.

---

### Expanding the Example to Use Tokens

Previously, we returned just an integer category from `yylex()`. Now, we’ll:

1. Allocate a new `token` object each time we recognize a lexeme.
2. Fill the token with the category, text, line number, etc.
3. Return the integer category to the parser, but also store the `token` object in a global variable (`yylval` in many lex-based systems).

**Key Concept:**  
A `scan()` function will simplify token creation. It sets `yylval` to a new token object and returns the integer category code.

---

### Example: `nnws-tok.l`

Below is a revised lex specification that builds on the earlier `nnws.l` example. It adds:

- A `scan()` call that creates tokens.
- Line number tracking with `increment_lineno()`.
- Proper handling of whitespace and newlines.

```lex
%%
%int
%%
[a-zA-Z]+  { return simple2.scan(1); }
[0-9]+     { return simple2.scan(2); }
[ \t]+     { /* ignore whitespace */ }
\r?\n      { simple2.increment_lineno(); }
.          { simple2.lexErr("unrecognized character"); }
```

**Explanation:**

- `[a-zA-Z]+` returns a token of category `1` (e.g., name).
- `[0-9]+` returns a token of category `2` (e.g., number).
- Whitespace is ignored.
- `\r?\n` increments line number each time a newline (with optional carriage return) is encountered.
- `.` catches any unknown character and reports an error.

---

### Unicon Integration: `simple2.icn`

**Key Points:**

- `yylval` is a global variable that will hold the current `token`.
- A `token` record type is defined to store `cat`, `text`, and `lineno`.
- `scan(cat)` creates a new token and assigns it to `yylval`.
- `increment_yylineno()` increments the global `yylineno` count.

**`simple2.icn`:**

```unicon
global yylineno, yylval

procedure main(argv)
   simple2 := simple2()
   yyin := open(argv[1]) | stop("usage: simple2 filename")
   yylineno := 1
   while i := yylex() do
      write("token ", i, " (line ", yylval.lineno, "): ", yytext)
end

class simple2()
   method lexErr(s)
      stop(s, ": line ", yylineno, ": ", yytext)
   end
   method scan(cat)
      yylval := token(cat, yytext, yylineno)
      return cat
   end
   method increment_lineno()
      yylineno +:= 1
   end
end

record token(cat, text, lineno)
```

---

### Java Integration: `simple2.java` and `token.java`

**`simple2.java`:**

```java
import java.io.FileReader;

public class simple2 {
   static Yylex lex;
   public static int yylineno;
   public static token yylval;

   public static void main(String argv[]) throws Exception {
      lex = new Yylex(new FileReader(argv[0]));
      yylineno = 1;
      int i;
      while ((i=lex.yylex()) != Yylex.YYEOF)
         System.out.println("token " + i +
             " (line " + yylval.lineno + "): " + yytext());
   }

   public static String yytext() {
      return lex.yytext();
   }

   public static void lexErr(String s) {
      System.err.println(s + ": line " + yylineno + ": " + yytext());
      System.exit(1);
   }

   public static int scan(int cat) {
      yylval = new token(cat, yytext(), yylineno);
      return cat;
   }

   public static void increment_lineno() {
      yylineno++;
   }
}
```

**`token.java`:**

```java
public class token {
   public int cat;
   public String text;
   public int lineno;

   public token(int c, String s, int l) {
      cat = c; text = s; lineno = l;
   }
}
```

**Explanation:**

- `token` is a simple class holding category, text, and line number.
- `scan()` in Java sets `yylval` before returning the category.
- `increment_lineno()` updates `yylineno`.
- `lexErr()` handles errors with line info.

---

### Testing the Enhanced Scanner

**Input:** `dorrie2.in`

```plaintext
Dorrie
is 1
fine puppy.
```

- The period (`.`) is not recognized as a valid token, so it should trigger an error with line info.

**Running Unicon Version:**

```bash
uflex nnws-tok.l
unicon simple2 nnws-tok
./simple2 dorrie2.in
```

**Running Java Version:**

```bash
jflex nnws-tok.l
javac token.java simple2.java Yylex.java
java simple2 dorrie2.in
```

**Expected Output:**

```plaintext
token 1 (line 1): Dorrie
token 1 (line 2): is
token 2 (line 2): 1
token 1 (line 3): fine
token 1 (line 3): puppy
unrecognized character: line 3: .
```

**Interpretation:**

- The scanner now prints the line number for each token.
- We see a lexical error message that includes the line number.

---

### Beyond the Basics: Jzero’s Scanner

For a real language (like our Jzero subset of Java), the scanner is more complex:

- Includes rules for comments (`/* ... */` and `// ...`).
- Includes reserved words and returns unique category codes for them.
- Recognizes operators, punctuation, and literals (int, double, string).
- Tracks columns in addition to line numbers for finer error localization.
- May assign the same category code to multiple reserved words if they are syntactically identical.

**Conclusion:**

- Storing lexical attributes in tokens is critical for later stages of compilation (parsing, semantic analysis, error reporting).
- The `scan()` function pattern simplifies token creation and keeps your lex specification clean.
- A real-world scanner integrates many complex patterns but uses the same fundamental approach.

---

**Next Steps:**

- Expand token attributes (column numbers, filenames).
- Integrate tokens with a parser to handle syntax rules.
- Consider operator precedence, associativity, and other grammar-level concerns.

---

**Related Links:**

- [[Regular Expressions]]  
- [[Building a Parser]]  
- [[Error Handling in Compilers]]

---

> [!NOTE]  
> Always test your scanner thoroughly with various inputs, including edge cases and invalid characters, to ensure robust error handling and correct tokenization.

---

End of Notes.