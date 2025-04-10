**Context:**  
The Jzero scanner needs to integrate with a parser that provides integer category codes for tokens. In Unicon, we simulate Java’s constants by creating a `parser` record that stores token categories. We also expand the token data structure to include column numbers and handle string escapes, ensuring compatibility with both Unicon and Java approaches.

---

### Overview

- **Goal:** Create a scanner for Jzero, a Java subset, using Unicon.
- **Key Features:**
  - Track line and column numbers.
  - Create tokens with lexical attributes (cat, text, lineno, colno).
  - Handle comments, whitespace, and newline adjustments.
  - Convert lexemes for literal constants into their binary representations.
  - Implement `ord()`, `scan()`, `whitespace()`, `newline()`, and `comment()` methods.

---

### Unicon Main Program: `j0.icn`

**Main procedure:**

```unicon
global yylineno, yycolno, yylval
procedure main(argv)
   j0 := j0()
   parser := parser(257,258,259,260,261,262,263,264,265,
                    266,267,268,269,270,273,274,275,276,
                    277,278,280,298,300,301,302,303,304,
                    306,307,256)
   yyin := open(argv[1]) | stop("usage: simple2 filename")
   yylineno := yycolno := 1
   while i := yylex() do
      write("token ", i, ":", yylval.lineno, " ", yytext)
end
```

**Notes:**
- `parser` record holds token category codes, mimicking Java constants.
- `yylineno` and `yycolno` track current line and column.
- The `yylex()` function is generated by UFlex from `javalex.l`.
- For each token recognized, print the category, line, and lexeme.

---

### `j0` Class

```unicon
class j0()
   method lexErr(s)
      stop(s, ": ", yytext) 
   end

   method scan(cat)
      yylval := token(cat, yytext, yylineno, yycolno)
      yycolno +:= *yytext
      return cat
   end

   method whitespace()
      yycolno +:= *yytext
   end

   method newline()
      yylineno +:= 1; yycolno := 1
   end

   method comment()
      yytext ? {
         while tab(find("\n")+1) do newline()
         yycolno +:= *tab(0)
      }
   end

   method ord(s)
      return proc("ord",0)(s[1])
   end
end
```

**Key Points:**
- `scan(cat)`: Creates a token and updates the column counter.
- `whitespace()`: Advances `yycolno` by length of whitespace.
- `newline()`: Increments line, resets column.
- `comment()`: Updates line/column counts for multi-line comments.
- `ord(s)`: Returns ASCII code of the character `s[1]`.

---

### Token Class

The token now includes more logic, including de-escaping string literals.

```unicon
class token(cat, text, lineno, colno, ival, dval, sval)
   method deEscape(sin)
      local sout := ""
      sin := sin[2:-1]
      sin ? {
         while c := move(1) do {
            if c == "\\" then {
               if not (c := move(1)) then
                  j0.lexErr("malformed string literal")
               else case c of {
                  "t": { sout ||:= "\t" }
                  "n": { sout ||:= "\n" }
                  default: j0.lexErr("unrecognized escape")
               }
            } else
               sout ||:= c
         }
      }
      return sout
   end

initially
   case cat of {
     parser.INTLIT:    { ival := integer(text) }
     parser.DOUBLELIT: { dval := real(text) }
     parser.STRINGLIT: { sval := deEscape(text) }
   }
end
```

**Highlights:**
- `deEscape()` handles string escapes like `\t` and `\n`.
- On initialization, convert literals to internal representations.

---

### Parser Record (for Unicon Compatibility)

```unicon
record parser(BREAK,PUBLIC,DOUBLE,ELSE,FOR,IF,INT,RETURN,VOID,
             WHILE,IDENTIFIER,CLASSNAME,CLASS,STATIC,STRING,
             BOOL,INTLIT,DOUBLELIT,STRINGLIT,BOOLLIT,
             NULLVAL,LESSTHANOREQUAL,GREATERTHANOREQUAL,
             ISEQUALTO,NOTEQUALTO,LOGICALAND,LOGICALOR,
             INCREMENT,DECREMENT,YYERRCODE)
```

**Explanation:**
- Using a `parser` record to hold integer codes is a workaround to maintain a common lex spec with Java.

---

## Java Jzero Code

**Context:**
The Java implementation mirrors the Unicon version. It uses classes for tokens and parser codes defined as `public static final short`. The scanning logic is similar, but in Java syntax.

---

### `j0.java` (Main Class)

```java
import java.io.FileReader;
public class j0 {
   static Yylex lex;
   public static int yylineno, yycolno;
   public static token yylval;

   public static void main(String argv[]) throws Exception {
      lex = new Yylex(new FileReader(argv[0]));
      yylineno = yycolno = 1;
      int i;
      while ((i=lex.yylex()) != Yylex.YYEOF) {
         System.out.println("token " + i + ":" + yylineno + " " + yytext());
      }
   }

   public static String yytext() {
      return lex.yytext();
   }

   public static void lexErr(String s) {
      System.err.println(s + ": line " + yylineno + ": " + yytext());
      System.exit(1);
   }

   public static int scan(int cat) {
      yylval = new token(cat, yytext(), yylineno, yycolno);
      yycolno += yytext().length();
      return cat;
   }

   public static void whitespace() {
      yycolno += yytext().length();
   }

   public static void newline() {
      yylineno++; yycolno = 1;
   }

   public static void comment() {
      String s = yytext();
      for (int i=0; i<s.length(); i++)
         if (s.charAt(i) == '\n') {
            yylineno++; yycolno=1;
         } else yycolno++;
   }

   public short ord(String s) {
      return (short)(s.charAt(0));
   }
}
```

---

### `parser.java`

```java
public class parser {
  public final static short BREAK=257;
  public final static short PUBLIC=258;
  public final static short DOUBLE=259;
  ...
  public final static short INCREMENT=306;
  public final static short DECREMENT=307;
  public final static short YYERRCODE=256;
}
```

**ASCII Diagram:**

```ascii
+----------+       +-----------+
|  source  | ----> |  scanner  | ---> tokens (yylval)
+----------+       +-----------+
                         |
                      parser codes
                         |
                       parser
```

---

### `token.java`

```java
public class token {
   public int cat;
   public String text;
   public int lineno, colno;
   int ival;
   double dval;
   String sval;

   private String deEscape(String sin) {
      String sout = "";
      sin = sin.substring(1,sin.length()-1);  // remove quotes
      while (sin.length() > 0) {
         char c = sin.charAt(0);
         sin = sin.substring(1);
         if (c == '\\') {
            if (sin.length()<1) j0.lexErr("malformed string literal");
            c = sin.charAt(0);
            sin = sin.substring(1);
            switch(c) {
               case 't': sout += "\t"; break;
               case 'n': sout += "\n"; break;
               default: j0.lexErr("unrecognized escape");
            }
         } else {
            sout += c;
         }
      }
      return sout;
   }

   public token(int c, String s, int ln, int col) {
      cat = c; text = s; lineno = ln; colno = col;
      switch (cat) {
         case parser.INTLIT:    ival = Integer.parseInt(s); break;
         case parser.DOUBLELIT: dval = Double.parseDouble(s); break;
         case parser.STRINGLIT: sval = deEscape(s); break;
      }
   }
}
```

---

### Running the Jzero Scanner

**Input (`hello.java`):**
```java
public class hello {
   public static void main(String argv[]) {
      System.out.println("hello, jzero!");
   }
}
```

**Commands:**

- Unicon:
  ```bash
  uflex javalex.l
  unicon j0 javalex
  j0 hello.java
  ```

- Java:
  ```bash
  jflex javalex.l
  javac j0.java Yylex.java token.java parser.java
  java j0 hello.java
  ```

**Expected Output:**
```plaintext
token 258:1 public
token 269:1 class
token 267:1 hello
token 123:1 {
token 258:2 public
...
token 125:5 }
```

---

### Regular Expressions Are Not Always Enough

**Key Insight:**
- Sometimes, language features (like semicolon insertion) cannot be expressed purely with regex-based rules.
- May require extra logic, memory of the previous token, or even one-token lookahead.
- Example: Go language’s semicolon insertion rules.
- Unicon/Go approach: after newline, decide if a semicolon should be inserted based on the previous token.

**ASCII Note:**
```ascii
+-----------------------+
|   Lex Specification   |
|  (regular expressions)|
+-----------+-----------+
            |
            V
          Scanner
            |
            V
        Tokens/Actions
          (some logic beyond RE)
```

---

### Summary

- We integrated the scanner with complex token data structures in both Unicon and Java.
- We added line/column tracking and logic for whitespace, comments, and string escapes.
- We learned that while regular expressions handle most lexical tasks, certain language features (like semicolon insertion) need extra logic.
- With the scanner now producing a stream of tokens, we’re ready to move on to parsing, where the grammar and syntax rules come into play.

---

**Next Steps:**
- Proceed to parsing, where tokens form syntactic constructs.
- Consider operator precedence and grammar design.
- Integrate the scanner fully with the parser to build and analyze syntax trees.

---

**Related Links:**
- [[Parsing Techniques]]
- [[Compiler Construction Basics]]
- [[Error Handling in Lexical Analysis]]

---

> [!TIP]
> Always test your scanner on various inputs, including tricky cases (unclosed strings, nested comments, unusual whitespace), to ensure robust tokenization and correct error messages.

---

End of Notes.