### Topics Covered
- Product rule and chain rule of derivatives of multiple and embedded functions.

### Introduction
- Derivatives of simple functions are straightforward.
- Computing derivatives of more complicated functions (interacting functions) becomes more complicated and requires more practice.
- Deep learning libraries like PyTorch handle the calculation of complicated derivatives efficiently and accurately.

### Product Rule
- The product rule states that the derivative of two multiplied functions (F(x) and G(x)) is equal to the derivative of the first times the second function plus the first function times the derivative of the second.

Equation: (F * G)' = F' * G + F * G'

### Chain Rule
- The chain rule is used when one function is embedded inside another function (F(G(x))).
- The chain rule states that the derivative of F(G(x)) with respect to x is the derivative of F with G still inside, multiplied by the derivative of G.

Equation: (F(G(x)))' = F'(G(x)) * G'(x)

### Example
- Given F(x) = (x^2 + 4x^3)^5
- G(x) = x^2 + 4x^3
- F(G(x)) = (x^2 + 4x^3)^5
- Applying the chain rule: (F(G(x)))' = 5 * (x^2 + 4x^3)^4 * (2x + 12x^2)

### Python Implementation
- Python libraries like NumPy and SymPy can be used to compute derivatives and implement the product and chain rules.
- The IPython.display library can be used to display SymPy expressions in a more readable format.
### Conclusion
- The product rule and chain rule are essential tools for computing derivatives of more complicated functions.
- Deep learning libraries like PyTorch can handle these calculations efficiently and accurately, allowing you to focus on more important conceptual aspects of gradient descent algorithms.