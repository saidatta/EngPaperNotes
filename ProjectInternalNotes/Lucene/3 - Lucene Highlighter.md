Absolutely, here's an enriched version of the notes, incorporating the code examples and further relevant details.

---

# Lucene Highlighting Mechanics

## Background Introduction
- A search engine typically highlights keywords in results.
- Example: Searching "lucene engine" in Google results in highlighted keywords in red.
- Objective: Dive into the mechanics behind this highlighting.

## Self-Questioning
- **Prompt**: How would you design the highlighting function?
- **Straightforward Answer**:
  1. Traverse through returned documents.
  2. Spot matching keywords to highlight.
  3. Display fragments with a high count of keyword matches.
  4. More matches = More relevance.

## Steps for Highlighting Implementation
1. **Find matches**.
2. **Highlight matches**.
3. **Crop highlighted segments**.
4. **Retrieve fragments with many matches**.

**Note**: Lucene targets these exact challenges for highlighting.
## Summary
- Lucene has three keyword highlighting implementations:
  1. Highlighter
  2. FastVectorHighlighter
  3. UnifiedHighlighter
- **Today's Focus**: **Highlighter**
## Highlighter Mechanics
- Oldest tool in Lucene for highlighting.
- Relies on four key components:
  1. **Fragmenter**: Dictates each highlighted fragment's length.
  2. **Encoder**: Enables escaping content within highlighted segments.
  3. **Scorer**: Grants scores to each highlighted fragment; results get arranged by score.
  4. **Formatter**: Decides how matching keywords get highlighted (e.g., bold or color).

```java
public class HighlighterDemo {
    public static void main(String[] args) throws IOException, InvalidTokenOffsetsException {
        StandardAnalyzer analyzer = new StandardAnalyzer();
        PhraseQuery phraseQuery = new PhraseQuery(1, "field0", "lucene", "search");
        SimpleHTMLFormatter formatter = new SimpleHTMLFormatter();
        DefaultEncoder encoder = new DefaultEncoder();
        QueryTermScorer scorer = new QueryTermScorer(phraseQuery);
        Highlighter highlighter = new Highlighter(formatter, encoder, scorer);
        SimpleFragmenter fragmenter = new SimpleFragmenter(10);
        highlighter.setTextFragmenter(fragmenter);
        String[] fragments = highlighter.getBestFragments(analyzer, "field0", "The goal of Apache Lucene is to provide world class search capabilities.", 5);
        for(String fragment: fragments) {
            System.out.println(fragment);
        }
    }
}
```
**Output**:
```css
The goal of Apache <B>Lucene</B> is to provide
class <B>search</B>
```
## Preliminary Knowledge: Understanding TokenStream
- **TokenStream**: Enumerates the token (or term) sequence of text. Essentially, it's a tokenizer.
- During word segmentation: Can retrieve token's content, its position, offset info, etc.
- Each token attribute is an "Attribute" in Lucene. 
- Different tokenizers = Different attribute sets.

```java
public class TokenStreamDemo {
    public static void main(String[] args) throws IOException {
        WhitespaceAnalyzer analyzer = new WhitespaceAnalyzer();
        TokenStream tokenStream = analyzer.tokenStream("test", "My name is zjc, what's your name.");
        OffsetAttribute offsetAttribute = tokenStream.addAttribute(OffsetAttribute.class);
        CharTermAttribute charTermAttribute = tokenStream.addAttribute(CharTermAttribute.class);
        tokenStream.reset();
        while (tokenStream.incrementToken()) {
            String token = charTermAttribute.toString();
            int startOffset = offsetAttribute.startOffset();
            int endOffset = offsetAttribute.endOffset();
            System.out.println(token + ": " + startOffset + ", " + endOffset);
        }
    }
}
```
**Output**:
```makefile
My: 0, 2
name: 3, 7
is: 8, 10
zjc,: 11, 15
what's: 16, 22
your: 23, 27
name.: 28, 33
```

For TokenStream, key takeaway: Register attributes pre-tokenizer execution. After `incrementToken`, these attributes can be fetched.

---
#### TokenGroup
- **Description**: Represents a token group containing one or multiple overlapping tokens. Overlapping tokens usually arise from synonyms. Generally, a TokenGroup has only one token.
- **Highlighting Criteria**: Based on TokenGroup. If the total score in TokenGroup is > 0 (indicating a keyword match), the whole group is highlighted.
- **Member Variables**:
  - Max number of tokens a group can have: `MAX_NUM_TOKENS_PER_GROUP = 50`
  - Scores array for each token: `scores[]`
  - Number of tokens in the group: `numTokens`
  - Start and end positions of the group in the text: `startOffset`, `endOffset`
  - Total score of the group: `tot`
  - Start and end positions of the matching token in the text: `matchStartOffset`, `matchEndOffset`
  - Offset attribute for current token: `offsetAtt`
- **Key Methods**:
  1. Constructor mainly registers offset and term attributes.
  2. `addToken(float score)` method to add token score.
  3. `isDistinct()` checks if the current token overlaps with other tokens in the group.
#### QueryTermExtractor
- **Purpose**: Extracts all terms from the query and assigns a weight to each term.
- **Core Methods**:
  1. `getTerms(Query query, boolean prohibited, String fieldName)` which traverses the query tree and assigns weights to terms.
  2. `BoostedTermExtractor` class, a visitor pattern that helps in traversing the query tree.
  3. If the reader is available, `getIdfWeightedTerms(Query query, IndexReader reader, String fieldName)` calculates the idf as the weight of the term.
#### TextFragment
- **Description**: Represents a highlighted segment. It encapsulates a text sequence and uses start and end positions to denote a section as highlighted.
- **Member Variables**:
  - Text sequence from which a highlighted fragment is a subset: `markedUpText`
  - Highlighted fragment number: `fragNum`
  - Start and end positions of the highlighted fragment in text: `textStartPos`, `textEndPos`
  - Score of the highlighted fragment: `score`

- **Main Methods**:
  1. `merge(TextFragment frag2)` merges two adjacent highlighted fragments.
  2. `follows(TextFragment fragment)` checks if the current highlighted fragment is adjacent to another.
  3. `toString()` returns the actual highlighted fragment.

#### Four Major Highlighter Components:
1. **Formatter**:
   - **Role**: Formats the matching token found in the highlighted segment, typically using HTML tags.
   - **Interface Method**: 
     ```java
     String highlightTerm(String originalText, TokenGroup tokenGroup);
     ```
   - **Examples**:
     - `SimpleHTMLFormatter`: Adds bold HTML tags by default.
     - `GradientFormatter`: Colors highlighted fragments based on their TokenGroup score.
     - `SpanGradientFormatter`: Same as GradientFormatter but uses the span tag for highlighting.
  
2. **Encoder**:
   - **Role**: Encodes the highlighted text. 
   - **Examples**:
     - `DefaultEncoder`: Returns text as-is.
     - `SimpleHTMLEncoder`: Escapes HTML special characters.
  
3. **Scorer**:
   - **Role**: Gives scores to highlighted fragments.
   - **Interface Methods**:
     1. `init(TokenStream tokenStream)`
     2. `startFragment(TextFragment newFragment)`
     3. `getTokenScore()`
     4. `getFragmentScore()`

---
**Obsidian Notes on Lucene's QueryTermScorer and Highlighter Mechanism**

---
### 📌 QueryTermScorer 

- **Definition**: 
    - Only considers the match of a single term across all queries.
    - Ignores positional information between terms.
- **Limitations**:
    - Doesn't rewrite queries.
    - Prefix matching and regex matching aren't supported.
    - This is the culprit for problem 2 in the example: inability to find a phrase match where the distance between terms ("lucene" & "search") doesn't exceed 1.
    
- **Key Variables**:
    - `currentTextFragment`: The highlighted fragment being processed.
    - `uniqueTermsInFragment`: Set of terms in the current highlighted fragment.
    - `totalScore`: The aggregate score.
    - `maxTermWeight`: Maximum weight of any term.
    - `termsToFind`: Cache mapping from term to `WeightedTerm`. This is essentially the score of a term.
    - `termAtt`: Attribute for term.

- **Constructors**:
    - There are different construction methods that extract terms from the provided query, using `QueryTermExtractor`. They can also compute term idf as a weight.

- **Scoring methods**:
    - `getTokenScore()`: Returns the score of an individual token.
    - `getFragmentScore()`: Retrieves the score of a highlighted fragment.

---

### 📌 QueryScorer

- More versatile than `QueryTermScorer`.
- Supports all types of queries.
- Converts `PhraseQuery` and `MultiPhraseQuery` into `SpanQuery`.
- Only highlights term groups that meet the slop distance requirement.
- **Note**: The in-depth mechanism is more involved and thus isn't elaborated here.

---

### 📌 Fragmenter

- **Role**: Determines if a token from a `TokenStream` belongs to a new text segment.

- **Methods**:
    - `start()`: Registers the attributes of the tokenizer.
    - `isNewFragment()`: Checks if the current token denotes the start of a new fragment.

- **Implementations**:
    - `NullFragmenter`: Does not segment.
    - `SimpleFragmenter`: Segments text according to a fixed number of tokens.
    - `SimpleSpanFragmenter`: Builds upon `SimpleFragmenter`. If a `SpanQuery` meets location conditions, it ensures all corresponding terms are included in one highlighted fragment.

---

### 📌 Highlighter Main Logic

- **Role**: Contains the core logic of the highlighting process.
- **Methods**:
    - `getBestFragments()`: Retrieves the best fragments for highlighting based on scoring.

- **Four Major Steps in `getBestTextFragments` method**:
    1. Retrieve all fragments.
    2. Process the last fragment.
    3. Sort fragments by score.
    4. Merge contiguous fragments.

- **Highlighting process**:
    - Every token is iterated over.
    - For each token group that's distinct and contains terms, the text is highlighted.
    - If a new fragment starts, the current fragment's score is set, and a new fragment is initialized.

---
### Examples & Code snippets
**Example**: The highlighted piece of text between "lucene" and "search" with a distance of <= 1.

```java
TextFragment currentTextFragment = null;
HashSet<String> uniqueTermsInFragment;
float totalScore = 0;
float maxTermWeight = 0;
private HashMap<String, WeightedTerm> termsToFind;
private CharTermAttribute termAtt;
```

**Code**: Extracting terms from a query using `QueryTermExtractor`.

```java
public QueryTermScorer(Query query) {
    this(QueryTermExtractor.getTerms(query));
}
```

**Code**: Checking if a new fragment has started in the `SimpleFragmenter`.

```java
public boolean isNewFragment() {
    boolean isNewFrag = offsetAtt.endOffset() >= (fragmentSize * currentNumFrags);
    if (isNewFrag) {
        currentNumFrags++;
    }
    return isNewFrag;
}
```
This is a brief breakdown of the provided Java code and the accompanying explanation:

1. **Sorting Fragments by Score**:
    - The code starts with a method `getBestTextFragments` that presumably gets the best text fragments after tokenizing the input text.
    - Within this method, fragments are added to a priority queue (`fragQueue`) for sorting based on their score. 
    - After sorting, fragments are popped from the priority queue and stored in an array, in descending order of scores.

2. **Merging Contiguous Fragments**:
    - If the `mergeContiguousFragments` flag is true, the code checks for contiguous fragments and merges them.
    - After merging, fragments with a score greater than 0 are added to an ArrayList (`fragTexts`).
    - The ArrayList is converted to an array of `TextFragment` for the final return.

3. **Handling TokenStream**:
    - After the primary logic of the method, it ensures that the `tokenStream` is ended and closed. If any exception occurs during this process, it's caught and ignored.

4. **Summary from the Article**:
    - The Highlighter works by tokenizing documents and then highlighting keywords based on their relevance. 
    - It's noted that Highlighter is inefficient for large documents since it traverses tokens to identify keywords.
    - An issue with the Highlighter is its logic for merging end-to-end fragments, which may include fragments that don't contain keywords, potentially diluting the overall relevance of the results.

5. **Authorship Note**: 
    - The content is written by "沧叔解码".
    - The link provided leads to a post on the website "稀土掘金" (Juejin).
    - The author's rights are acknowledged, with a note about commercial vs. non-commercial use.

From the analysis, this code snippet and explanation give insights into the workings of a highlighting mechanism (presumably from the Lucene library or similar), detailing the process of sorting, merging, and token handling. The accompanying summary provides a critique of the Highlighter, particularly in its inefficiency with larger documents and potential dilution of relevance when merging contiguous fragments.