Elasticsearch is commonly known for full-text search but has evolved to support vector similarity search, an essential feature for semantic search. By utilizing the `dense_vector` data type and leveraging `script_score`, Elasticsearch allows for powerful vector-based search functionalities. This note covers how Elasticsearch can be used as a vector database for semantic search, deep-diving into the internals of vector search, the dense vector data type, and the script_score function.

---
### **1. Semantic Search vs. Syntactic Search**
Before diving into vector search, it's crucial to understand the difference between syntactic search and semantic search.

#### **Syntactic Search:**
- Searches for documents based on the exact words present in the query.
- Limited to matching exact words, which might result in missing contextually relevant documents.
- Example:
  - Query: `"apple alcoholic beverage"`
  - Results: Documents containing the words "apple", "alcoholic", and "beverage" exactly.

#### **Semantic Search:**
- Uses vector representations of words, phrases, or sentences to understand the meaning or intent behind the query.
- Finds semantically related content, even if the exact words do not match.
- Example:
  - Query: `"apple alcoholic beverage"`
  - Results: Documents related to "apple brandy", "apple bourbon", or "appletini", based on vector similarity.

### **2. Vector Search Mechanism**

In vector search, every word, phrase, or sentence is represented as a high-dimensional vector using embedding models such as **Word2Vec**, **BERT**, or **FastText**. The proximity between vectors represents the semantic similarity between different queries and documents.

#### **Steps in Vector Search**:
1. **Query Transformation**: 
   - The search query is transformed into a vector using an embedding model.
   
2. **Distance Calculation**:
   - The similarity between the query vector and document vectors is computed using metrics such as **cosine similarity** or **dot product**.

3. **Ranking**:
   - The documents are ranked based on their similarity to the query vector, and those with the closest vectors are returned as the top results.

---

### **3. Vector Space in Embedding Models**

A **vector space** is a mathematical structure where vectors represent data points. For example, words or sentences in Natural Language Processing (NLP) are mapped to vectors in a high-dimensional space where semantic relationships can be computed.

#### **Key Concepts**:
- **Dimensionality**: 
  - Each dimension represents a feature or property of the data. More dimensions allow capturing more details but increase computational complexity.
  
- **Distance and Similarity**:
  - **Cosine similarity** or **Euclidean distance** is used to compute how similar two vectors are.
  
- **Contextual Learning**: 
  - Embedding models like **Word2Vec** and **BERT** learn from vast amounts of text data to capture the contextual meaning of words or sentences.

#### **Example**:
In a well-trained vector space, the operation:
```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```
This demonstrates how relational semantics are captured in embedding models.

---

### **4. Dense Vector Data Type in Elasticsearch**

Elasticsearch's `dense_vector` data type allows the storage of high-dimensional vectors in the database. These vectors are often produced by machine learning models and used for tasks such as semantic search.

#### **Dense Vector Mapping** Example:
```json
PUT /my_index
{
  "mappings": {
    "properties": {
      "text_vector": {
        "type": "dense_vector",
        "dims": 512
      }
    }
  }
}
```
- **dims**: The number of dimensions (features) in the vector, typically depending on the embedding model (e.g., BERT produces 512-dimensional vectors).

---

### **5. Script Score for Vector Similarity Search**

The **script_score** function in Elasticsearch is used to calculate a custom similarity score between a query vector and the vectors stored in the database. The **dot product** is a common method for measuring the similarity of vectors.

#### **Script Score Query Example**:
```json
POST /my_index/_search
{
  "query": {
    "script_score": {
      "query": {
        "match_all": {}
      },
      "script": {
        "source": "dotProduct(params.queryVector, 'text_vector') + 1.0",
        "params": {
          "queryVector": [0.1, 0.2, ..., 0.512]
        }
      }
    }
  }
}
```
- **dotProduct**: Measures the similarity between two vectors. The query vector is compared with the vector stored in the `text_vector` field.
- **+1.0**: Ensures that all score values are positive (since Elasticsearch cannot handle negative scores).

#### **Why Add +1.0?**
Elasticsearch does not support negative score values, so adding `1.0` ensures all returned scores are positive. However, this can distort the results when the dot product is near zero. Post-processing the scores can remove the offset if more precise results are required.

---

### **6. Practical Application: Vector Search for Semantic Matching**

Consider a scenario where you have an e-commerce platform, and users want to search for semantically related products, not just exact matches.

#### **Use Case**: 
A user searches for `"Bluetooth headphones"`, and the system should return products related to wireless audio devices, such as:
- **Sony WH-1000XM4**
- **Bose QuietComfort 35**
- **Apple AirPods Pro**

By using a pre-trained model to generate vector embeddings for these products and storing them in Elasticsearch as `dense_vector`, you can implement a powerful, semantically aware search function.

---

### **7. Advantages of Elasticsearch Over Other Vector Search Libraries**

While specialized vector search libraries like **Faiss** and **ChromaDB** are optimized for speed and efficiency, Elasticsearch offers the advantage of combining vector search with rich query capabilities, including filtering and metadata-based queries.

#### **Key Features**:
- **Complex Filtering**: Elasticsearch allows filtering based on metadata alongside vector similarity search. For example, you can filter results by product category, price range, or availability while ranking by vector similarity.
- **Context-aware Search**: Using Elasticsearch’s versatile query environment, you can layer vector-based search results with more complex, context-aware filters.
  
#### **Example**:
```json
POST /my_index/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "script_score": {
            "script": {
              "source": "dotProduct(params.queryVector, 'text_vector') + 1.0",
              "params": {
                "queryVector": [0.1, 0.2, ..., 0.512]
              }
            }
          }
        },
        {
          "range": {
            "price": {
              "gte": 100,
              "lte": 300
            }
          }
        }
      ]
    }
  }
}
```
This query combines vector similarity with a price range filter, returning products that are similar to the query vector and priced between $100 and $300.

---

### **8. Advances in Elasticsearch Version 8**

Elasticsearch has introduced support for **HNSW** (Hierarchical Navigable Small World) graphs in version 8, significantly improving vector search performance and scalability. **HNSW** is a graph-based algorithm that optimizes approximate nearest neighbor search (ANN), making it faster and more efficient for high-dimensional vector searches.

#### **HNSW** in Elasticsearch:
- Faster search for high-dimensional vectors.
- Improved scalability for large-scale vector searches.
  
---

### **ASCII Diagram: Vector Search Flow in Elasticsearch**

```
+---------------------+
| User Input          |
+---------------------+
          |
          v
+---------------------+
| Embedding Model     |    (e.g., BERT, Word2Vec)
+---------------------+
          |
          v
+---------------------+
| Query Vector        |    (e.g., 512-dimensional vector)
+---------------------+
          |
          v
+---------------------+
| Elasticsearch       |
| (dense_vector field)|
+---------------------+
          |
          v
+---------------------+
| Vector Similarity   |    (dotProduct, cosine similarity)
+---------------------+
          |
          v
+---------------------+
| Search Results      |
| (Ranked by similarity)|
+---------------------+
```

---

### **Conclusion**

Using Elasticsearch as a vector database allows for powerful and efficient semantic search capabilities. By leveraging the `dense_vector` data type, `script_score`, and the newly introduced **HNSW** support in Elasticsearch 8, developers can implement vector search for a variety of applications, ranging from recommendation systems to natural language search in e-commerce and content discovery platforms. The ability to combine vector search with Elasticsearch’s rich query capabilities provides a versatile and comprehensive solution for modern search problems.

