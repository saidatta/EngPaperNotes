## Overview

In this chapter, we explore the evolving landscape of real-time data, focusing on the **accelerated convergence of streaming and databases**. This convergence is reshaping how we handle data across different planes—operational, analytical, and streaming. We will delve into:

- The integration of streaming capabilities into various types of databases.
- The movement of operational databases toward the streaming plane.
- The incorporation of streaming functionalities by analytical data platforms.
- The convergence of streaming and lakehouse architectures.

---

## Table of Contents

1. [Introduction](#introduction)
2. [The Convergence of the Data Planes](#the-convergence-of-the-data-planes)
3. [Graph Databases](#graph-databases)
   - [Introduction to Graph Databases](#introduction-to-graph-databases)
   - [Use Cases](#use-cases)
   - [Memgraph](#memgraph)
     - [Direct Ingestion from Streaming Platforms](#direct-ingestion-from-streaming-platforms)
     - [Example: Ingesting Data from Kafka](#example-ingesting-data-from-kafka)
   - [thatDot/Quine](#thatdotquine)
     - [Streaming Graph for Data Pipelines](#streaming-graph-for-data-pipelines)
     - [Example: Ingesting Wikipedia Revision Streams](#example-ingesting-wikipedia-revision-streams)
4. [Vector Databases](#vector-databases)
   - [Introduction to Vector Databases](#introduction-to-vector-databases)
   - [Use Cases](#vector-database-use-cases)
   - [Milvus 2.x: Streaming as the Central Backbone](#milvus-2x-streaming-as-the-central-backbone)
     - [Architecture Overview](#architecture-overview)
   - [RTOLAP Databases Adding Vector Search](#rtolap-databases-adding-vector-search)
     - [ClickHouse](#clickhouse)
       - [Example: Combining Vector Search and Metadata Filtering](#example-combining-vector-search-and-metadata-filtering)
     - [Rockset](#rockset)
       - [Example: Setting Up an ANN Index and Querying](#example-setting-up-an-ann-index-and-querying)
5. [Summary](#summary)
6. [References](#references)
7. [Tags](#tags)

---

## Introduction

The future of real-time data is being shaped by the merging of streaming technologies and databases. This convergence is evident across various database types, including:

- **Graph Databases**: Integrating streaming capabilities to handle dynamic, real-time data.
- **Vector Databases**: Incorporating streaming architectures to manage high-dimensional vector data efficiently.
- **Operational Databases**: Extending functionalities to process streaming data.
- **Analytical Platforms**: Enhancing streaming capabilities to provide real-time analytics.

We will explore how these developments are blurring the lines between the operational, analytical, and streaming data planes.

---

## The Convergence of the Data Planes

![Venn Diagram of Data Planes](venn_diagram_data_planes.png)

*Figure 11-1: Venn Diagram Illustrating the Convergence of the Operational, Analytical, and Streaming Data Planes.*

- **Operational Data Plane**: Traditional databases used in day-to-day operations (e.g., OLTP systems).
- **Analytical Data Plane**: Systems designed for data analysis and insights (e.g., data warehouses).
- **Streaming Data Plane**: Platforms handling data in motion, enabling real-time data processing.

The overlapping areas represent databases and technologies that combine features from multiple planes. For example:

- **Operational + Streaming**: Databases that handle operational data with streaming capabilities.
- **Analytical + Streaming**: Analytical platforms incorporating streaming data processing.

---

## Graph Databases

### Introduction to Graph Databases

**Graph Databases** are specialized databases that use graph structures with nodes, edges, and properties to represent and store data. They excel at managing highly connected data and performing complex queries on relationships.

**Popular Graph Databases**:

- **Neo4j**
- **ArangoDB**
- **TigerGraph**

### Use Cases

1. **Social Networks**:

   - **Application**: Modeling and querying social relationships.
   - **Example**: Users as nodes, relationships like "follows" or "friends" as edges.

2. **Recommendation Systems**:

   - **Application**: Personalizing content or product recommendations.
   - **Example**: Connecting users to products they've interacted with to suggest similar items.

3. **Knowledge Graphs**:

   - **Application**: Enhancing semantic search and understanding.
   - **Example**: Linking concepts and entities to improve search accuracy.

4. **Fraud Detection**:

   - **Application**: Identifying fraudulent activities by analyzing patterns.
   - **Example**: Detecting unusual connections between entities.

5. **Supply Chain Management**:

   - **Application**: Optimizing logistics and inventory.
   - **Example**: Modeling products, locations, and movements to streamline operations.

### Memgraph

**Memgraph** is a graph database that integrates directly with streaming platforms, enabling real-time data ingestion and processing.

#### Direct Ingestion from Streaming Platforms

- **Supported Platforms**: Apache Kafka, Apache Pulsar.
- **Transformation Modules**: Custom code to transform incoming messages.
  - **Languages**: C API, Python API.

#### Example: Ingesting Data from Kafka

**Step 1: Create a Kafka Stream**

```sql
CREATE KAFKA STREAM stream_name
  TOPICS topic1 [, topic2, ...]
  TRANSFORM transform_procedure
  [CONSUMER_GROUP consumer_group]
  [BATCH_INTERVAL duration]
  [BATCH_SIZE size]
  [BOOTSTRAP_SERVERS servers]
  [CONFIGS { key1: value1 [, key2: value2, ...]}]
  [CREDENTIALS { key1: value1 [, key2: value2, ...]}];

START STREAM stream_name [BATCH_LIMIT count] [TIMEOUT milliseconds];
```

- **Explanation**:
  - **`CREATE KAFKA STREAM`**: Defines a new stream that listens to Kafka topics.
  - **`TRANSFORM`**: Specifies a procedure to transform incoming messages.

**Step 2: Define a Transformation Module in Python**

```python
import mgp

@mgp.transformation
def transform_messages(context: mgp.TransCtx, messages: mgp.Messages) -> mgp.Record(query=str, parameters=mgp.Nullable[mgp.Map]):
    result_queries = []
    for i in range(messages.total_messages()):
        message = messages.message_at(i)
        payload = message.payload().decode("utf-8")
        result_queries.append(mgp.Record(
            query=f"CREATE (n:MESSAGE {{timestamp: '{message.timestamp()}', payload: '{payload}', topic: '{message.topic_name()}'}})",
            parameters=None
        ))
    return result_queries
```

- **Explanation**:
  - **Decorator `@mgp.transformation`**: Indicates a transformation function.
  - **Function `transform_messages`**: Processes each message, creating a Cypher query to insert data into the graph.

### thatDot/Quine

**Quine** is an open-source streaming graph database developed by **thatDot**. It combines event streams into a single graph and allows for real-time pattern matching and actions.

#### Streaming Graph for Data Pipelines

- **Features**:
  - **Standing Queries**: Continuous queries that detect patterns over time without predefined windows.
  - **Event Ingestion**: Reads from streams like Kafka, Kinesis, and server-sent events.
  - **Actions on Patterns**: Trigger actions like updating the graph, writing to streams, or invoking webhooks.

#### Example: Ingesting Wikipedia Revision Streams

**Step 1: Set Up Ingestion from Wikipedia's Revision Stream**

```shell
curl -X POST "http://127.0.0.1:8080/api/v1/ingest/wikipedia-revision-create" \
     -H 'Content-Type: application/json' \
     -d '{
  "format": {
    "query": "CREATE ($that)",
    "parameter": "that",
    "type": "CypherJson"
  },
  "type": "ServerSentEventsIngest",
  "url": "https://stream.wikimedia.org/v2/stream/mediawiki.revision-create"
}'
```

- **Explanation**:
  - **Ingest Type**: `ServerSentEventsIngest` connects to a streaming source.
  - **Format**: Specifies how to transform incoming data into graph nodes using Cypher.

**Step 2: Define Ingest Query in Cypher**

```cypher
MATCH (revNode), (pageNode), (dbNode), (userNode), (parentNode)
WHERE id(revNode) = idFrom('revision', $that.rev_id)
  AND id(pageNode) = idFrom('page', $that.page_id)
  AND id(dbNode) = idFrom('db', $that.database)
  AND id(userNode) = idFrom('id', $that.performer.user_id)
  AND id(parentNode) = idFrom('revision', $that.rev_parent_id)
SET revNode = $that,
    revNode.bot = $that.performer.user_is_bot,
    revNode:revision
...
CREATE (revNode)-[:TO]->(pageNode),
       (pageNode)-[:IN]->(dbNode),
       (userNode)-[:RESPONSIBLE_FOR]->(revNode),
       (parentNode)-[:NEXT]->(revNode)
```

- **Explanation**:
  - **Node Creation and Relationships**: Maps incoming data to nodes and edges in the graph.
  - **Labels and Properties**: Assigns labels like `:revision`, `:page`, and sets properties from the data.

**Step 3: Create a Standing Query**

```json
{
  "pattern": {
    "query": "MATCH (n)-[:has_father]->(m) WHERE exists(n.name) AND exists(m.name)
    RETURN DISTINCT strId(n) AS kidWithDad",
    "type": "Cypher"
  },
  "outputs": {
    "file-of-results": {
      "path": "kidsWithDads.jsonl",
      "type": "WriteToFile"
    }
  }
}
```

- **Explanation**:
  - **Pattern Matching**: Continuously matches nodes with a `has_father` relationship.
  - **Action**: Writes the results to a file `kidsWithDads.jsonl`.

**Step 4: Query the Graph**

```cypher
MATCH (userNode:user {user_is_bot: false})-[:RESPONSIBLE_FOR]->(revNode:revision {database: 'enwiki'})
RETURN DISTINCT strid(userNode) as NodeID,
                revNode.page_title as Title,
                revNode.performer.user_text as User
LIMIT 10
```

- **Explanation**:
  - **Filters**: Excludes bot users and focuses on the English Wikipedia.
  - **Result**: Retrieves the last 10 revisions made by non-bot users.

---

## Vector Databases

### Introduction to Vector Databases

**Vector Databases** are specialized databases optimized for storing and querying high-dimensional vector data. They enable efficient similarity searches using techniques like Approximate Nearest Neighbor (ANN) algorithms.

**Popular Vector Databases**:

- **Milvus**
- **Weaviate**
- **Pinecone**
- **Vespa**
- **Qdrant**

**Traditional Databases Adding Vector Support**:

- **ClickHouse**
- **Rockset**
- **PostgreSQL**
- **Cassandra**
- **Elastic**
- **Redis**
- **SingleStore**

### Vector Database Use Cases

1. **Recommendation Systems**:

   - **Application**: Finding items similar to a user's preferences.
   - **Example**: Suggesting products based on similarity in feature space.

2. **Fraud Detection**:

   - **Application**: Identifying anomalous patterns in transactions.
   - **Example**: Flagging transactions that are vectorially similar to known fraudulent activities.

3. **Chatbots and Generative AI**:

   - **Application**: Enhancing natural language understanding and responses.
   - **Example**: Using embeddings for intent classification or retrieval-augmented generation (RAG).

### Milvus 2.x: Streaming as the Central Backbone

#### Architecture Overview

- **Streaming Platform Integration**: Utilizes a message broker for data persistence and asynchronous queries.
- **Log as Data Principle**: Emphasizes logging persistence over maintaining physical tables.
- **Unified Batch and Stream Processing**: Implements a lambda architecture to handle both historical and real-time data.
- **Watermark Mechanism**: Breaks unbounded data streams into bounded windows for processing.

![Milvus 2.x Architecture](milvus_2x_architecture.png)

*Figure 11-3: Milvus 2.x Architecture with Streaming as the Central Backbone.*

- **Components**:
  - **Message Storage**: Central backbone for data streaming and persistence.
  - **Worker Nodes**: Handle query execution and data indexing.
  - **Query Nodes**: Execute search and query operations.
  - **Data Nodes**: Manage data ingestion and storage.

### RTOLAP Databases Adding Vector Search

**Real-Time Online Analytical Processing (RTOLAP) Databases** are incorporating vector search capabilities to handle vector data alongside traditional analytical workloads.

#### ClickHouse

- **Vector Embeddings**: Stored using `Array<Float32>` data type.
- **Distance Functions**: Provides functions like `cosineDistance` and `L2Distance` for similarity calculations.
- **Indexing**: Supports ANN indexing methods like Annoy indexes.

##### Example: Combining Vector Search and Metadata Filtering

**SQL Query**:

```sql
SELECT
    url,
    caption,
    L2Distance(image_embedding, [/* query embedding vector */]) AS score
FROM images
WHERE
    width >= 300 AND
    height >= 500 AND
    copyright = '' AND
    similarity > 0.3
ORDER BY score ASC
LIMIT 10
FORMAT Vertical
```

- **Explanation**:
  - **Metadata Filtering**: Filters images based on dimensions and copyright status.
  - **Vector Search**: Calculates the L2 distance between stored embeddings and the query vector.
  - **Ordering**: Sorts results by similarity score in ascending order.

#### Rockset

- **Vector Search Support**: Implements K-Nearest Neighbors (KNN) and Approximate Nearest Neighbors (ANN).
- **Indexing**: Allows creating ANN indexes using algorithms like Faiss.

##### Example: Setting Up an ANN Index and Querying

**Step 1: Create an ANN Index**

```sql
CREATE SIMILARITY INDEX book_embeddings_index
ON FIELD books_dataset.book_embedding DIMENSION 1536 AS 'faiss::IVF256,Flat';
```

- **Explanation**:
  - **Index Name**: `book_embeddings_index`.
  - **Target Field**: `books_dataset.book_embedding`.
  - **Dimension**: Specifies the dimensionality of the embeddings.
  - **Index Type**: Uses Faiss's IVF index.

**Step 2: Query Using Vector Search**

```sql
SELECT
    books_dataset.title,
    books_dataset.author
FROM
    books_dataset
    JOIN book_metadata ON books_dataset.isbn = book_metadata.isbn
WHERE
    book_metadata.publish_date > DATE(2010, 12, 26) AND
    book_metadata.rating >= 4 AND
    book_metadata.price < 50
ORDER BY
    APPROXIMATE_NEAREST_NEIGHBORS(books_dataset.book_embedding, :target_embedding) DESC
LIMIT 30;
```

- **Explanation**:
  - **Metadata Filtering**: Filters books based on publication date, rating, and price.
  - **Vector Search**: Orders results by similarity to the target embedding.
  - **Function**: `APPROXIMATE_NEAREST_NEIGHBORS` computes similarity scores.

---

## Summary

The future of real-time data is marked by the convergence of streaming technologies and databases across all data planes:

- **Operational Databases** like graph and vector databases are integrating streaming capabilities to handle dynamic, real-time data ingestion and querying.
- **Graph Databases** like Memgraph and thatDot/Quine are pioneering direct integration with streaming platforms and offering features like standing queries for real-time pattern detection.
- **Vector Databases** are adopting streaming architectures internally (e.g., Milvus 2.x) and being enhanced with vector search capabilities in RTOLAP databases like ClickHouse and Rockset.
- **Streaming as a Backbone**: Databases are leveraging streaming platforms not just for data ingestion but as central components of their architecture, enabling scalability, resilience, and real-time processing.

This convergence is leading to more versatile databases capable of handling a wider range of workloads, bridging the gap between operational and analytical applications, and offering new possibilities for real-time data processing and analytics.

---

## References

1. **Memgraph Documentation**: [https://memgraph.com/docs](https://memgraph.com/docs)
2. **thatDot/Quine**: [https://quine.io](https://quine.io)
3. **Milvus 2.x Architecture**: [https://milvus.io/docs/architecture_overview](https://milvus.io/docs/architecture_overview)
4. **ClickHouse Vector Search**: [https://clickhouse.com/docs/en/sql-reference/functions/associative-search-functions](https://clickhouse.com/docs/en/sql-reference/functions/associative-search-functions)
5. **Rockset Vector Search**: [https://rockset.com/blog/introducing-vector-search/](https://rockset.com/blog/introducing-vector-search/)

---

## Tags

#RealTimeData #StreamingDatabases #GraphDatabases #VectorDatabases #Memgraph #Quine #Milvus #ClickHouse #Rockset #DataEngineering #StaffPlusNotes

---

Feel free to reach out if you have any questions or need further clarification on any of these topics.