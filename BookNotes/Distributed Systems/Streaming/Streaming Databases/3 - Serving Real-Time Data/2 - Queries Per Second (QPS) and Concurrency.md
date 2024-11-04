## Overview

**Queries per Second (QPS)** is a critical metric that measures how well an OLAP data store can process and return query results. It indicates the volume of queries an OLAP system can handle within a second and directly correlates with the number of concurrent queries it can support.

In our clickstream use case, we aim to serve real-time analytics to users through dashboards that refresh frequently to maintain data freshness. This scenario requires careful consideration of:

- **Latency Requirements**
- **Concurrency Levels**
- **Indexing Strategies**
- **System Scalability**

---

## Calculating QPS Requirements

### Scenario

- **Number of Users**: 1,000
- **Dashboard Refresh Rate**: Every 5 seconds
- **Assumption**: Each user triggers one query per dashboard refresh

### Calculations

**Total Queries per Second (QPS):**

\[
\text{QPS} = \frac{\text{Number of Users}}{\text{Refresh Rate (in seconds)}}
\]

\[
\text{QPS} = \frac{1,000}{5} = 200 \text{ queries/second}
\]

**Latency per Query:**

To achieve 200 QPS, each query must complete within:

\[
\text{Latency} = \frac{1}{\text{QPS}} = \frac{1}{200} = 0.005 \text{ seconds} = 5 \text{ milliseconds}
\]

### Implications

- **High Performance Requirement**: Each query must return results in under 5 milliseconds.
- **Scalability Concerns**: Supporting more users or higher refresh rates requires scaling the OLAP system or optimizing queries.
- **Indexing Necessity**: Proper indexing strategies are crucial to meet latency requirements without excessive hardware scaling.

---

# Indexing in OLAP Data Stores

## Importance of Indexing

Indexing is pivotal in improving QPS for OLAP databases by:

- **Reducing Query Response Time**: Allows the database to quickly locate and retrieve relevant data.
- **Optimizing Query Execution Plans**: Provides metadata and statistics for the query optimizer.
- **Enabling Data Pruning**: Eliminates unnecessary data scans by narrowing down the search space.
- **Minimizing Disk I/O**: Organizes data to reduce disk read operations during queries.

## Types of Indexes

Different indexing strategies are suited for various data types and query patterns.

### Apache Pinot Indexes

Apache Pinot, an RTOLAP data store, supports several types of indexes:

| Index Type     | Description                                                                 | Use Case                        |
|----------------|-----------------------------------------------------------------------------|---------------------------------|
| **Bitmap Index**   | Efficient for low-cardinality columns. Creates a bitmap for each distinct value. | Columns with few distinct values. Example: Gender (Male/Female). |
| **Sorted Index**   | Stores column values in sorted order. Ideal for range queries.           | Date and timestamp columns.     |
| **Inverted Index** | Maps each unique value to the set of rows containing that value.        | High-cardinality columns. Example: User IDs. |
| **Forward Index**  | Stores raw column values in their original order.                       | Columns not used for filtering or aggregations but needed for result output. |
| **Text Index**     | Designed for full-text search capabilities.                             | Text columns requiring search. Example: Product descriptions. |

### Index Selection Strategy

- **Low-Cardinality Columns**: Use **Bitmap Indexes**.
- **High-Cardinality Columns**: Use **Inverted Indexes**.
- **Range Queries**: Use **Sorted Indexes**.
- **Full-Text Search**: Use **Text Indexes**.

## Example: Implementing Indexes in Apache Pinot

### Defining Indexes in Table Configuration

```json
{
  "tableName": "clickstream_events",
  "tableType": "REALTIME",
  "segmentsConfig": {
    "timeColumnName": "event_time",
    "schemaName": "clickstream_events_schema",
    "replication": "3"
  },
  "tableIndexConfig": {
    "loadMode": "MMAP",
    "invertedIndexColumns": ["user_id", "session_id"],
    "sortedColumn": ["event_time"],
    "textIndexColumns": ["search_terms"]
  },
  "ingestionConfig": {
    "streamConfigs": {
      "streamType": "kafka",
      "stream.kafka.topic.name": "clickstream_events",
      "stream.kafka.broker.list": "kafka-broker:9092",
      "stream.kafka.consumer.type": "lowlevel",
      "stream.kafka.decoder.class.name": "org.apache.pinot.plugin.stream.kafka.KafkaJSONMessageDecoder"
    }
  }
}
```

### Explanation

- **Inverted Indexes**: Applied to `user_id` and `session_id` for efficient lookups.
- **Sorted Column**: `event_time` is sorted to optimize range queries on time.
- **Text Index**: `search_terms` column is indexed for full-text search capabilities.

---

## Star-Tree Index in Apache Pinot

### Concept

The **Star-Tree Index** is an advanced indexing technique that pre-aggregates data across multiple dimensions, significantly speeding up aggregation queries.

### How It Works

1. **Tree Structure**: Organizes data hierarchically based on dimension cardinality.
2. **Pre-Aggregation**: Aggregates metrics at different levels of the tree during data ingestion.
3. **Query Optimization**: Queries traverse the tree to find pre-aggregated results, reducing scan times.

### Example Scenario

Assume we have dimensions **D1** (Country) and **D2** (Product Category), and a metric **M1** (Sales).

#### Query 1: Total Sales for Country = 'USA'

- **SQL**:
  ```sql
  SELECT SUM(sales) FROM sales_data WHERE country = 'USA';
  ```
- **Star-Tree Path**:
  - Root → 'USA' → * (Wildcard on D2)
  - Retrieves pre-aggregated sales for 'USA' across all product categories.

#### Query 2: Total Sales for Country = 'USA' and Category = 'Electronics'

- **SQL**:
  ```sql
  SELECT SUM(sales) FROM sales_data WHERE country = 'USA' AND category = 'Electronics';
  ```
- **Star-Tree Path**:
  - Root → 'USA' → 'Electronics'
  - Retrieves pre-aggregated sales for 'USA' in 'Electronics' category.

### Configuring Star-Tree Index

```json
{
  "tableIndexConfig": {
    "starTreeIndexConfigs": [
      {
        "dimensionsSplitOrder": ["country", "category"],
        "skipStarNodeCreationForDimensions": [],
        "functionColumnPairs": ["SUM__sales"],
        "maxLeafRecords": 10000
      }
    ]
  }
}
```

- **`dimensionsSplitOrder`**: Order of dimensions in the tree based on cardinality.
- **`functionColumnPairs`**: Metrics to pre-aggregate.
- **`maxLeafRecords`**: Controls the size of leaf nodes to balance between index size and query performance.

### Benefits

- **Reduced Query Latency**: Queries access pre-aggregated data.
- **Controlled Scan Size**: Limits the number of records scanned per query.
- **Efficient Aggregations**: Ideal for high-cardinality datasets.

---

## Segment Generation in Apache Pinot

### What is a Segment?

- **Definition**: A segment is a chunk of data in Pinot that contains a subset of the table's data.
- **Purpose**: Allows horizontal scaling by distributing data across multiple nodes.
- **Storage**: Segments store data in a columnar fashion, along with dictionaries and indexes.

### Segment Creation Process

1. **Data Ingestion**: Raw data is ingested from sources like Kafka.
2. **Transformation**: Ingestion transformations are applied (e.g., data formatting).
3. **Indexing**: Indexes are built, including star-tree indexes if configured.
4. **Segment Assignment**: Segments are distributed across Pinot server nodes.

---

# Serving Analytical Results

## Synchronous Queries (Pull Queries)

### Characteristics

- **Client-Initiated**: Clients request data when needed.
- **Request-Response Pattern**: Follows a traditional query model.
- **High QPS Dependency**: System performance measured by how many queries it can handle per second.

### Implementation

- **Clients**: Dashboards, applications, or users submitting queries.
- **Communication**: Use drivers and dialects specific to the OLAP system.
- **Example Query**:

  ```sql
  SELECT product_name, SUM(clicks) as total_clicks
  FROM clickstream_data
  WHERE event_date = '2023-07-15'
  GROUP BY product_name
  ORDER BY total_clicks DESC
  LIMIT 10;
  ```

### Challenges

- **High Refresh Rates**: Real-time dashboards may refresh every few seconds.
- **Resource Intensive**: Frequent queries can strain the OLAP system.
- **Latency Requirements**: Must maintain low query latency despite load.

---

## Asynchronous Queries (Push Queries)

### Characteristics

- **Server-Initiated**: Server pushes data to clients when updates occur.
- **Event-Driven**: Clients subscribe to data changes.
- **Reduced QPS Dependency**: Eliminates need for frequent polling.

### Implementation

- **Communication Protocols**: Use Server-Sent Events (SSEs) or WebSockets.
- **Client Subscription**: Clients subscribe to topics or data streams.
- **Example Flow**:

  1. **Initial Subscription**: Client subscribes to a data stream.
  2. **Data Push**: Server pushes updates to the client as they occur.
  3. **Client Update**: Client application updates the UI or processes data.

### Advantages

- **Lower Server Load**: Reduces the number of queries processed.
- **Real-Time Updates**: Clients receive data as soon as it's available.
- **Efficient Resource Utilization**: Avoids unnecessary data retrieval.

---

## Push vs. Pull Queries

### Comparison

| Aspect            | Push Queries                       | Pull Queries                      |
|-------------------|------------------------------------|-----------------------------------|
| **Initiation**    | Server pushes data to clients      | Clients request data from server  |
| **Communication** | Asynchronous                       | Synchronous                       |
| **Latency**       | Lower latency for updates          | Depends on polling frequency      |
| **Resource Usage**| Efficient for server, less network traffic | Potentially higher server load and network traffic |
| **Use Cases**     | Real-time notifications, live feeds| Ad-hoc queries, on-demand data    |

### Combining Push and Pull Queries

- **Initial Data Load**: Client performs a pull query to get the current dataset.
- **Continuous Updates**: Client subscribes to push queries for incremental updates.
- **Benefits**:
  - Ensures data consistency with initial snapshot.
  - Reduces server load by avoiding repeated full data retrievals.

### Illustration

![Push and Pull Queries Combined](push_pull_queries.png)

*Figure: A pull query is invoked by the dashboard and automatically subscribes to changes to the view.*

---

## Limitations in OLAP Data Stores

- **Lack of Push Query Support**: Most OLAP systems are optimized for pull queries and do not natively support push mechanisms.
- **Dead Ends for Raw Data**: OLAP systems focus on serving aggregated results, making it difficult to extract raw data changes.
- **Workarounds**:
  - **Use of Middleware**: Implement external systems to monitor changes and push updates.
  - **Hybrid Approaches**: Combine OLAP systems with streaming platforms for push capabilities.

---

# Summary

In this chapter, we've explored:

- **Queries Per Second (QPS)**: Understanding how QPS and concurrency affect system performance and user experience.
- **Indexing Strategies**: Implementing indexes like bitmap, inverted, and star-tree indexes to optimize query performance.
- **Star-Tree Index in Apache Pinot**: Utilizing pre-aggregation during ingestion to speed up aggregation queries.
- **Serving Analytical Results**: Differentiating between synchronous (pull) and asynchronous (push) queries.
- **Push vs. Pull Queries**: Evaluating their advantages, limitations, and how they can be combined.
- **Limitations of OLAP Systems**: Recognizing that traditional OLAP systems may lack native support for push queries.

---

# Mathematical Concepts

## Calculating Latency from QPS

Given:

- **Number of Users (U)**: 1,000
- **Dashboard Refresh Rate (R)**: Every 5 seconds

**Total QPS (Q)**:

\[
Q = \frac{U}{R} = \frac{1,000}{5} = 200 \text{ queries/second}
\]

**Latency per Query (L)**:

\[
L = \frac{1}{Q} = \frac{1}{200} = 0.005 \text{ seconds} = 5 \text{ milliseconds}
\]

---

## Star-Tree Index Mechanics

- **Tree Depth**: Determined by the number of dimensions.
- **Nodes**: Represent combinations of dimension values.
- **Pre-Aggregated Metrics**: Stored at nodes for quick retrieval.

### Example

Given dimensions **D1** (Country with values USA, Canada) and **D2** (Product Category with values Electronics, Clothing):

- **Root Node**: Represents all data (*).
- **Level 1 Nodes**:
  - D1 = USA
  - D1 = Canada
- **Level 2 Nodes**:
  - D1 = USA, D2 = Electronics
  - D1 = USA, D2 = Clothing
  - D1 = Canada, D2 = Electronics
  - D1 = Canada, D2 = Clothing

Pre-aggregated metrics (e.g., total sales) are stored at each node.

---

# Code Examples

## Implementing Push Queries with WebSockets

### Server-Side (Node.js Example using WebSocket)

```javascript
const WebSocket = require('ws');
const wss = new WebSocket.Server({ port: 8080 });

// Simulated data update function
function sendDataUpdate(ws) {
  const dataUpdate = {
    timestamp: new Date(),
    metric: Math.random() * 100
  };
  ws.send(JSON.stringify(dataUpdate));
}

wss.on('connection', (ws) => {
  console.log('Client connected');

  // Send initial data
  sendDataUpdate(ws);

  // Set up interval to send data updates every second
  const intervalId = setInterval(() => {
    sendDataUpdate(ws);
  }, 1000);

  ws.on('close', () => {
    console.log('Client disconnected');
    clearInterval(intervalId);
  });
});
```

### Client-Side (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8080');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received data update:', data);
  // Update dashboard or UI components
};
```

---

## Combining Pull and Push Queries

### Initial Data Fetch (Pull Query)

```javascript
fetch('/api/initialData')
  .then(response => response.json())
  .then(data => {
    // Render initial dashboard with data
  });
```

### Subscribing to Updates (Push Query)

```javascript
const eventSource = new EventSource('/api/dataUpdates');

eventSource.onmessage = (event) => {
  const dataUpdate = JSON.parse(event.data);
  // Update dashboard incrementally
};
```

---

# Best Practices

1. **Optimize Indexing**: Tailor indexing strategies to query patterns and data characteristics.
2. **Monitor Performance Metrics**: Regularly check QPS, latency, and resource utilization.
3. **Combine Push and Pull Mechanisms**: Use pull queries for initial data loads and push queries for updates.
4. **Scale Judiciously**: Scale out the OLAP cluster when necessary but prioritize query optimization first.
5. **Leverage Ingestion Transformations**: Preprocess data during ingestion to reduce query-time computations.

---

# Next Steps

In **Chapter 4**, we will delve deeper into:

- **Materialized Views**: Understanding their role in stream processing and OLAP systems.
- **Advanced Query Serving Techniques**: Exploring how to further optimize real-time data delivery.
- **Streaming Databases**: Introducing databases that natively support both stream processing and materialized views.

Understanding the concepts from this chapter will be crucial as we explore more advanced topics in real-time data serving and analytics.

---

# References

- **Apache Pinot Documentation**: [Apache Pinot](https://pinot.apache.org/)
- **WebSockets vs. Server-Sent Events**: [MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)
- **Optimizing OLAP Systems**: [OLAP Design Best Practices](https://www.olap.com/learn-bi-olap/olap-design-best-practices/)
- **Star-Tree Index in Pinot**: [Star-Tree Index](https://docs.pinot.apache.org/basics/indexing/star-tree)

---

# Footnotes

1. **Segments in Pinot**: Segments are the unit of data storage and management in Pinot, similar to partitions or shards in other databases.

2. **ACID Compliance**: Ensures that OLTP databases maintain data integrity through transactions.

---

Feel free to reach out with any questions or for further clarification on any of these topics.