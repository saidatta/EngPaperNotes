This document covers a detailed architecture and workflow of the Percolator system, which is designed to deliver real-time notifications of metadata changes to analytic computations in a distributed system. It breaks down all major components, interactions, and specific design considerations with regards to query percolation, Kafka integration, Thrift API, and Kafka event schemas.

---
## Overview
### Key Problem
Analytics currently refreshes its queries every 9 minutes by default, leading to metadata changes taking a long time to propagate. The Percolator system aims to reduce this delay by offering real-time notifications about metadata changes, improving responsiveness for ephemeral environments where metadata might change quickly.
### Goal
The goal of the Percolator is to provide real-time notifications of metadata changes to analytics, reducing lag caused by periodic refreshes and enabling real-time query updates.

---
## Architecture
### Main Components
- **Thrift Service**: Exposes API endpoints for query registration/unregistration and retrieves query results. This service listens to streams of metadata updates (MTS creation, activation, dimension updates) and notifies clients of changes in real time.
- **Kafka Events**: Each Percolator client has a dedicated Kafka topic for event notifications regarding query result changes.
- **State Storage**: Query registrations and results are stored in Cassandra (C*), providing persistent query data. This allows the system to recover states when a node rejoins the cluster.
### Load Drivers in Percolator
- **MTS Creations/Second**: A major contributor to system load. Scaling can be achieved by adding more Percolator instances, each handling fewer MTS partitions.
- **Total Query Count**: All queries are processed by every node, which could become a bottleneck as the number of queries increases.
- **Dimension Updates/Second**: These updates are infrequent but still contribute to system load.

---

## Main Components Breakdown

### 1. **Percolator Server**
#### Query Registration/Management
- **Persistence**: Stores queries and their results in Cassandra.
- **Query Cache**: Manages query results in-memory for fast access.
- **Interaction with Meatballs**: Fetches query results from the underlying Meatballs datastore.
- **Broadcasts**: Propagates new query registrations across all nodes via Kafka.

#### Code Example (Query Registration Logic in DAO Layer):
```java
public void registerQuery(QuerySpec query) {
    CassandraDAO.persistQuery(query);  // Persist query to Cassandra
    cache.put(query.getId(), query);   // Cache query in memory
    kafka.broadcast(query);            // Notify other nodes of the new query
}
```

### 2. **Thrift Service Handler**
- Routes incoming API requests to the Percolation Manager.
- Handles query registrations, updates, and retrievals via Thrift.

### 3. **Percolation Manager**
- The central coordinator for query management, percolation logic, and event dispatching. Responsible for orchestrating the interaction between various components.

### 4. **Percolation Engine**
- Processes MTS (Metric Time Series) creation events, activation streams, and dimension updates, determining when to refresh a query.
- For each stream event, it evaluates if a query’s results should change based on metadata updates.

#### ASCII Flow of Percolation Logic:
```
MTS Creation Event
       |
       v
  Query Evaluation -> Check against registered queries
       |
       v
   Percolation Manager -> If match, notify client through Kafka
```

### 5. **Event Dispatch**
- Publishes events to the appropriate Kafka topic when the Percolation Engine decides that a query needs to be refreshed.

### 6. **Metadata Manager**
- Manages the caching and handling of dimension and property information, allowing the percolator to fully evaluate queries that filter on properties.

---

## Percolator Client Library

- **Thrift and Kafka Integration**: Hides the complexity of direct Thrift and Kafka interactions, offering a user-friendly API to register queries and receive notifications.
- **PubSub Model**: Clients register queries and provide handlers that are invoked when query results change.

### Kafka Topic Management
- Handles the creation of Kafka topics as needed for percolator notifications.
- Monitors Kafka messages related to query updates and invokes the necessary handlers in the client.

---

## API (Draft)

### Thrift API

The API exposes methods to register, refresh, and unregister queries, and allows real-time notifications through Kafka topics. 

#### Code Example (Thrift API Definition):
```thrift
service PercolatorService {
    PercolationResult register(1: ID orgId, 2: QuerySpec querySpec, 3: PercolationSpec percolationSpec);
    PercolationResult refresh(1: ID percolationId, 2: PercolationVersion minimumVersion);
    void keepAlive(1: ID percolationId);
    bool unregister(1: ID percolationId);
}
```

#### Percolation Result Structure:
```thrift
struct PercolationResult {
    1: MTSSearchResult searchResults;
    2: PercolationVersion version;
}

struct PercolationVersion {
    1: List<PartitionOffset> mtsCreationOffsets;
    2: List<PartitionOffset> mtsActivationOffsets;
    3: List<PartitionOffset> dimUpdateOffsets;
}
```

---

## Event Schema (Kafka)

### Example Event Structure:
```json
{
  "eventType": "CHANGE",
  "percolationId": "12345",
  "customId": "jobId:0",
  "version": {
    "mtsCreation": [
      { "partition": 0, "offset": 20 },
      { "partition": 1, "offset": 29 }
    ],
    "mtsActivation": [],
    "dimUpdates": [
      { "partition": 2, "offset": 6 }
    ]
  }
}
```

---

## Percolation Mechanics

### Query Fields
Each query has two key sets of fields:
- **Filter Fields**: Fields that determine whether a given MTS matches the query (e.g., `sf_metric`, `sf_tags`).
- **Fetch Fields**: Fields that are returned with the query results but do not influence the match.

### Percolation Behavior
Percolation occurs in response to the following events:
1. **New MTS Creation**: A new MTS is added, and queries are evaluated to see if they match the new data.
2. **MTS Activation/Deactivation**: Changes in MTS state, such as becoming active or inactive, may trigger a query refresh.
3. **Dimension Updates**: Changes in MTS dimensions or properties may cause queries to either include or exclude specific MTS records.

#### ASCII Example - MTS Stream Processing:
```
New MTS Stream
      |
      v
  Evaluate Filter Fields
      |
      v
Match? -----> Yes -> Notify Client (Kafka Event)
      |
      v
   No -> Ignore
```

---

## Open Issues

### 1. **Query Consistency**
There is potential for inconsistency when the percolator and the underlying datastore (Meatballs) consume updates at different rates. The percolator might notify the client of a change before Meatballs reflects that change in query results.

### 2. **Event Batching**
How to batch events to reduce load on Kafka while ensuring timely delivery of query result changes.

### 3. **Percolation State Distribution**
Efficiently distributing query percolation responsibilities across multiple nodes to scale out effectively.

---

## Planning

### Milestones
- **Milestone 1**: Basic percolation using SFC (service flow control) mechanism.
- **Milestone 2**: Analytics integration with percolator in a test environment, collecting metrics.
- **Milestone 3**: Percolator beta release, demonstrating live query percolation during Twitch demo.
- **Milestone 4**: General Availability (GA) release with multi-node support.
- **Milestone 5**: Post-GA enhancements, adding robustness and performance optimizations.

### Prep Work
- Add metrics to track query refresh rate and determine cases where no changes are detected.
- Analyze existing analytics queries to determine the proportion that are pure dimension-based queries.

### Dependencies
- **MTS Properties**: Extend the active/inactive events to include MTS properties.
- **Query Versioning**: Modify the Meatballs query interface to accept offset-based version data for enhanced consistency.

---