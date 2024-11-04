## Overview

A **streaming data mesh** implements all the pillars of a data mesh using real-time streams. It enables real-time analytics for all domain consumers. By using **streaming databases**, domains can build streaming data products without needing to deeply understand streaming concepts—they can leverage their existing database knowledge for streaming databases[^1].

In this note, we will explore:

- How streaming databases facilitate a streaming data mesh.
- The role of data locality and replication in a streaming data mesh.
- Implementing data replication using tools like Kafka's MirrorMaker 2.0 (MM2).
- Alternative replication solutions.
- A summary of how streaming databases simplify the adoption of a data mesh.

---

## Table of Contents

1. [Introduction to Streaming Data Mesh](#introduction-to-streaming-data-mesh)
2. [Streaming Databases](#streaming-databases)
   - [Definition](#definition)
   - [Capabilities](#capabilities)
3. [Data Locality](#data-locality)
   - [Performance Implications](#performance-implications)
   - [Security Implications](#security-implications)
4. [Data Replication](#data-replication)
   - [Importance in Streaming Data Mesh](#importance-in-streaming-data-mesh)
   - [Replication Mechanisms](#replication-mechanisms)
5. [Implementing Data Replication with Kafka MirrorMaker 2.0](#implementing-data-replication-with-kafka-mirrormaker-20)
   - [Configuration Example](#configuration-example)
   - [Explanation of Configuration](#explanation-of-configuration)
6. [Alternative Replication Solutions](#alternative-replication-solutions)
   - [Confluent's Cluster Linking](#confluents-cluster-linking)
7. [Summary](#summary)
8. [References](#references)
9. [Tags](#tags)

---

## Introduction to Streaming Data Mesh

A **streaming data mesh** is an architectural paradigm that extends the principles of a traditional data mesh by leveraging real-time data streams. It incorporates all four pillars of a data mesh:

1. **Domain-Oriented Decentralized Data Ownership**: Each domain owns and manages its data streams.
2. **Data as a Product**: Data streams are treated as products, with a focus on quality, documentation, and accessibility.
3. **Self-Serve Data Infrastructure**: Domains can autonomously produce and consume data streams without deep streaming expertise.
4. **Federated Computational Governance**: Shared standards and policies are enforced across domains while allowing local autonomy.

By utilizing streaming databases, domains can build streaming data products using familiar database concepts and SQL queries, reducing the need for specialized streaming knowledge.

![Figure 9-6: The streaming plane uses streaming databases to consume and produce data products as well as connectors and streaming platforms to replicate data products and analytics](figure_9-6.png)

*Figure 9-6: The streaming plane uses streaming databases to consume and produce data products as well as connectors and streaming platforms to replicate data products and analytics.*

---

## Streaming Databases

### Definition

A **streaming database** is a database system that can:

- Consume and emit data streams.
- Execute materialized views asynchronously.
- Provide continuous query processing over streaming data.

### Capabilities

- **Data Ingestion**: Consumes data from streaming platforms like Kafka.
- **Real-Time Processing**: Performs transformations, aggregations, and computations on data in motion.
- **Materialized Views**: Maintains up-to-date views that reflect the current state of streaming data.
- **Data Emission**: Emits processed data as streams for consumption by other systems or domains.

---

## Data Locality

### Performance Implications

- **Reduced Latency**: Local consumption minimizes network latency, leading to faster data access and processing.
- **Optimized Data Processing**: Domains can tailor infrastructure to meet specific performance requirements.
- **Scalability**: Each domain can independently scale its data processing infrastructure based on its needs.

### Security Implications

- **Domain-Level Security Policies**: Domains can enforce their own security measures and access controls.
- **Federated Governance**: Aligns with data mesh principles, balancing local autonomy with global standards.
- **Data Compliance**: Ensures adherence to regulations by implementing security at the domain level.

---

## Data Replication

### Importance in Streaming Data Mesh

- **Ensures Data Availability**: Replicates data across domains for local consumption.
- **Maintains Data Consistency**: Real-time replication keeps data synchronized between domains.
- **Supports Scalability**: Efficient distribution of data processing workloads across domains.

### Replication Mechanisms

- **Streaming Platforms**: Use of technologies like Kafka for data replication.
- **Connectors**: Tools that facilitate data movement between clusters or systems.
- **Network of Connected Clusters**: Establishes interconnected domains sharing data streams.

---

## Implementing Data Replication with Kafka MirrorMaker 2.0

### Introduction to MirrorMaker 2.0 (MM2)

**Kafka MirrorMaker 2.0 (MM2)** is a tool that replicates topics from one Kafka cluster to another, facilitating data replication across different data centers or regions.

### Configuration Example

**Example 9-1: MirrorMaker 2.0 Configuration**

```properties
# Specify any number of cluster aliases
clusters = source, destination  # 1

# Connection information for each cluster
source.bootstrap.servers = kafka-source1:9092,kafka-source2:9092,kafka-source3:9092
destination.bootstrap.servers = kafka-dest1:9092,kafka-dest2:9092,kafka-dest3:9092

# Enable and configure individual replication flows
source->destination.enabled = true

# Define which topics get replicated
source->destination.topics = foo  # 2
groups=.*
topics.blacklist=*.internal,__.*

# Setting replication factor of newly created remote topics
replication.factor=3

checkpoints.topic.replication.factor=1
heartbeats.topic.replication.factor=1
offset-syncs.topic.replication.factor=1

offset.storage.replication.factor=1
status.storage.replication.factor=1
config.storage.replication.factor=1
```

### Explanation of Configuration

1. **Cluster Aliases**:
   - `clusters = source, destination`: Defines aliases for the clusters involved in replication.
   - **Source Cluster**: The cluster where data originates.
   - **Destination Cluster**: The cluster where data is replicated to.

2. **Bootstrap Servers**:
   - `source.bootstrap.servers`: Comma-separated list of host:port pairs for the source cluster.
   - `destination.bootstrap.servers`: Comma-separated list of host:port pairs for the destination cluster.

3. **Replication Flows**:
   - `source->destination.enabled = true`: Enables replication from the source to the destination cluster.

4. **Topic Selection**:
   - `source->destination.topics = foo`: Specifies the topics to replicate (e.g., topics starting with `foo`).
   - `groups=.*`: Replicates all consumer groups.
   - `topics.blacklist=*.internal,__.*`: Excludes internal Kafka topics from replication.

5. **Replication Factors**:
   - `replication.factor=3`: Sets the replication factor for the replicated topics in the destination cluster.
   - Other replication factors for internal topics and storage configurations are set to `1` for simplicity.

### Deployment Considerations

- **Multiple Clusters**: MM2 can replicate between multiple Kafka clusters, but each replication flow (source to destination) requires its own configuration.
- **Kafka Connect**: MM2 is implemented as a connector running within a Kafka Connect cluster.
- **Scaling**: For additional regions or domains, separate instances of MM2 need to be deployed.

### Mathematical Representation

Let:

- \( T_s \): Set of topics in the source cluster.
- \( T_d \): Set of topics in the destination cluster.
- \( R \): Replication function.

Then:

\[
T_d = R(T_s)
\]

Where \( R \) replicates selected topics from \( T_s \) to \( T_d \) based on the configuration.

---

## Alternative Replication Solutions

### Confluent's Cluster Linking (CL)

- **Description**: A proprietary solution provided by Confluent for connecting Kafka clusters and mirroring topics between them.
- **Advantages over MM2**:
  - **No Kafka Connect Required**: Simplifies the infrastructure by eliminating the need for Kafka Connect clusters.
  - **Simplified Configuration**: Easier setup and management of replication flows.
  - **Higher Performance**: Optimized for better performance and lower latency.

### Example of Cluster Linking

```shell
# On the destination cluster, create a link to the source cluster
bin/kafka-cluster-links --bootstrap-server destination:9092 \
  --create --link my-link \
  --cluster-id source-cluster-id \
  --config bootstrap.servers=source:9092

# List available topics on the source cluster
bin/kafka-cluster-links --bootstrap-server destination:9092 \
  --list-topics --link my-link

# Mirror a topic from the source to the destination cluster
bin/kafka-mirrors --bootstrap-server destination:9092 \
  --create --mirror-topic foo \
  --link my-link
```

- **Note**: Requires appropriate configurations and security settings.

---

## Summary

- **Streaming Data Mesh**: Extends data mesh principles using real-time streams, enabling real-time analytics across domains.
- **Streaming Databases**: Facilitate the implementation of a streaming data mesh by providing familiar database interfaces for streaming data.
- **Data Locality**: Improves performance and security by consuming data locally within domains.
- **Data Replication**: Essential for distributing data across domains; implemented using tools like Kafka's MirrorMaker 2.0 or Confluent's Cluster Linking.
- **Simplification**: Streaming databases abstract the complexity of the streaming plane, enabling domains to participate in a data mesh without deep streaming expertise.

---

## References

1. Frank McSherry, "[A Guided Tour Through Materialize’s Product Principles](https://materialize.com/blog/a-guided-tour-through-materializes-product-principles/)," *Materialize*, September 22, 2023.

2. **Streaming Data Mesh (O'Reilly)**: [Streaming Data Mesh](https://www.oreilly.com/library/view/streaming-data-mesh/9781098145697/) by Zhamak Dehghani.

3. "Kafka and RisingWave": [RisingWave and Streaming Databases](https://www.risingwave.com/blog/what-is-streaming-database)

---

## Tags

#StreamingDataMesh #DataMesh #StreamingDatabases #DataReplication #Kafka #MirrorMaker2 #DataLocality #DataEngineering #StaffPlusNotes

---

Feel free to reach out if you have any questions or need further clarification on any of these topics.

[^1]: For more information, see *Streaming Data Mesh* (O'Reilly) and "Kafka and RisingWave".