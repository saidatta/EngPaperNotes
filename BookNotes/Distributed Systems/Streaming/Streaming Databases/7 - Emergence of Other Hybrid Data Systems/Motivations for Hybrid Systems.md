## Overview

Modern data processing systems are evolving to meet the demands of real-time analytics while reducing infrastructure complexity and increasing developer accessibility. **Hybrid systems** are emerging as a solution, combining features of stream processing, OLTP, and OLAP databases. This note explores the motivations behind hybrid systems, their influence on the industry, and the potential future directions for next-generation hybrid databases.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Stream Processing Complexity](#stream-processing-complexity)
3. [OLTP Databases Adopting OLAP Features](#oltp-databases-adopting-olap-features)
4. [Common Goals of Hybrid Systems](#common-goals-of-hybrid-systems)
5. [Influence of PostgreSQL on Hybrid Databases](#influence-of-postgresql-on-hybrid-databases)
6. [Near-Edge Analytics](#near-edge-analytics)
7. [Next-Generation Hybrid Databases](#next-generation-hybrid-databases)
   - [Next-Generation Streaming OLTP Databases](#next-generation-streaming-oltp-databases)
   - [Next-Generation Streaming RTOLAP Databases](#next-generation-streaming-rtolap-databases)
   - [Next-Generation HTAP Databases](#next-generation-htap-databases)
8. [Summary](#summary)

---

## Introduction

Hybrid systems aim to bridge the gap between different data processing paradigms:

- **Stream Processing Systems**: Provide real-time data processing but are often complex to adopt.
- **OLTP Databases**: Handle transactional workloads but are integrating OLAP features to serve analytical queries.
- **OLAP Databases**: Handle analytical workloads but may lack the immediacy required for real-time applications.

By combining features from these systems, hybrid databases strive to offer real-time analytics with less infrastructure and greater accessibility.

---

## Stream Processing Complexity

- **Stigma of Complexity**: Stream processing is perceived as hard and complex, discouraging adoption.
- **Knowledge Gap**: Organizations lack skilled data engineers proficient in stream processing.
- **Vendor Solutions**:
  - Simplify APIs and interfaces.
  - Provide user-friendly tools similar to traditional databases.
  - Bridge the gap by offering a database-like experience in stream processing systems.

### Challenges

- **Adoption Barriers**: Complexity leads to reluctance in embracing real-time analytics.
- **Resource Scarcity**: Shortage of experts in real-time data processing.
- **Integration Difficulty**: Combining stream processing with existing systems is non-trivial.

---

## OLTP Databases Adopting OLAP Features

- **Motivation**: To better serve analytical queries at the operational plane.
- **Avoiding Round Trips**: Reduce the need to transfer data to the analytical plane, which adds complexity.
- **Features Being Integrated**:
  - **Streaming Capabilities**: Incorporating streaming features for real-time data replication and synchronization.
  - **Analytical Query Support**: Enhancing capabilities to handle analytical workloads directly.

### Challenges

- **Data Consistency**: Ensuring consistency across distributed environments.
- **Complexity of Integration**: Adding OLAP features without compromising OLTP performance.

---

## Common Goals of Hybrid Systems

- **Provide Real-Time Analytics**: Deliver insights promptly to support immediate decision-making.
- **Reduce Infrastructure Complexity**: Minimize the number of systems and data movement required.
- **Increase Accessibility**: Make systems easier to adopt for engineers with varying expertise levels.

### Key Strategies

- **Convergence of Systems**: Blending features of different data planes.
- **Simplified Architectures**: Building systems that can handle multiple workloads efficiently.
- **Developer-Friendly Interfaces**: Offering tools and APIs that are intuitive and familiar.

---

## Influence of PostgreSQL on Hybrid Databases

### Popularity Factors

| Factor               | Description                                                                                                                                                  |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Extensibility**    | Allows creation of custom data types, operators, and functions, making it adaptable to various applications.                                                 |
| **Performance**      | Ongoing optimizations result in efficient query processing and competitive performance.                                                                      |
| **Third-Party Ecosystem** | Rich ecosystem of tools, libraries, and extensions enhances capabilities and flexibility.                                                            |
| **Enterprise Adoption**    | Widely adopted by large organizations, adding credibility and fostering community growth.                                                             |
| **Global Reach**     | Not region or industry-specific, appealing to a diverse global user base.                                                                                    |

### Impact on Hybrid Databases

- **Adoption in Hybrid Systems**: Many hybrid databases use PostgreSQL as a foundation (e.g., RisingWave, Materialize).
- **Community Support**: Leveraging the extensive PostgreSQL community accelerates development and adoption.
- **Feature-Rich Platform**: Provides a solid base for adding hybrid features.

---

## Near-Edge Analytics

- **Definition**: Bringing analytics closer to the operational plane to reduce latency.
- **Goal**: Provide timely insights to end-users without accessing the full data repository.
- **Hybrid Databases Role**: Offer analytical capabilities without replicating the entire analytical plane.

### Limitations of Full Analytical Workloads on Operational Plane

1. **Large Historical Data Aggregation**:
   - Petabytes of data unsuitable for operational infrastructure.
2. **Machine Learning Model Training**:
   - Requires specialized systems not present in the operational plane.
3. **Highly Distributed Systems**:
   - Need for partitioning data for MPP not supported operationally.
4. **Ad Hoc Query Flexibility**:
   - Internal data analysts need flexibility not typically exposed to external users.

### Strategy

- **Optimized Data Delivery**: Provide relevant subsets of data for immediate decisions.
- **Hybrid Approach**: Use hybrid databases to balance performance and capability.

---

## Next-Generation Hybrid Databases

![Next-Generation Databases](next_gen_databases.png)

*Figure 7-6: Visualization of next-generation real-time databases.*

### Characteristics

- **Convergence of Data Planes**: Incorporate features from streaming, operational, and analytical planes.
- **Features**:
  - **Stateful Stream Processing**
  - **Columnar Storage for Analytical Workloads**
  - **Consistency for Operational Workloads**

### Potential Development Paths

- **HTAP Databases**: Add stream processing capabilities.
- **Streaming OLTP Databases**: Integrate columnar storage.
- **Streaming OLAP Databases**: Enhance consistency mechanisms.

---

### Next-Generation Streaming OLTP Databases

#### Areas of Improvement

1. **Data Consistency Models**

   - **Necessity**: Required for participating in application logic.
   - **Approach**: Implement stronger consistency guarantees in stream processing components.

2. **Access to Change Data (WAL)**

   - **Current Challenge**: CDC requires external connectors, increasing complexity.
   - **Solution**: Emit CDC transactions directly to streaming platforms (e.g., Kafka).

   **Example Diagram**:

   ![Emitting CDC Data](cdc_emission.png)

   *Figure 7-7: Emitting CDC data directly into a Kafka-compliant topic.*

3. **Incorporate Columnar Storage**

   - **Goal**: Support analytical workloads without storing all historical data.
   - **Method**: Embed OLAP databases like DuckDB for handling analytical queries.
   - **Data Handling**: Provide subsets of historical data relevant to the application's domain.

---

### Next-Generation Streaming RTOLAP Databases

#### Current Limitations

- **Lack of Stream Processing Capabilities**
- **Dependence on External Stream Processors**
- **Complexity in Data Transformation and Ingestion**

#### Enhancements

1. **Incorporate Stream Processing**

   - **Benefit**: Reduce reliance on external systems.
   - **Feature**: Provide push-query capabilities to data analysts.

2. **Improved Ingestion Mechanisms**

   - **Customization**: Allow specific transformations tailored to downstream needs.
   - **Efficiency**: Publish data for specific subscribers, reducing unnecessary processing.

---

### Next-Generation HTAP Databases

#### Implement Incremental View Maintenance (IVM)

- **Definition**: Asynchronously applies incremental modifications to materialized views.
- **Benefit**: Avoids full reevaluation of views, akin to stream processing.

#### Transform Transactions

- **Row-Based to Column-Based**: Facilitate low-latency analytical queries.
- **Without Data Egress**: Keep data within the system, reducing movement and latency.

#### Ingress Limited Historical Data

- **Purpose**: Provide historical context to real-time analytical data.
- **Method**: Integrate limited data from the analytical plane.

---

## Summary

- **Hybrid Databases**: Bridging the gap between different data processing paradigms to provide real-time analytics efficiently.
- **Convergence Trends**: Systems are evolving by adding features from other data planes to reduce complexity and latency.
- **Future Directions**:
  - **Avoid Monolithic Pitfalls**: Aim for distributed yet integrated systems.
  - **Enhanced Features**: Incorporate stream processing, columnar storage, and consistency across systems.
- **Goal**: Create systems that are flexible, scalable, and accessible, enabling organizations to leverage real-time data effectively.

---

## Additional Notes

### Mathematical Concepts

#### Consistency in Stream Processing

- **Strong Consistency**: Guarantees immediate visibility of data changes.
- **Eventual Consistency**: Updates propagate over time; eventual convergence.

#### Implications for Hybrid Systems

- **Application Logic Participation**: Requires strong consistency to make reliable decisions.
- **Data Replication and Synchronization**: Efficient mechanisms reduce latency and errors.

### Code Examples

#### Emitting CDC Data to Kafka

**Pseudo-code for Emitting CDC Events:**

```python
import psycopg2
from kafka import KafkaProducer
import json

# Connect to PostgreSQL
conn = psycopg2.connect("dbname=test user=postgres password=secret")
cur = conn.cursor()

# Set up Kafka producer
producer = KafkaProducer(bootstrap_servers='localhost:9092')

# Listen to WAL changes (simplified)
cur.execute("LISTEN wal_changes;")
while True:
    conn.poll()
    while conn.notifies:
        notify = conn.notifies.pop(0)
        change = json.loads(notify.payload)
        # Emit change to Kafka topic
        producer.send('cdc_topic', value=json.dumps(change).encode('utf-8'))
```

- **Explanation**:
  - Connects to the PostgreSQL database.
  - Listens for changes in the Write-Ahead Log (WAL).
  - Emits changes to a Kafka topic named `cdc_topic`.

---

## References

1. **Gartner's HTAP Definition**: [Hybrid Transactional/Analytical Processing](https://www.gartner.com/en/documents/2658415)
2. **Incremental View Maintenance**: [Wikipedia Article](https://en.wikipedia.org/wiki/Incremental_view_maintenance)
3. **PostgreSQL Documentation**: [Official PostgreSQL Documentation](https://www.postgresql.org/docs/)
4. **Apache Kafka Documentation**: [Kafka Official Documentation](https://kafka.apache.org/documentation/)

---

## Tags

- #HybridDatabases
- #StreamProcessing
- #OLTP
- #OLAP
- #RealTimeAnalytics
- #DataConsistency
- #PostgreSQL
- #IncrementalViewMaintenance
- #DataEngineering
- #StaffPlusNotes

---

## Footnotes

- **1**: *Consistency Models* - Different levels of data consistency impact the reliability and performance of data systems.
- **2**: *Incremental View Maintenance (IVM)* - A method to update materialized views by only applying changes rather than recomputing the entire view.

---

Feel free to reach out if you have any questions or need further clarification on any of these topics.