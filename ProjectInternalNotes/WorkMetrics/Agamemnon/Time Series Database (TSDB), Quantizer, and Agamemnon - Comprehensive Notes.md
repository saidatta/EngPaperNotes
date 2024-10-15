Here are the very detailed Obsidian notes for "Time Series Database (TSDB), Quantizer, and Agamemnon" for Staff+ engineers. This document covers a comprehensive analysis of data management using TSDB, specifically focusing on mappings of Summary MTS (SMTS) and Ephemeral MTS (EMTS), the role of Quantizer, and Agamemnon's architecture in managing these mappings.
## Overview
Seamonkey's primary function is to manage the mappings between **Summary MTS (SMTS)** and their corresponding **Ephemeral MTS (EMTS)**. These mappings are crucial for analytics as they help determine which EMTSes belong to a given SMTS during a specific time range. To ensure accurate and complete data aggregation, all EMTS datapoints continue to reside in TSDB (Time Series Database) even after their metadata expires from metabase.

The architecture involves multiple layers: **Memory Tier, Cassandra, S3/GCS**, with the mappings transitioning between these layers in a structured manner. This setup allows seamless querying and backfill operations using Agamemnon.
### Components
- **TSDB**: Stores all EMTS datapoints.
- **Quantizer**: Manages memory-tier data points and mappings.
- **Agamemnon**: Facilitates read/write operations for mappings stored in memory, Cassandra, and S3/GCS.
## Architecture Design

### High-Level Diagram (ASCII Representation)
```
                +--------------------+
                |   Mapping Service  |
                +---------+----------+
                          |
             +------------+------------+
             |  Kafka Topic (Seamonkey) |
             +------------+------------+
                          |
         +----------------+----------------+
         |                                     |
+--------v--------+                   +--------v--------+
|  Memory Tier 1  |                   |  Memory Tier 2  |
+--------+--------+                   +--------+--------+
         |                                     |
 +-------v---------+                    +-------v---------+
 |   Cassandra     |                    |   Agamemnon     |
 +-------+---------+                    +----------------+
         |
 +-------v---------+
 |   S3 / GCS      |
 +-----------------+
```

The red box highlights the area of interest: Memory Tier, Cassandra, Agamemnon, and S3/GCS interactions.
## Write Path
### Mappings Write Process
1. **Mapping Generation**:
   - The Mapping Service generates new association or dissociation records and publishes them to a Kafka topic on a dedicated Seamonkey cluster.
2. **Memory Tier Ingestion**:
   - **Check for Late Mappings**: The Memory Tier service checks if the timestamp of the mapping is older than the memory tier's lower bound. 
   - **Direct Write**: If not late, the mappings are written to the memory tier directly.
   - **Backfill Path**: Late mappings are sent to Agamemnon for backfilling.
### Code Example (Mapping Write Operation)
```java
// Pseudocode for adding association mappings
AgamemnonClient client = new AgamemnonClient();
client.addAssociation(parentId, childId, effectiveTime);
```
### Memory Tier Data Structures
- **MappingsSubTable**: Each SMTS is indexed in its own subtable.
- **MappingsMemoryRow**: Encodes mappings and stores them in sequence.
## Read Path
The read path enables querying of mappings using Agamemnon's **MMTS partitioner**. It supports two types of reads: `readByIds` and `bulkRead`.
### ReadByIds
1. **Partition Query**: The query is partitioned based on vnode ownership.
2. **GetBlocks Method**: Each SMTS ID is processed by the `getBlocks` method, identifying the relevant data from MappingsTables.
### BulkRead
1. **Shard-Based Query**: Queries are partitioned by shards to balance load.
2. **Paged Retrieval**: Queries are paginated with tokens to manage large datasets.
### ASCII Representation of Read Path
```
+-----------------+
|  Agamemnon      |
+--------+--------+
         |
+--------v--------+
| Memory Tier Read|
+--------+--------+
         |
+--------v--------+
| Cassandra Read  |
+-----------------+
```

## Carry-Forward Mechanism
The **carry-forward mechanism** ensures that mappings are propagated from one table to the next in the memory tier. This ensures that mapping records are available even if their corresponding dissociation records haven't been processed.

### Carry-Forward Methods
1. **At New Table Creation**:
   - Copies mappings from the previous table to the new table.
2. **At Table Migration**:
   - Ensures that mappings carried over during migration are consistent.
## Data Formats and Schema
### Data Encoding
- **MappingValue**: Represents individual mappings with fields for `type`, `timestamp`, and `childId`.
- **MappingsBrick**: Encodes blocks of mappings to handle data migration without decoding individual values.
- **MappingsBundle**: Primary data structure for storing mappings in Cassandra and S3/GCS.
### Example Data Structure (MappingValue)
```java
public class MappingValue {
    byte type;        // Associate or Dissociate
    long timestamp;   // Effective time of the record
    long childId;     // ID of the EMTS
}
```
## Agamemnon Schema and Segment Configurations
Agamemnon manages data using different segment types:
- **MemoryTierMappings**
- **CassandraMappings**
- **S3Mappings and GCSMappings**
### Cassandra Schema for Mappings
```sql
CREATE TABLE <keyspace>.<table> (
    shard32 int,
    time timestamp,
    value blob,
    PRIMARY KEY (shard32, time)
) WITH COMPACT STORAGE;
```

## QueryEngine and Querying Process
### QueryEngine Workflow
- Uses the `read()` and `write()` methods to handle mappings queries.
- Determines which segments need to be queried based on the specified time range.
### Querying Mappings Example (Thrift Interface)
```java
// Example thrift call for querying mappings
MappingsResult result = agamemnonClient.getTimeSeriesMappings(smtsIdSet, startTime, endTime);
```
## Failure Scenarios and Recovery
- **Memory Tier Instance Crash**: Fallback to the redundant cluster; resume consumption from Kafka upon recovery.
- **Backfill Write Failure**: Use a redo log to ensure no data is lost during retries.
- **Migration Failure**: Retain data in the memory tier until successful migration to Cassandra.
## Performance and Scalability
### Performance Metrics
- Focus on **low-latency reads** to meet analytics requirements.
- Caching techniques to improve read latency between max count and mappings queries.

### Scalability Challenges
- **High Cardinality Mappings**: Strategies to distribute records across MMTS instances.
- **Handling Frequent Changes**: Efficient memory usage through stream-based storage.

## Security Considerations
- No public interface access to mappings.
- Data is not encrypted at rest in the memory tier, Cassandra, or S3.

## Code Examples and Thrift Interfaces
### Agamemnon Thrift Methods
```java
// Example Thrift methods for mappings operations
void addAssociation(long parentId, long childId, long effectiveTime);
MappingsResult getTimeSeriesMappings(Set<Long> smtsIds, long start, long end);
```

## Deployment and Upgrade Plan
1. **Kafka Configuration**: Set up kafka-seamonkey for writing mappings.
2. **Agamemnon Configuration**: Enable schema with mappings retention in memory, Cassandra, and S3.
3. **Rolling Upgrades**: Gradual deployment to prevent service disruptions.

## Observability and Monitoring
### Key Metrics
- **Query Latency**: Measure query response times for analytics.
- **Mappings Migration Backlog**: Monitor memory tier to Cassandra migration queue.

### Dashboards
- Add specific charts for analytics query latency, mappings bundle size, and EMTS counts per SMTS.

## Testing and Validation
- **Unit Tests**: Test the new segment types and memory tier storage logic.
- **Functional Tests**: Validate mappings write and read paths using the Thrift API.
- **Performance Tests**: Assess query latency and resource usage during high-load scenarios.

### Migration Plan
- Migrate mappings from memory tier to Cassandra with fitness checks.
- Use bulkRead() to fetch mappings and bulkWrite() to persist in Cassandra.

## Conclusion
The integration of TSDB, Quantizer, and Agamemnon for managing SMTS and EMTS mappings provides a robust and scalable architecture. The setup ensures efficient data handling, high availability, and quick recovery from node failures. Agamemnon’s flexible data management capabilities and seamless integration with memory tiers and Cassandra facilitate optimal performance even at large scales.