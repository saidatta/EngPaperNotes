## Table of Contents
1. [Introduction to Time Series Databases (TSDB)](#introduction-to-time-series-databases-tsdb)
2. [Core Concepts and Data Structures](#core-concepts-and-data-structures)
   - [Time Series Data Model](#time-series-data-model)
   - [Data Points and Batching](#data-points-and-batching)
   - [Rollups and Aggregations](#rollups-and-aggregations)
3. [Thrift Definitions and API](#thrift-definitions-and-api)
   - [Namespaces and Includes](#namespaces-and-includes)
   - [Structs and Enums](#structs-and-enums)
   - [Service Interfaces](#service-interfaces)
4. [Internals of TSDB](#internals-of-tsdb)
   - [Data Ingestion Pipeline](#data-ingestion-pipeline)
   - [Storage Engine](#storage-engine)
   - [Query Processing](#query-processing)
5. [Exception Handling](#exception-handling)
6. [Advanced Features](#advanced-features)
   - [Distributed Systems Considerations](#distributed-systems-considerations)
   - [Scalability and Reliability](#scalability-and-reliability)
7. [Code Examples](#code-examples)
   - [Thrift Service Implementation](#thrift-service-implementation)
   - [Data Ingestion Example](#data-ingestion-example)
   - [Querying Data](#querying-data)
8. [Equations and Algorithms](#equations-and-algorithms)
9. [ASCII Diagrams](#ascii-diagrams)
10. [Practical Applications](#practical-applications)
11. [References](#references)

---

## Introduction to Time Series Databases (TSDB)

**Time Series Databases (TSDBs)** are optimized for storing, retrieving, and managing time-stamped data, commonly known as time series data. This type of data is prevalent in various domains such as monitoring systems, financial services, IoT applications, and more.

### Key Characteristics:
- **Time-Stamped Data**: Each data point is associated with a timestamp.
- **High Write Throughput**: Capable of handling high ingestion rates.
- **Efficient Range Queries**: Optimized for queries over specific time ranges.
- **Compression and Storage Optimization**: Efficient storage mechanisms to handle large volumes of data.
- **Scalability**: Ability to scale horizontally to accommodate growing data.

---

## Core Concepts and Data Structures

### Time Series Data Model

A **time series** is a sequence of data points indexed in time order. In TSDBs, each time series is typically identified by a unique metric and source combination.
#### Struct: `TimeSeries`
```thrift
struct TimeSeries {
  10: required ID.ID mtsId,
  12: required DataTypes.RollupType rollup,
  15: required i64 resolutionMs = 0,
  20: required list<ValueTypes.TimeValue> timeValues
}
```
- **mtsId**: Unique identifier for the metric-time series.
- **rollup**: Type of data aggregation applied.
- **resolutionMs**: Granularity of data points in milliseconds.
- **timeValues**: Ordered list of timestamp-value pairs.

### Data Points and Batching
Data points represent individual measurements within a time series.
#### Struct: `Datum`
```thrift
struct Datum {
  10: required ID.ID tsId,
  20: required i64 timestampMs,
  30: optional ValueTypes.Value value,
  40: optional bool synthetic = false
}
```
- **tsId**: Identifier of the time series.
- **timestampMs**: Timestamp in milliseconds since epoch.
- **value**: The measured value.
- **synthetic**: Indicates if the data point is interpolated or extrapolated.
#### Struct: `DataBatch`
```thrift
struct DataBatch {
  10: i64 timestampMs,
  20: list<Datum> data,
  30: optional ID.ID traceId,
  31: optional map<string, string> traceContext,
  40: optional i64 maxDelayMs
}
```
- **timestampMs**: Logical timestamp for the batch.
- **data**: List of `Datum` objects sharing the same logical timestamp.
- **traceId & traceContext**: For tracing and debugging.
- **maxDelayMs**: Maximum allowable delay for data processing.
### Rollups and Aggregations
**Rollups** are aggregated representations of time series data, allowing for efficient storage and querying at different granularities.
#### Struct: `MetricTimeSeriesRollup`
```thrift
struct MetricTimeSeriesRollup {
  10: required ID.ID mtsId,
  20: required DataTypes.RollupType rollup
}
```
- **mtsId**: Identifier for the metric-time series.
- **rollup**: Type of aggregation applied (e.g., SUM, AVG).
---
## Thrift Definitions and API
Thrift is used to define the service interfaces and data structures for TSDB interactions.
### Namespaces and Includes
```thrift
include "SignalFuseService.thrift"
include "SignalFuseErrors.thrift"
include "ID.thrift"
include "ValueTypes.thrift"
include "MetabaseTypes.thrift"
include "DataTypes.thrift"
include "EventTypes.thrift"

namespace java sf.timeseries.thrift
namespace cpp sf.timeseries.thrift
namespace py sf.timeseries.thrift
namespace rb sf.timeseries.thrift
```
- **Includes**: Import necessary Thrift definitions.
- **Namespaces**: Define namespaces for different programming languages.
### Structs and Enums
Key data structures (`struct`) and enumerations (`enum`) define the schema for time series data and possible exception types.
#### Example: `TsdbExceptionType`
```thrift
enum TsdbExceptionType {
    BACKEND_READ_FAILURE = 10,
    INVALID_RESOLUTION = 20,
    BACKEND_WRITE_FAILURE = 30,
    RESULT_SET_TOO_LARGE = 40,
    BLOCK_DECODE = 50,
    BLOCK_ENCODE = 60,
    INVALID_TIMESTAMP = 70,
    INVALID_SURVEY_OPTIONS = 80,
    DEPRECATED_UNRETRYABLE = 90,
    REJECTED_EXECUTION = 100,
    UNHANDLED_EXCEPTION = 110,
    RATE_LIMITED = 120
}
```
### Service Interfaces
Defines the methods available for interacting with the TSDB.
#### Service: `Tsdb`
```thrift
service Tsdb extends SignalFuseService.SignalFuseService {
  TimeSeriesResult getTimeSeriesByIds(10: set<MetricTimeSeriesRollup> mtsrs, 20: TsdbQueryParams queryParams) throws (10: TsdbException te);
  
  // Other methods...
}
```

#### Service: `Agamemnon`
Extends `Tsdb` with additional shard-based operations.
```thrift
service Agamemnon extends Tsdb {
  map<i64, list<binary>> diffTimeSeriesShardGrain(10: i64 resolution, 20: i64 startTime, 30: i64 endTime, 40: i32 shard, 50: i32 grain) throws (10: TsdbException tw);
  
  // Other methods...
}
```

---

## Internals of TSDB

### Data Ingestion Pipeline
1. **Data Ingestion**: Incoming data is received via APIs like `putTimeSeriesBlocks`.
2. **Parsing and Validation**: Data is parsed into `Datum` or `DatumEx` structures and validated.
3. **Batching**: Data points are batched into `DataBatch` for efficient processing.
4. **Storage**: Batched data is written to the storage engine, handling compression and indexing.

### Storage Engine
- **Columnar Storage**: Optimized for time series data with columnar formats.
- **Indexing**: Time-based indexing for efficient range queries.
- **Compression**: Techniques like Gorilla compression to reduce storage footprint.
- **Replication and Sharding**: Data is replicated and sharded across nodes for fault tolerance and scalability.

### Query Processing
1. **Query Parsing**: API requests are parsed into `TsdbQueryParams`.
2. **Optimization**: Queries are optimized for the best execution path.
3. **Execution**: Data is fetched from storage, applying any necessary rollups or aggregations.
4. **Result Formation**: Results are structured into `TimeSeriesResult` for client consumption.

---

## Exception Handling

TSDB defines specific exceptions to handle various error scenarios.

### Struct: `TsdbException`

```thrift
exception TsdbException {
  10: i32 errorCode,
  20: string message,
  30: optional bool retryable = true,
  40: optional i64 retryTimestamp
}
```

- **errorCode**: Numeric code representing the error type.
- **message**: Descriptive error message.
- **retryable**: Indicates if the operation can be retried.
- **retryTimestamp**: Suggested timestamp for retry.

### Struct: `TsdbWriteException`

```thrift
exception TsdbWriteException {
  10: string message,
  20: map<DatumEx, TsdbException> errors
}
```

- **message**: Description of the write exception.
- **errors**: Map of data points to their respective exceptions.

---

## Advanced Features

### Distributed Systems Considerations

- **Sharding**: Distributing data across multiple nodes to balance load.
- **Replication**: Ensuring data is copied across nodes for high availability.
- **Consistency Models**: Managing data consistency across replicas (e.g., eventual consistency).

### Scalability and Reliability

- **Horizontal Scaling**: Adding more nodes to handle increased load.
- **Fault Tolerance**: Designing systems to handle node failures gracefully.
- **Load Balancing**: Distributing incoming requests evenly across nodes.

---

## Code Examples

### Thrift Service Implementation

#### Java Example

```java
public class TsdbServiceImpl implements Tsdb.Iface {
    @Override
    public TimeSeriesResult getTimeSeriesByIds(Set<MetricTimeSeriesRollup> mtsrs, TsdbQueryParams queryParams) throws TsdbException {
        // Implementation logic
    }
    
    // Other method implementations...
}
```

### Data Ingestion Example

#### Python Example

```python
from sf.timeseries.thrift import Tsdb
from sf.timeseries.thrift.ttypes import Datum, DataBatch, ID

def ingest_data(tsdb_client, mts_id, timestamp, value):
    datum = Datum(tsId=ID(id=mts_id), timestampMs=timestamp, value=value)
    batch = DataBatch(timestampMs=timestamp, data=[datum])
    try:
        tsdb_client.putTimeSeriesBlocks(resolutionMs=1000, blocks=[batch])
    except TsdbException as e:
        print(f"Error ingesting data: {e.message}")
```

### Querying Data

#### C++ Example

```cpp
#include "Tsdb.h"

void query_time_series(TsdbClient& client, std::set<MetricTimeSeriesRollup> mtsrs, TsdbQueryParams params) {
    try {
        TimeSeriesResult result = client.getTimeSeriesByIds(mtsrs, params);
        // Process result
    } catch (TsdbException& e) {
        std::cerr << "Query failed: " << e.message << std::endl;
    }
}
```

---

## Equations and Algorithms

### Time Range Alignment

To align a timestamp to the nearest lower multiple of resolution:

\[
\text{aligned\_time} = \left\lfloor \frac{\text{timestamp}}{\text{resolution}} \right\rfloor \times \text{resolution}
\]

### Sharding Algorithm

Distribute MTS IDs across shards using modulo operation:

\[
\text{shard\_id} = \text{mtsId} \mod \text{number\_of\_shards}
\]

### Replication Factor

Ensure each data block is replicated across multiple nodes:

\[
\text{replication\_factor} = 3 \quad (\text{example})
\]

---

## ASCII Diagrams

### Data Ingestion Pipeline

```
+-------------+      +--------------+      +------------+      +--------------+
| Data Source | ---> | Ingestion API| ---> | Validator  | ---> | Storage Engine|
+-------------+      +--------------+      +------------+      +--------------+
```

### Sharding and Replication

```
+----------+       +----------+       +----------+
| Shard 1  | <---> | Shard 2  | <---> | Shard 3  |
| Replica1 |       | Replica1 |       | Replica1 |
| Replica2 |       | Replica2 |       | Replica2 |
+----------+       +----------+       +----------+
```

### Query Processing Flow

```
+---------+      +-----------+      +--------------+      +-----------+
| Client  | ---> | Query API | ---> | Query Engine | ---> | Storage   |
+---------+      +-----------+      +--------------+      +-----------+
                                                          |
                                                          v
                                                 +----------------+
                                                 | Indexing Layer |
                                                 +----------------+
```

---

## Practical Applications

- **Monitoring Systems**: Collecting metrics from servers, applications, and network devices.
- **Financial Services**: Tracking stock prices, trading volumes, and other financial indicators.
- **IoT Devices**: Managing data from sensors, smart devices, and industrial equipment.
- **Energy Sector**: Monitoring energy consumption, production metrics, and grid data.
- **Healthcare**: Recording patient vitals, medical device data, and other health-related metrics.

---

## References

- **Thrift Documentation**: [Apache Thrift](https://thrift.apache.org/docs)
- **Time Series Databases**: 
  - [InfluxDB](https://www.influxdata.com/)
  - [Prometheus](https://prometheus.io/)
  - [TimescaleDB](https://www.timescale.com/)
- **Distributed Systems**: 
  - [Designing Data-Intensive Applications](https://dataintensive.net/)
  - [Distributed Systems: Principles and Paradigms](https://www.distributed-systems.net/)
- **Compression Algorithms**:
  - [Gorilla Compression](https://www.vldb.org/pvldb/vol8/p1816-teller.pdf)

---

# Detailed Breakdown of "Time Series Databases and Its Internals"

## 1. Introduction to Time Series Databases (TSDB)

Time Series Databases (TSDBs) are specialized databases optimized for handling time-stamped data. They are crucial in applications where data points are continuously collected over time, such as monitoring, IoT, financial analysis, and more.

### Why Use TSDB?

- **Efficient Storage**: Optimized for time-based data, enabling high compression rates.
- **Fast Ingestion**: Capable of handling high write throughput.
- **Optimized Queries**: Designed for efficient range queries and aggregations over time intervals.
- **Scalability**: Easily scalable to accommodate growing volumes of data.

### Common Use Cases

- **System Monitoring**: Collecting metrics like CPU usage, memory consumption, and network traffic.
- **IoT Applications**: Gathering sensor data from devices.
- **Financial Data**: Tracking stock prices, trading volumes, and other financial indicators.
- **Application Performance Monitoring (APM)**: Monitoring application metrics and logs.

## 2. Core Concepts and Data Structures

### Time Series Data Model

A time series is a collection of data points indexed by time. Each data point typically consists of a timestamp and a value.

#### Key Components:

- **Metric**: Represents what is being measured (e.g., CPU usage).
- **Tags/Labels**: Metadata to categorize metrics (e.g., host, region).
- **Timestamp**: The point in time the data was recorded.
- **Value**: The measurement at the given timestamp.

#### Example

```
Metric: cpu_usage
Tags: {host: server1, region: us-west}
Data Points:
  - (1617184800000, 55.5)
  - (1617184860000, 60.2)
  - (1617184920000, 58.1)
```

### Data Points and Batching

**Data Points** represent individual measurements within a time series. To optimize performance, data points are often grouped into batches before being written to storage.

#### Struct: `Datum`

```thrift
struct Datum {
  10: required ID.ID tsId,
  20: required i64 timestampMs,
  30: optional ValueTypes.Value value,
  40: optional bool synthetic = false
}
```

- **tsId**: Identifier of the time series.
- **timestampMs**: Timestamp in milliseconds since epoch.
- **value**: The measured value.
- **synthetic**: Indicates if the data point is synthetic (e.g., interpolated).

#### Struct: `DataBatch`

```thrift
struct DataBatch {
  10: i64 timestampMs,
  20: list<Datum> data,
  30: optional ID.ID traceId,
  31: optional map<string, string> traceContext,
  40: optional i64 maxDelayMs
}
```

- **timestampMs**: Logical timestamp for the batch.
- **data**: List of `Datum` objects.
- **traceId & traceContext**: For tracing the data flow.
- **maxDelayMs**: Maximum allowed delay for processing.

### Rollups and Aggregations

**Rollups** are pre-aggregated summaries of time series data, enabling efficient storage and faster query responses for aggregated data.

#### Struct: `MetricTimeSeriesRollup`

```thrift
struct MetricTimeSeriesRollup {
  10: required ID.ID mtsId,
  20: required DataTypes.RollupType rollup
}
```

- **mtsId**: Metric-Time Series ID.
- **rollup**: Type of aggregation (e.g., SUM, AVG).

### Struct: `TimeSeries`

```thrift
struct TimeSeries {
  10: required ID.ID mtsId,
  12: required DataTypes.RollupType rollup,
  15: required i64 resolutionMs = 0,
  20: required list<ValueTypes.TimeValue> timeValues
}
```

- **mtsId**: Identifier for the metric-time series.
- **rollup**: Aggregation type applied.
- **resolutionMs**: Data point resolution in milliseconds.
- **timeValues**: Ordered list of timestamp-value pairs.

## 3. Thrift Definitions and API

Thrift is used to define the interfaces and data structures for TSDB services, enabling cross-language compatibility.

### Namespaces and Includes

```thrift
include "SignalFuseService.thrift"
include "SignalFuseErrors.thrift"
include "ID.thrift"
include "ValueTypes.thrift"
include "MetabaseTypes.thrift"
include "DataTypes.thrift"
include "EventTypes.thrift"

namespace java sf.timeseries.thrift
namespace cpp sf.timeseries.thrift
namespace py sf.timeseries.thrift
namespace rb sf.timeseries.thrift
```

- **Includes**: Import definitions from other Thrift files.
- **Namespaces**: Define namespaces for different programming languages (Java, C++, Python, Ruby).

### Structs and Enums

Defines the data models and enumerations used across the TSDB services.

#### Example: `TsdbExceptionType`

```thrift
enum TsdbExceptionType {
    BACKEND_READ_FAILURE = 10,
    INVALID_RESOLUTION = 20,
    BACKEND_WRITE_FAILURE = 30,
    RESULT_SET_TOO_LARGE = 40,
    BLOCK_DECODE = 50,
    BLOCK_ENCODE = 60,
    INVALID_TIMESTAMP = 70,
    INVALID_SURVEY_OPTIONS = 80,
    DEPRECATED_UNRETRYABLE = 90,
    REJECTED_EXECUTION = 100,
    UNHANDLED_EXCEPTION = 110,
    RATE_LIMITED = 120
}
```

- Enumerates various exception types that can occur within the TSDB.

### Service Interfaces

Defines the methods exposed by the TSDB services for data ingestion, querying, and maintenance.

#### Service: `Tsdb`

```thrift
service Tsdb extends SignalFuseService.SignalFuseService {
  TimeSeriesResult getTimeSeriesByIds(10: set<MetricTimeSeriesRollup> mtsrs, 20: TsdbQueryParams queryParams) throws (10: TsdbException te);
  
  // Additional methods...
}
```

#### Service: `Agamemnon`

Extends `Tsdb` with shard-specific operations, allowing for more granular control and maintenance.

```thrift
service Agamemnon extends Tsdb {
  map<i64, list<binary>> diffTimeSeriesShardGrain(10: i64 resolution, 20: i64 startTime, 30: i64 endTime, 40: i32 shard, 50: i32 grain) throws (10: TsdbException tw);
  
  // Additional methods...
}
```

## 4. Internals of TSDB

### Data Ingestion Pipeline

1. **Data Reception**: Data is received via APIs like `putTimeSeriesBlocks`.
2. **Parsing and Validation**: Incoming data is parsed into `Datum` or `DatumEx` structures and validated for correctness.
3. **Batching**: Data points are grouped into `DataBatch` structures to optimize write operations.
4. **Storage Allocation**: Batches are assigned to appropriate shards and replicas.
5. **Writing to Storage**: Data is written to the storage engine, handling compression and indexing.

#### Diagram: Data Ingestion Pipeline

```
+-------------+      +--------------+      +------------+      +--------------+
| Data Source | ---> | Ingestion API| ---> | Validator  | ---> | Storage Engine|
+-------------+      +--------------+      +------------+      +--------------+
```

### Storage Engine

The storage engine is the core component responsible for persisting time series data efficiently.

#### Key Features:

- **Columnar Storage**: Stores data in columns for efficient compression and retrieval.
- **Indexing**: Utilizes time-based and tag-based indexing for quick access.
- **Compression**: Implements algorithms like Gorilla compression to reduce storage size.
- **Sharding and Replication**: Distributes data across multiple nodes for scalability and fault tolerance.

### Query Processing

1. **Query Reception**: Queries are received via APIs like `getTimeSeriesData`.
2. **Parsing and Validation**: Query parameters are parsed into `TsdbQueryParams` structures.
3. **Optimization**: The query planner optimizes the query execution path.
4. **Data Retrieval**: Data is fetched from the storage engine, applying any necessary rollups or aggregations.
5. **Result Formation**: Results are structured into `TimeSeriesResult` objects for client consumption.

#### Diagram: Query Processing Flow

```
+---------+      +-----------+      +--------------+      +-----------+
| Client  | ---> | Query API | ---> | Query Engine | ---> | Storage   |
+---------+      +-----------+      +--------------+      +-----------+
                                                          |
                                                          v
                                                 +----------------+
                                                 | Indexing Layer |
                                                 +----------------+
```

## 5. Exception Handling

Proper exception handling is crucial for maintaining the reliability and robustness of TSDB services.

### Struct: `TsdbException`

```thrift
exception TsdbException {
  10: i32 errorCode,
  20: string message,
  30: optional bool retryable = true,
  40: optional i64 retryTimestamp
}
```

- **errorCode**: Numerical code representing the error type.
- **message**: Detailed error message.
- **retryable**: Indicates if the operation can be retried.
- **retryTimestamp**: Suggests when to retry the operation.

### Struct: `TsdbWriteException`

```thrift
exception TsdbWriteException {
  10: string message,
  20: map<DatumEx, TsdbException> errors
}
```

- **message**: Description of the write exception.
- **errors**: Map linking each `DatumEx` to its respective `TsdbException`.

### Example: Handling Exceptions in Code

#### Java Example

```java
try {
    TimeSeriesResult result = tsdbClient.getTimeSeriesByIds(mtsrs, queryParams);
    // Process result
} catch (TsdbException e) {
    if (e.isRetryable()) {
        // Implement retry logic
    } else {
        // Handle non-retryable error
    }
}
```

## 6. Advanced Features

### Distributed Systems Considerations

- **Sharding**: Dividing data into distinct shards to distribute load.
- **Replication**: Maintaining multiple copies of data across different nodes for fault tolerance.
- **Consistency Models**: Ensuring data consistency across replicas (e.g., eventual consistency).

### Scalability and Reliability

- **Horizontal Scaling**: Adding more nodes to handle increased data volume and query load.
- **Load Balancing**: Distributing incoming requests evenly across nodes to prevent bottlenecks.
- **Fault Tolerance**: Designing the system to continue operating smoothly in the event of node failures.

#### Sharding Algorithm Example

```python
def get_shard_id(mts_id, number_of_shards):
    return mts_id % number_of_shards
```

### Caching Strategies

- **Distributed Cache**: Utilizing caches like Redis or Memcached to store frequently accessed data.
- **In-Memory Caching**: Keeping hot data in memory for faster access.

### Data Compression

- **Gorilla Compression**: A compression algorithm optimized for time series data.
  
  **Gorilla Encoding Steps:**
  1. Store the first timestamp and value as raw bits.
  2. For subsequent data points, store the delta of timestamps and XOR of values.
  3. Use variable-length encoding to minimize storage.

### Data Retention Policies

- **Retention Periods**: Defining how long data is kept before being purged.
- **Downsampling**: Aggregating older data to lower resolutions to save space.

---

## 7. Code Examples

### Thrift Service Implementation

#### Java Implementation of `Tsdb` Service

```java
public class TsdbServiceImpl implements Tsdb.Iface {
    @Override
    public TimeSeriesResult getTimeSeriesByIds(Set<MetricTimeSeriesRollup> mtsrs, TsdbQueryParams queryParams) throws TsdbException {
        // Validate input
        if (mtsrs.isEmpty()) {
            throw new TsdbException(400, "No MetricTimeSeriesRollup provided", false, 0);
        }
        
        // Fetch data from storage
        Map<MetricTimeSeriesRollup, TimeSeries> data = fetchTimeSeriesData(mtsrs, queryParams);
        List<TsdbException> errors = new ArrayList<>();
        
        // Handle potential errors
        // ...
        
        return new TimeSeriesResult(data, errors);
    }
    
    // Other method implementations...
    
    private Map<MetricTimeSeriesRollup, TimeSeries> fetchTimeSeriesData(Set<MetricTimeSeriesRollup> mtsrs, TsdbQueryParams params) {
        // Implementation logic to fetch data
        return new HashMap<>();
    }
}
```

### Data Ingestion Example

#### Python Ingestion Script

```python
from sf.timeseries.thrift import Tsdb
from sf.timeseries.thrift.ttypes import Datum, DataBatch, ID, TsdbException
from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol

def ingest_data(tsdb_client, mts_id, timestamp, value):
    try:
        datum = Datum(tsId=ID(id=mts_id), timestampMs=timestamp, value=value)
        batch = DataBatch(timestampMs=timestamp, data=[datum])
        tsdb_client.putTimeSeriesBlocks(resolutionMs=1000, blocks=[batch])
        print("Data ingested successfully.")
    except TsdbException as e:
        print(f"Error ingesting data: {e.message}")

def main():
    transport = TSocket.TSocket('localhost', 9090)
    transport = TTransport.TBufferedTransport(transport)
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    client = Tsdb.Client(protocol)
    
    transport.open()
    ingest_data(client, mts_id=12345, timestamp=1617184800000, value=55.5)
    transport.close()

if __name__ == "__main__":
    main()
```

### Querying Data

#### C++ Query Example

```cpp
#include "Tsdb.h"
#include <iostream>

void query_time_series(TsdbClient& client, std::set<MetricTimeSeriesRollup> mtsrs, TsdbQueryParams params) {
    try {
        TimeSeriesResult result = client.getTimeSeriesByIds(mtsrs, params);
        // Iterate through the results
        for (const auto& [mtsRollup, timeSeries] : result.data) {
            std::cout << "MTS ID: " << mtsRollup.mtsId.id << ", Rollup: " << mtsRollup.rollup << std::endl;
            for (const auto& timeValue : timeSeries.timeValues) {
                std::cout << "Timestamp: " << timeValue.timestampMs << ", Value: " << timeValue.value << std::endl;
            }
        }
    } catch (TsdbException& e) {
        std::cerr << "Query failed: " << e.message << std::endl;
    }
}

int main() {
    // Initialize client and parameters
    TsdbClient client("localhost", 9090);
    std::set<MetricTimeSeriesRollup> mtsrs = { MetricTimeSeriesRollup(ID(12345), RollupType::SUM) };
    TsdbQueryParams params;
    params.resolutionMs = 1000;
    params.timeRange = TsdbQueryTimeRange(1617184800000, 1617188400000, 0);
    
    query_time_series(client, mtsrs, params);
    return 0;
}
```

---

## 8. Equations and Algorithms

### Time Range Alignment

Aligning a timestamp to the nearest lower multiple of resolution:

\[
\text{aligned\_time} = \left\lfloor \frac{\text{timestamp}}{\text{resolution}} \right\rfloor \times \text{resolution}
\]

**Example:**

- **Timestamp**: 1617184865000 ms
- **Resolution**: 1000 ms
- **Aligned Time**: \( \left\lfloor \frac{1617184865000}{1000} \right\rfloor \times 1000 = 1617184865000 \) ms

### Sharding Algorithm

Distribute MTS IDs across shards to balance load.

\[
\text{shard\_id} = \text{mtsId} \mod \text{number\_of\_shards}
\]

**Example:**

- **mtsId**: 12345
- **Number of Shards**: 10
- **shard_id**: \( 12345 \mod 10 = 5 \)

### Compression Ratio

Calculating the compression ratio achieved by the storage engine.

\[
\text{Compression Ratio} = \frac{\text{Original Size}}{\text{Compressed Size}}
\]

**Example:**

- **Original Size**: 1000 MB
- **Compressed Size**: 250 MB
- **Compression Ratio**: \( \frac{1000}{250} = 4 \)

### Query Performance

Estimating query execution time based on data volume and indexing.

\[
\text{Query Time} = \frac{\text{Number of Data Points}}{\text{Indexing Factor}} \times \text{Lookup Time}
\]

**Variables:**

- **Number of Data Points**: Total points to scan.
- **Indexing Factor**: Efficiency factor due to indexing.
- **Lookup Time**: Time per data point lookup.

---

## 9. ASCII Diagrams

### Sharding and Replication Architecture

```
+-----------+       +-----------+       +-----------+
| Shard 0   |       | Shard 1   |       | Shard N   |
| Replica 1 |       | Replica 1 |       | Replica 1 |
| Replica 2 |       | Replica 2 |       | Replica 2 |
+-----------+       +-----------+       +-----------+
```

### Data Flow Diagram

```
+----------+      +----------+      +----------+      +----------+
| Producer | ---> | Ingestion| ---> | Validator| ---> | Storage  |
+----------+      +----------+      +----------+      +----------+
                                                        |
                                                        v
                                                 +--------------+
                                                 | Compression  |
                                                 +--------------+
                                                        |
                                                        v
                                                 +--------------+
                                                 | Indexing     |
                                                 +--------------+
```

### Query Execution Flow

```
+---------+      +-----------+      +-----------+      +-----------+
| Client  | ---> | Query API | ---> | Query Planner| --> | Execution |
+---------+      +-----------+      +-----------+      +-----------+
                                                          |
                                                          v
                                                 +----------------+
                                                 | Storage Engine |
                                                 +----------------+
```

## 10. Practical Applications

### System Monitoring

- **Use Case**: Monitoring server health, application performance, and network traffic.
- **Metrics**: CPU usage, memory consumption, request rates, error rates.
- **Example**: Using Prometheus (a TSDB) to collect and query metrics.

### IoT Data Management

- **Use Case**: Collecting data from sensors and devices.
- **Metrics**: Temperature, humidity, device status, energy consumption.
- **Example**: InfluxDB used in smart home systems to monitor environmental conditions.

### Financial Analysis

- **Use Case**: Tracking stock prices, trading volumes, and financial indicators.
- **Metrics**: Open, high, low, close (OHLC) prices, volume traded.
- **Example**: TimescaleDB used for storing and analyzing financial market data.

### Application Performance Monitoring (APM)

- **Use Case**: Monitoring application metrics and logs to ensure optimal performance.
- **Metrics**: Response times, throughput, error rates, resource utilization.
- **Example**: Using Elasticsearch (with time series capabilities) to aggregate and query application logs.

---

## 11. References

- **Apache Thrift Documentation**: [https://thrift.apache.org/docs](https://thrift.apache.org/docs)
- **InfluxDB Documentation**: [https://docs.influxdata.com/influxdb/](https://docs.influxdata.com/influxdb/)
- **Prometheus Documentation**: [https://prometheus.io/docs/introduction/overview/](https://prometheus.io/docs/introduction/overview/)
- **TimescaleDB Documentation**: [https://docs.timescale.com/](https://docs.timescale.com/)
- **Gorilla Compression Paper**: [https://www.vldb.org/pvldb/vol8/p1816-teller.pdf](https://www.vldb.org/pvldb/vol8/p1816-teller.pdf)
- **Designing Data-Intensive Applications** by Martin Kleppmann

---

# Appendix

## Thrift Definitions Overview

### Included Files

- **SignalFuseService.thrift**: Base service definitions.
- **SignalFuseErrors.thrift**: Error and exception definitions.
- **ID.thrift**: Definitions for unique identifiers.
- **ValueTypes.thrift**: Definitions for value types used in data points.
- **MetabaseTypes.thrift**: Metadata-related type definitions.
- **DataTypes.thrift**: Various data type definitions used across services.
- **EventTypes.thrift**: Event-related type definitions.

### Namespace Definitions

```thrift
namespace java sf.timeseries.thrift
namespace cpp sf.timeseries.thrift
namespace py sf.timeseries.thrift
namespace rb sf.timeseries.thrift
```

- **java**: Java namespace.
- **cpp**: C++ namespace.
- **py**: Python namespace.
- **rb**: Ruby namespace.

These namespaces ensure that the generated code adheres to language-specific conventions and packaging.

---

## Detailed Struct Descriptions

### `SourceMetric`

```thrift
struct SourceMetric {
  10: string source,
  20: string metric
}
```

- **source**: Origin of the metric (e.g., application name).
- **metric**: Name of the metric (e.g., "cpu_usage").

### `TimeSeries`

A core structure representing a single time series with multiple data points.

```thrift
struct TimeSeries {
  10: required ID.ID mtsId,
  12: required DataTypes.RollupType rollup,
  15: required i64 resolutionMs = 0,
  20: required list<ValueTypes.TimeValue> timeValues
}
```

- **mtsId**: Unique identifier for the metric-time series.
- **rollup**: Type of aggregation applied.
- **resolutionMs**: Data resolution in milliseconds.
- **timeValues**: Ordered list of timestamp-value pairs.

### `DatumEx`

Extended data point structure for more detailed information.

```thrift
struct DatumEx {
  10: required ID.ID mtsId,
  15: required DataTypes.RollupType rollup,
  20: required i64 timestampMs,
  30: optional ValueTypes.Value value,
  40: optional bool synthetic = false,
  50: optional ID.ID orgId,
  60: optional ID.ID tokenId
}
```

- **mtsId**: Metric-Time Series ID.
- **rollup**: Aggregation type.
- **timestampMs**: Timestamp in milliseconds.
- **value**: Measured value.
- **synthetic**: Indicates if the data point is synthetic.
- **orgId**: Organization identifier.
- **tokenId**: Token identifier for authentication or tracking.

### `TimeSeriesResult`

Result structure for time series queries.

```thrift
struct TimeSeriesResult {
  10: required map<MetricTimeSeriesRollup, TimeSeries> data,
  20: required list<TsdbException> errors
}
```

- **data**: Map of MetricTimeSeriesRollup to TimeSeries.
- **errors**: List of exceptions encountered during the query.

### `TsdbQueryParams`

Parameters for querying the TSDB.

```thrift
struct TsdbQueryParams {
  10: required i64 resolutionMs,
  20: required TsdbQueryTimeRange timeRange
}
```

- **resolutionMs**: Desired data resolution in milliseconds.
- **timeRange**: Time range for the query.

---

## Additional ASCII Diagrams

### Exception Handling Flow

```
+---------+      +------------+      +----------------+
| Client  | ---> | API Server | ---> | Service Method |
+---------+      +------------+      +----------------+
                                          |
                             +------------+-------------+
                             | TsdbException Thrift      |
                             | - errorCode               |
                             | - message                 |
                             | - retryable               |
                             +------------+-------------+
                                          |
                                          v
                                +---------------------+
                                | Client Exception    |
                                | Handling Logic      |
                                +---------------------+
```

### Replication Workflow

```
+---------+        +---------+        +---------+
| Primary | -----> | Replica | -----> | Replica |
| Node    |        | Node    |        | Node    |
+---------+        +---------+        +---------+
     |                  |                  |
     v                  v                  v
+---------+        +---------+        +---------+
| Write   |        | Write   |        | Write   |
| Operation|       | Operation|       | Operation|
+---------+        +---------+        +---------+
```

---

## Glossary

- **TSDB**: Time Series Database.
- **MTS**: Metric-Time Series.
- **Thrift**: A software framework for scalable cross-language services development.
- **Rollup**: Aggregated summary of data points.
- **Shard**: A horizontal partition of data in a database.
- **Replication**: Duplication of data across multiple nodes.
- **Gorilla Compression**: A compression algorithm optimized for time series data.
- **Indexing**: Process of creating data structures to enable fast data retrieval.
- **Downsampling**: Reducing data resolution by aggregating data points.

---

# Conclusion

Understanding the intricacies of Time Series Databases and their internal mechanisms is crucial for designing scalable, efficient, and reliable data systems. This comprehensive overview covers the foundational concepts, data structures, API definitions, internal workflows, and advanced features essential for mastering TSDBs at a Staff+ level. By leveraging these insights, engineers can build robust solutions capable of handling vast amounts of time-stamped data with precision and performance.