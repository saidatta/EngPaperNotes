In **Chapter 1**, we introduced a simple use case of delivering real-time data to consumers and discussed how connectors can convert data at rest into data in motion (event streams) and publish them into topics in streaming platforms.

In this chapter, we'll delve deeper into **stream processing platforms**, focusing on:

- The importance of cleansing, preparing, and enriching event data.
- The role of stateful transformations in streaming data pipelines.
- The differences between ETL and ELT in streaming contexts.
- How data moves from operational to analytical data planes.
- Overcoming ELT limitations in streaming real-time use cases.

---

## The Need for Data Preparation in Event Streams

Event streams, once read, often aren't immediately usable by consumers. They usually require:

- **Cleansing**: Correcting or removing erroneous data.
- **Preparation**: Structuring data for analysis.
- **Enrichment**: Adding contextual information.

### Importance of Data Preparation

- **Improved Data Quality**: Addresses issues like missing values, inconsistencies, duplicates, and outliers.
- **Enhanced Performance**: Optimizes data layout, indexing, and partitioning for efficient retrieval and processing.
- **Data Governance and Compliance**: Enforces security measures, anonymizes sensitive info, and adheres to privacy regulations.

> **Note**: Techniques like denormalization, columnar storage, and indexing strategies will be covered in **Chapter 4**.

---

## Role of Stream Processing Platforms

The **stream processing platform** acts as an intermediary between event streams and the destination data store, performing necessary transformations.

![Stream Processing Platform](stream_processing_platform.png)

*Figure: Cleanse, prepare, and enrich event data prior to reaching the destination.*

### Why Early Transformation Matters

- **Resource Efficiency**: Reduces the workload on destination OLAP systems.
- **Incremental Processing**: Handling small data chunks is less resource-intensive.
- **Optimizing OLAP Resources**: OLAP systems focus on serving analytical queries quickly.

---

## OLAP vs. OLTP Data Stores

- **OLAP (Online Analytical Processing)**:
  - Optimized for analytical reads.
  - Typically use columnar-based storage.
  - Power dashboards and analytical applications.
- **OLTP (Online Transaction Processing)**:
  - Capture information from applications (event sources).
  - Optimized for writing and single-row lookups.
  - Source for Change Data Capture (CDC) extraction.

---

## Stateful Transformations

Complex transformations that require maintaining state across multiple events are **stateful transformations**. Examples include:

- **Joins**: Combining data from different streams.
- **Aggregations**: Calculating sums, averages, counts, etc.
- **Windowing**: Applying functions over time or count-based windows.

### Why State Stores are Needed

Stateful transformations need to remember intermediate data, which requires a **state store** to hold this information.

---

## Examples of Stateful Transformations

### Rolling Averages

To compute a rolling average over a window of size **N**:

\[
\text{Rolling Average} = \frac{1}{N} \sum_{i=0}^{N-1} x_i
\]

- **Requires**: Maintaining a sum and count of previous data points.
- **Use Case**: Monitoring average user activity over time.

### Sessionization

Grouping events into sessions based on user activity within a time threshold.

- **Requires**: Tracking session start/end times and associated events.
- **Use Case**: Web analytics to understand user behavior.

### Deduplication

Removing duplicate events in a stream.

- **Requires**: Maintaining a set of seen event IDs.
- **Use Case**: Ensuring data integrity when events might be duplicated.

### Windowed Aggregations

Aggregations over time or count-based windows.

- **Requires**: Accumulating values within each window.
- **Use Case**: Real-time monitoring of metrics like sum, count, or max.

---

## Limitations of Pure Streaming Platforms

Platforms like **Apache Kafka** don't natively support stateful transformations.

- **Solution**: Use additional components like **Kafka Streams** or **Apache Flink**.

---

## Data Pipelines: Connecting Operational and Analytical Data Planes

![Data Pipeline](data_pipeline.png)

*Figure: Streaming data pipelines follow the plumbing metaphor.*

### The Plumbing Metaphor

- **Pipes**: Data pipelines.
- **Water**: Data flowing through pipelines.
- **Filters/Heaters**: Transformations applied to data.

### Operational vs. Analytical Data Planes

- **Operational Data Plane**:
  - Applications, microservices, OLTP databases.
  - Data sources generating events.
- **Analytical Data Plane**:
  - Analytical systems, OLAP data stores.
  - Consumers of transformed data for analysis.

---

## ETL in Streaming Data Pipelines

### ETL Process

1. **Extract**:
   - Data is extracted from source systems (e.g., OLTP databases).
   - Handled by **source connectors**.

2. **Transform**:
   - Performed in the stream processing platform.
   - Data is cleansed, prepared, and enriched.
   - **Stateful transformations** are applied.

3. **Load**:
   - Transformed data is loaded into the OLAP data store.
   - Data is ready for analytical queries.

![ETL Pipeline](etl_pipeline.png)

*Figure: An ETL data pipeline extracting from an OLTP source, performing stateful transformations, and writing to an OLAP datastore.*

### Advantages of ETL in Streaming

- **Real-Time Processing**: Data is transformed while in motion.
- **Resource Optimization**: Reduces load on OLAP systems.
- **Improved Data Quality**: Ensures data is ready for analysis upon arrival.

---

## Use Case: Clickstream Data Enrichment

### Scenario

A customer clicks on a green T-shirt in a mobile app.

### Data Flow

1. **Click Events**:
   - Captured by a microservice.
   - Sent to a **"Click Event Topic"** in the streaming platform.

2. **Product and Customer Data**:
   - Stored in an **OLTP database**.
   - Changes captured via **CDC connectors**.
   - Published to **"Products Topic"** and **"Customers Topic"**.

![ETL Use Case](etl_use_case.png)

*Figure: ETL streaming data pipeline with source and sink topics.*

### Stream Processing

- **Stream Processor**:
  - Consumes from **click events**, **products**, and **customers** topics.
  - Performs **stateful transformations** (e.g., joins).
  - Enriches click events with product and customer info.
  - Publishes to a **sink topic**.

- **OLAP Data Store**:
  - Consumes from the **sink topic**.
  - Serves data to dashboards and analytical tools.

---

## ELT Limitations in Streaming

### ELT Process

1. **Extract**:
   - Data is extracted from source systems.

2. **Load**:
   - Data is loaded into the target system without transformation.

3. **Transform**:
   - Transformations are performed within the target system after loading.

![ELT Pipeline](elt_pipeline.png)

*Figure: An ELT data pipeline forcing transformations in the destination data store.*

### Challenges in Streaming Context

- **Batch-Oriented**: ELT assumes data batches, not continuous streams.
- **Delayed Transformations**: Increases latency.
- **Resource Intensive**: Overloads OLAP systems with heavy transformations.

![ELT Limitations](elt_limitations.png)

*Figure: ELT pipeline failing to transform events in real-time.*

---

## Overcoming ELT Limitations in Streaming

### Adapting ELT for Streaming

Use a **stream processing platform** as the target system in ELT.

![Streaming ELT](streaming_elt.png)

*Figure: ELT with a stream processing platform as the target.*

### Steps

1. **Extract**:
   - Data is extracted from streaming sources.

2. **Load**:
   - Data is loaded into a **stream processing platform**.

3. **Transform**:
   - Transformations are performed within the platform.
   - Transformed data is then sent to the destination data store.

### Considerations

- **Stateful Support**: Platform must support stateful transformations.
- **Scalability**: Must handle high-velocity data streams.
- **Persistence**: Streaming databases can provide a persistence layer.

---

## Code Examples

### Rolling Average with Apache Flink

```java
DataStream<Double> rollingAverage = inputStream
    .keyBy(value -> 1) // Single key for all data
    .countWindow(5, 1) // Window of size 5, sliding by 1
    .reduce(new ReduceFunction<Integer>() {
        @Override
        public Integer reduce(Integer value1, Integer value2) {
            return value1 + value2;
        }
    })
    .map(new MapFunction<Integer, Double>() {
        @Override
        public Double map(Integer sum) {
            return sum / 5.0;
        }
    });
```

### Sessionization with Kafka Streams

```java
KStream<String, ClickEvent> clicks = builder.stream("click-events");

KTable<Windowed<String>, Long> sessionCounts = clicks
    .groupByKey()
    .windowedBy(SessionWindows.with(Duration.ofMinutes(30)))
    .count();
```

### Deduplication in Apache Flink

```java
DataStream<Event> uniqueEvents = inputStream
    .keyBy(event -> event.getId())
    .process(new KeyedProcessFunction<String, Event, Event>() {
        private ValueState<Boolean> seen;

        @Override
        public void open(Configuration parameters) {
            seen = getRuntimeContext().getState(new ValueStateDescriptor<>("seen", Boolean.class));
        }

        @Override
        public void processElement(Event event, Context ctx, Collector<Event> out) throws Exception {
            if (seen.value() == null) {
                seen.update(true);
                out.collect(event);
            }
        }
    });
```

---

## Mathematical Concepts

### Sliding Window Aggregations

- **Time-Based Windows**:
  - Fixed windows (e.g., every 5 minutes).
  - Sliding windows with overlaps.
- **Count-Based Windows**:
  - Windows that process a fixed number of events.

**Aggregation Function**:

For a window \( W \) over data points \( x_i \):

\[
\text{Aggregate}(W) = f(x_{i}, x_{i+1}, \dots, x_{i+N-1})
\]

Where \( f \) is an aggregation function like sum, average, max, etc.

### Stream Joins

- **Stream-Stream Joins**:
  - Joining two streams based on a key and a time window.
- **Stream-Table Joins**:
  - Enriching a stream with data from a table (static or slowly changing).

**Join Condition**:

For streams \( A \) and \( B \):

\[
A \Join B = \{ (a, b) \mid a \in A, b \in B, a.\text{key} = b.\text{key} \}
\]

---

## Key Takeaways

- **Stateful Transformations** are essential for complex data processing in streaming pipelines.
- **ETL** is more suitable for streaming real-time use cases than ELT.
- **Stream Processing Platforms** enable real-time data enrichment before it reaches analytical systems.
- **Optimizing Resources**: Heavy transformations should not burden OLAP systems meant for serving queries.

---

## Conclusion

Stream processing platforms play a crucial role in modern data pipelines, enabling real-time analytics by performing necessary transformations early in the data flow. Understanding the limitations of traditional ELT processes and leveraging stateful stream processing can significantly enhance the efficiency and responsiveness of data-driven applications.

---