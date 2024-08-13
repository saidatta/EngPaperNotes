**Abstract**
Uber's business requires real-time data processing from a diverse range of users, including Uber drivers, riders, restaurants, and more. The company uses open-source technologies and customized solutions to meet the unique requirements and scale of Uber's environment. The paper identifies three scaling challenges that need to be addressed: data scaling, use-case scaling, and user scaling. It also presents the overall architecture of the real-time data infrastructure and explores several important use cases.

## 1 Introduction

### 1.1 Data sources and volume
Uber's data centers generate a vast amount of real-time data, originating from various sources like end-user applications (driver/rider/eater) or backend microservices. The data can be application or system logs, special events for tracking (trip updates, driver status change, order cancellation, etc.), or derived from the Online Transactional Processing (OLTP) database changelog used internally by microservices. As of October 2020, trillions of messages and petabytes of data were generated per day across all regions.

### 1.2 Real-time data processing

Real-time data processing is crucial for Uber's technology stack and has three broad areas:
1.  **Messaging platform** - allows communication between asynchronous producers and subscribers.
2.  **Stream processing** - applies computational logic on streams of messages.
3.  **Online Analytical Processing (OLAP)** - enables analytical queries over all this data in near real time.

### 1.3 Scaling challenges

Three fundamental scaling challenges within Uber are:

1.  **Scaling data** - The incoming real-time data volume grows exponentially. Uber also deploys its infrastructure across various geographical regions, increasing the data volume. Each real-time processing system must handle this increase while maintaining SLA around data freshness, end-to-end latency, and availability.
    
2.  **Scaling use cases** - As Uber's business grows, new use cases emerge, and each part of the organization has varying requirements for real-time data systems.
    
3.  **Scaling users** - The diversity of users interacting with the real-time data system covers a broad spectrum of technical skills, from non-technical personnel to advanced users. The growth of Uber's personnel increases challenges related to user-imposed complexities.
    

### 1.4 Adoption of open-source solutions

To overcome these challenges, Uber adopted open-source solutions, which offered advantages such as development velocity, cost-effectiveness, and the power of crowd innovation. Uber needed technologies that could scale with its data and be extensible enough for integration into the unified real-time data stack.

### 1.5 High-level data flow

Various kinds of analytical data are continuously collected from Uber’s data centers across multiple regions. These streams of raw data form the source of truth for all analytics at Uber. Most of these streams are incrementally archived in batch processing systems and ingested into the data warehouse for machine learning and other data science use cases. The Real-Time Data Infra component continuously processes such data streams for various critical use cases such as dynamic pricing (Surge), intelligent alerting, operational dashboards, etc.

![[High-level data flow at Uber infrastructure]](https://i.imgur.com/Fa5rjvh.png)

### 1.6 Paper organization

The paper covers the requirements derived from the use cases, an overview of the high-level abstractions of the real-time data infrastructure at Uber, the open-source technologies used, the enhancements and improvements on the open source solutions, analysis of several real-time use cases at Uber, important aspects of the real-time data infrastructure, related work, and lessons learned from building and operating real-time systems at Uber. The paper concludes by outlining future work.

## 2. REQUIREMENTS

Real-time data infrastructure use cases each have distinct requirements, including:

1.  **Consistency**: Data must be reliable across regions, with no data loss, and must maintain data quality certification.
2.  **Availability**: Infrastructure should ensure high availability (99.99 percentile guarantee). The loss of availability can significantly impact Uber’s business operations.
3.  **Data Freshness**: For most use cases, data must be available for processing or querying seconds after it's produced.
4.  **Query latency**: Some use cases need quick query execution on raw data streams (p99th query latency under 1 second).
5.  **Scalability**: The system must be able to handle petabytes of data daily, which is continuously increasing.
6.  **Cost**: As Uber is a low margin business, cost efficiency is crucial in data processing and serving.
7.  **Flexibility**: Infrastructure should accommodate diverse user groups by providing programmatic as well as SQL-like interfaces.

It’s important to note that guaranteeing all these requirements for the same use case is not feasible. Therefore, it's crucial to prioritize based on use case needs.

[[Real-time Data Infrastructure at Uber: INTRODUCTION]]

## 3. ABSTRACTIONS

The diagram (Figure 2) shows logical building blocks of a real-time analytics stack, from bottom to top:

1.  **Storage**: This layer offers generic object or blob storage with read-after-write consistency. This is used for long-term data storage.
2.  **Stream**: This layer provides a publish-subscribe interface for data streams, optimizing for low latency reads and writes.
3.  **Compute**: This layer performs arbitrary computation on underlying stream and storage layers. The compute layer can be a single technology for stream and batch processing, or two separate ones depending on complexity and operational overhead.
4.  **OLAP**: This layer provides limited SQL capabilities over data streams or storage. It optimizes for serving analytical queries in a high throughput, low latency manner.
5.  **SQL**: This layer offers full SQL query functionality over OLAP and Compute layers. Joins can be performed either at this layer or pre-materialized at the compute layer.
6.  **API**: This layer provides a programmatic interface to access the stream or specify a compute function.
7.  **Metadata**: This layer manages all kinds of metadata required by the aforementioned layers.

The choice of technologies in the lower layers will significantly impact the simplicity of the API layer.

[[Real-time Data Infrastructure at Uber: INTRODUCTION]]

![An abstraction of the real-time data infrastructure and the overview of the components](https://chat.openai.com/Figure2.jpg) _Figure 2: An abstraction of the real-time data infrastructure and the overview of the components_

# References

1.  Uber’s real-time data infrastructure paper: [arXiv:2104.00087v1](https://arxiv.org/abs/2104.00087)
2.  Dynamic Pricing at Uber: [Link](https://eng.uber.com/dynamic-pricing/)
3.  CAP Theorem: [Link](https://en.wikipedia.org/wiki/CAP_theorem)
4.  UberEats Restaurant Manager: [Link](https://www.uber.com/us/en/u/restaurant-manager/)

## Overview

The paper discusses Uber's unique implementation of real-time data infrastructure, with a significant focus on the Apache Kafka streaming storage system. It delves into Uber's large-scale deployment of Apache Kafka, handling trillions of messages and petabytes of data daily. Apache Kafka is a robust platform for managing real-time data streaming and storage, propelling various workflows, such as streaming analytics, database change-log streaming, and data ingestion into Uber's Apache Hadoop Data Lake. To cater to Uber's scalability, fault-tolerance needs, and unique requirements, several enhancements were made to the Kafka system.

## 4.1 Apache Kafka for Streaming Storage

Apache Kafka is a widely adopted distributed event streaming system that Uber started using in 2015. Kafka's performance, operational simplicity, mature open-source ecosystem, and wide industry adoption made it a clear choice for Uber.

**[[Uber's Kafka Deployment]]** Uber operates one of the largest deployments of Apache Kafka in the industry, handling trillions of messages and petabytes of data every day. It has been customized to improve fault tolerance and meet Uber's unique large-scale requirements.

### 4.1.1 Cluster Federation

Uber developed a novel federated Kafka cluster setup to improve availability and tolerate single-cluster failures. This setup abstracts the details of the individual clusters from the producers/consumers, allowing them to view a "logical cluster". A metadata server consolidates all the metadata information for the clusters and topics in a central location. This system routes the client's request to the correct physical cluster.

**[[Benefits of Cluster Federation]]**

1.  Improves reliability and scalability
2.  Simplifies topic management and enables seamless creation of new topics
3.  Allows for horizontal scaling by adding more clusters as required
4.  Facilitates redirection of consumer traffic to other physical clusters without restarting the application

### 4.1.2 Dead Letter Queue

Uber implemented a Dead Letter Queue (DLQ) strategy to handle cases where messages fail to be processed by downstream applications. If a consumer of a topic cannot process a message after several attempts, it will publish that message to the dead letter topic. The messages in the dead letter topic can be purged or merged on demand by the users. This keeps unprocessed messages separate and does not impede live traffic.

**[[Benefits of Dead Letter Queue]]**

1.  Ensures no data loss
2.  Prevents processing bottlenecks

### 4.1.3 Consumer Proxy

To handle the complexities and challenges of client management in a large organization like Uber, a proxy layer was built. This proxy layer consumes messages from Kafka and dispatches them to a user-registered gRPC service endpoint for all publish/subscribe use cases. It encapsulates the complexities of the consumer library in the proxy layer. Applications only need to adopt a very thin, machine-generated gRPC client.

**[[Consumer Proxy Features]]**

1.  Sophisticated error handling
2.  Change of delivery mechanism from message polling to push-based message dispatching
3.  Allows significantly more concurrent processing opportunities

### 4.1.4 Cross-cluster Replication

Uber operates multiple Kafka clusters in different data centers. Cross-cluster replication of Kafka messages is crucial for global data viewing and redundancy. Uber developed a robust and performant replicator across the Kafka clusters called uReplicator. It is designed for strong reliability and elasticity. An in-built rebalancing algorithm minimizes the number of the affected topic partitions during rebalancing. It's adaptive to workload, dynamically redistributing load to the standby workers during high-traffic periods.

**[[uReplicator Features]]**

1.  Strong reliability and elasticity
2.  In-built rebalancing algorithm
3.  Adaptive to workload

To ensure no data loss from cross-cluster replication, Uber developed an end-to-end auditing service called Chaperone. Chaperone collects key statistics, such as the number of unique messages in a tumbling time window, from every stage of the replication pipeline. It compares the collected statistics and generates alerts when a mismatch is detected.

## Diagrams and Figures

### Figure 3: Overview of the Real-Time Data Infrastructure at Uber

This diagram outlines Uber's implementation of the real-time data infrastructure.

[[Figure 3]] (Diagram not provided in the user input)

### Figure 4: Overview of the Kafka Consumer Proxy at Uber

This diagram provides a visual overview of how the Kafka Consumer Proxy functions at Uber, showing the change from a message polling system to a push-based message dispatching system.

[[Figure 4]] (Diagram not provided in the user input)

## References

1.  Apache Kafka, Kafka official documentation [45]
2.  Confluent's Kafka benchmark report [15]
3.  Apache Pulsar [12]
4.  RabbitMQ [23]
5.  Apache Samza [13]
6.  Apache Flink
7.  Apache Hadoop
8.  Uber's Kafka interface for Dead Letter Queue strategy [63]
9.  Uber's Kafka Replicator - uReplicator [59]
10.  Chaperone - Uber's end-to-end auditing service [64]

Note: The actual research paper might contain more references related to the topic, these references are based on the excerpt provided.

## Future Work

The paper concludes with a discussion of future work in this area, which includes scaling up to multiple regions across on-premises data centers and cloud, as well as improvements towards a more cost-efficient architecture. These developments will further enhance Uber's real-time data infrastructure.

## Conclusion

Uber's use of Apache Kafka demonstrates how open-source tools can be adapted and customized to meet specific large-scale needs. Their unique implementations and enhancements to Kafka provide valuable insights into developing robust, scalable, and reliable real-time data infrastructures.

## Introduction

This research paper explores how Uber utilizes Apache Flink, an open-source distributed stream processing framework, for processing their real-time data. Flink's high-throughput, low-latency engine, robust state management and checkpointing features, back-pressure handling capabilities, and rich ecosystem of components and toolings made it the ideal choice over other options like Apache Storm, Apache Spark, and Apache Samza.

## Apache Flink for Stream Processing at Uber

### Advantages of Apache Flink

Uber's choice of Apache Flink was influenced by several factors:

-   **Robustness**: Apache Flink offers continuous support for a large number of workloads due to its built-in state management and checkpointing features for failure recovery.
-   **Scalability**: It's capable of handling back-pressure efficiently when faced with a large input Kafka lag.
-   **Active Open Source Community**: It's supported by a large and active open source community with a rich ecosystem of components and toolings.

### Flink Use Cases at Uber

Uber uses Flink extensively both for customer-facing products and powering internal analytics, capturing a wide range of insights such as city-specific market conditions to global financial estimations. The stream processing logic can be expressed using a SQL dialect or a set of low-level APIs.

## Building Streaming Analytical Applications with SQL

One of the important contributions from Uber to the Apache Flink project is the introduction of FlinkSQL, a layer on top of Flink that transforms Apache Calcite SQL queries into efficient Flink jobs. FlinkSQL allows users of all technical levels to run their streaming processing applications in production regardless of scale, typically within a few hours.

Challenges faced by Uber in using FlinkSQL included:

-   **Resource Estimation and Auto Scaling**: It was necessary to establish a correlation between the common job types and the corresponding resource requirements through empirical analysis.
-   **Job Monitoring and Automatic Failure Recovery**: The platform needs to monitor the job and provide a strong reliability guarantee.

## Unified Architecture for Deployment, Management, and Operation

Uber identified commonalities between the two platforms they used for building and managing the stream processing pipelines, and converged them into a unified architecture for deployment, management, and operation.

The unified architecture consists of three layers:

1.  **Platform Layer**: Handles organizing the business logic and integration with other external platforms such as Machine learning feature training, workflow management and SQL compilation.
2.  **Job Management Layer**: Manages the Flink job’s lifecycle including validation, deployment, monitoring, and failure recovery.
3.  **Infrastructure Layer**: Consists of the compute clusters and storage backend. Provides the abstraction of the physical resources for flexibility and extensibility, regardless of the hosting infrastructure being on-prem or cloud.

### Figure 5: The Layers of the Unified Flink Architecture at Uber

[[Figure 5]] (Diagram not provided in the user input)

## Future Work

Future work includes unifying streaming/batch processing semantics in FlinkSQL, seamless data backfills without writing additional code, and the ability to restart a Flink job without any downtime.

## 4.3 Apache Pinot for OLAP

### Summary

Apache Pinot is an open-source, distributed OLAP system, designed to perform low latency analytical queries on terabytes-scale data. It is built with lambda architecture, providing a unified view of real-time and historical (offline) data.

The OLAP system is designed to efficiently process large amounts of data by distributing query execution across the database segments. It achieves this by using indexing techniques (inverted, range, sorted, and startree index) and a scatter-gather-merge approach to queries. This approach breaks the data into time-bound chunks (segments), executes sub-plans on these distributed segments in parallel, and then aggregates and merges them into a final result.

At Uber, Pinot was chosen over alternatives (Elasticsearch and Apache Druid) due to its significantly smaller memory and disk footprint and lower query latency. Uber uses Pinot to power real-time analytics use cases for products and backend services. This is especially important for use cases where data freshness and query latency need to be real-time in nature.

Uber has contributed several enhancements to Apache Pinot to suit their unique requirements around high availability, rich query support, and exactly-once semantics.

### 4.3.1 Upsert support

Upsert is a method that allows records to be updated during real-time ingestion into the OLAP store, essential for Uber use-cases such as correcting a ride fare or updating a delivery status. Upsert support was developed for Apache Pinot by organizing the input stream into multiple partitions by the primary key and assigning all records with the same primary key to the same node. This shared-nothing solution offers advantages such as better scalability, elimination of single point of failure, and ease of operation.

### 4.3.2 Full SQL support

To support subqueries and joins which were lacking in Pinot, it was integrated with Presto, which is the standard query engine for interactive queries at Uber. This combination allows for standard PrestoSQL queries on Pinot tables and enables sub-second query latencies for such queries, something that's not possible to do on standard backends like HDFS/Hive.

### 4.3.3 Integration with the rest of Data ecosystem

Uber has made efforts to integrate Pinot with the rest of its data ecosystem to ensure a seamless user experience. This includes automatic schema inference from the input Kafka topic, FlinkSQL integration as a data sink, and creating Pinot offline tables from Hive datasets via Spark.

### 4.3.4 Peer-to-peer segment recovery

To overcome the issues of synchronous backup to external archival or segment store, and the associated scalability bottleneck and data freshness violation, Uber's team designed and implemented an asynchronous solution. With this, server replicas can serve the archived segments in case of failures, replacing a centralized segment store with a peer-to-peer scheme. This solution maintains data and query consistency, removes the single node backup bottleneck, and improves overall data freshness.

### Future Work

Going forward, Uber is working on:

-   Low latency joins: Adding the ability to perform lookup joins to Pinot to support joining tables with commonly used dimension tables.
-   Semistructured (e.g., JSON) data support: Building native JSON support for both ingestion and queries.

## 4.4 HDFS for archival store

### Summary

At Uber, Hadoop Distributed File System (HDFS) is used as long-term storage for all data. Most of this data comes from Kafka in Avro format and is stored in HDFS as raw logs. These logs are then merged into a long-term Parquet data format using a compaction process and made available via standard processing engines such as Hive, Presto, or Spark. Such datasets are the source of truth for all

## 5 Use Cases Analysis
The paper presents several real-time use cases across 4 broad categories in production at Uber and discusses the design tradeoffs considered.

### 5.1 Analytical Application: Surge Pricing
**Surge pricing** is a dynamic pricing mechanism to balance the supply of drivers with demand. It's essentially a **streaming pipeline** for calculating pricing multipliers based on trip data, rider and driver status within a specific time window. The pipeline prioritizes data freshness and availability over consistency. For these purposes, Uber uses the **Kafka** cluster and an active-active setup (discussed in section 6).

### 5.2 Dashboards: UberEats Restaurant Manager
The **UberEats Restaurant Manager** dashboard gives owners insights about customer satisfaction, popular menu items, sales, and service quality analysis. This dashboard prioritizes fresher data and low query latency. Uber uses **Pinot** for efficient pre-aggregation indices and preprocessors in **Flink** for further processing time reduction. This case shows a tradeoff between preprocessing at transformation time (Flink) and query time (Pinot).

### 5.3 Machine Learning: Real-time Prediction Monitoring
Uber utilizes Machine Learning (ML) to monitor the accuracy of its predictions. Given the high volume and cardinality of data to be processed, scalability is critical. To address this, Uber set up a real-time prediction monitoring pipeline that uses **Flink** to aggregate metrics and detect abnormalities. Pre-aggregation is also performed in **Pinot** tables to boost query performance.

### 5.4 Ad-hoc Exploration: UberEats Ops Automation
This use case describes how the UberEats team uses real-time data for ad-hoc analysis and to automate decision-making. Uber uses **Pinot** for aggregating statistics, **Presto** for executing ad hoc queries, and **Flink** for further processing and reliability. This framework allows seamless transition from ad-hoc exploration to production rollout.

### Table 1: The components used by the example use cases
|                | Surge | Restaurant Manager | Real-time Prediction | Monitoring | Eats Ops | Automation |
|----------------|-------|-------------------|----------------------|------------|----------|------------|
| API            | Y     | Y                 |                      | Y          | Y        |            |
| SQL            | Y     | Y                 | Y                    | Y          |          |            |
| OLAP           | Y     | Y                 | Y                    | Y          |          |            |
| Compute        | Y     | Y                 | Y                    | Y          | Y        | Y          |
| Stream         | Y     | Y                 | Y                    | Y          | Y        | Y          |
| Storage        | Y     |                   |                      |            | Y        |            |

## 6 All-Active Strategy
Uber employs an **all-active strategy** for ensuring business resilience and continuity, with services deployed in geographically distributed data centers. The core of this multi-region real-time architecture is a **multi-region Kafka setup**.

### Active-Active Setup for Surge Pricing
![Active-Active Setup for Surge Pricing](figure6.jpg)

This setup ensures that the surge pricing service can continue operating even if one region experiences a disaster. Each region has an instance of the 'update service', with one region designated as primary. When disaster strikes, the primary region can be switched, allowing surge pricing calculations to fail over to another region. This approach is compute intensive as it runs redundant pipelines in each

 region.

### Active-Passive Setup for Stronger Consistency
![Active-Passive Setup for Stronger Consistency](figure7.jpg)

Services that favor strong consistency, such as payment processing and auditing, may opt for an active/passive mode. In the event of a disaster, the service can fail over to another region and resume its consumption progress. This approach requires a sophisticated offset management service to manage the offset mappings across regions, which is discussed in more detail in the paper. 

---
Note: This note is a summary of the research paper "Real-time Data Infrastructure at Uber". The original paper can be found [here](link_to_paper).


# Backfill
Backfill refers to the process of reprocessing historical stream data. At Uber, backfill is essential due to a variety of reasons:

- When a new data processing pipeline is created, it needs to be tested against existing data.
- A new machine learning model might require training with several months of data.
- A bug might be discovered in a real-time application that has already processed the data, requiring the data to be reprocessed.
- Changes in stream processing logic might necessitate the reprocessing of old data.

Given the ubiquity of backfill in real-time big data processing, architectures such as Lambda and Kappa have been proposed to handle this issue. However, both suffer from limitations. The Lambda architecture maintains separate systems for batch and stream processing, leading to maintenance and consistency issues. On the other hand, the Kappa architecture requires long-term data retention in Kafka and might not be as efficient in processing throughput.

Uber built its own backfill solution for stream processing use cases using Flink, which supports two modes of operation:

- **SQL-based:** This mode allows the same SQL query to be executed on both real-time (Kafka) and offline datasets (Hive). Even though it bears similarities to the Lambda architecture, the user doesn't need to maintain two distinct jobs.
- **API-based:** This approach, internally named Kappa+, is capable of reusing the stream processing logic like the Kappa architecture but can read archived data directly from offline datasets such as Hive. It addresses several issues related to batch datasets processing, including identification of the start/end boundary of the bounded input, handling of higher throughput from historical data, and fine-tuning job memory for buffering purposes.

Backfilling is an active area of investigation, with several edge cases still needing to be addressed in both solutions.

# Related Work

Real-time data infrastructure encompasses a wide range of components, and numerous related systems exist for each area.

## Messaging Systems
Traditional enterprise messaging systems like ActiveMQ, RabbitMQ, Oracle Enterprise Messaging Service, and IBM Websphere MQ have played a crucial role as an event bus for processing asynchronous data flows. However, these systems are incomparable to Kafka in terms of features, ecosystem, and system performance. A new messaging system, Apache Pulsar, has emerged with a novel tiered architecture that decouples data serving and data storage, providing better elasticity and easier operation.

## Stream Processing Systems
Highly scalable stream processing systems, like Storm, Samza, Heron, Spark Streaming, Apex, and others developed by large internet companies, have gained popularity due to the need for real-time data processing. Systems like Apache Flink are expanding to support batch processing use cases, while others, like Dataflow and Apache Beam, abstract away the underlying processing engines.

## Real-time OLAP Systems
Real-time OLAP systems like Apache Druid and Clickhouse, similar to Pinot, buffer ingested streams and utilize column stores for efficient column scans. These systems are becoming popular as businesses increasingly need to quickly transform newly obtained data into insights. Another approach to improve OLAP system performance is to pre-aggregate data into cubes. HTAP databases unify transactional and analytical processing, but they face the challenge of cleanly separating the two types of operations.

## SQL Systems
SQL systems that operate on large datasets have gained popularity over the past decade. Systems like Apache Hive, Dremel, Spark SQL, MySQL, Impala, and Drill each present a unique set of tradeoffs. Some systems, like Procella and Trill, have extended their support to query real-time data. Uber chose the open source Presto for its interactivity, flexibility, and extensibility, and enhanced it with real-time data availability via Pinot.

Other large-scale internet companies also make extensive use of

 real-time data. Facebook and Google have developed similar real-time data processing and analytics infrastructures. HSAP (Hybrid Serving and Analytical Processing) has emerged as a new architecture, exemplified by Alibaba's Hologres, fusing analytical processing and serving as well as online and offline analysis. At Uber, however, the preference has been for loosely coupled independent systems for the ease of customization and evolution of each component. 
# Lessons Learned

The journey of building and scaling Uber's real-time data infrastructure has provided many valuable lessons:

## Open Source Adoption

The majority of the real-time analytics stack at Uber is built on open source technologies, largely due to the ability of these technologies to iterate quickly and reduce time to market. However, open source technologies often require significant customization to fit the specific needs of an organization. Uber found this to be true when they had to modify Apache Kafka, originally designed for log propagation with Java applications, to fit their needs, including developing a RESTful ecosystem around Kafka for compatibility with various languages and customizing core routing and replication mechanisms. Other examples include integrating Uber’s container ecosystem, enhancing security policies, building a full SQL layer on Apache Pinot for a non-engineering audience, and creating a seamless backfilling process using Apache Flink.

## Rapid System Development and Evolution

Uber recognized the importance of enabling rapid software development to allow each system to evolve quickly and independently. A key strategy was to standardize interfaces to establish clear boundaries between services and minimize risks of breaking clients. They also opted for thin clients to reduce the frequency of client upgrades, and consolidated programming languages to simplify system interactions. To support quick system evolution, Uber integrated their infrastructure components with a proprietary Continuous Integration/Continuous Deployment (CI/CD) framework.

## Ease of Operation and Monitoring

To manage the challenge of rapidly scaling the infrastructure, Uber strategically invested in automation, building declarative frameworks to orchestrate system deployments. They also implemented real-time monitoring and alerting for system reliability and minimizing negative business impacts. Automated dashboards, alerts, and chargeback mechanisms were established for each use case related to Kafka, Flink, and Pinot.

## Ease of User Onboarding and Debugging

Uber created a self-serve system to automate user onboarding, failure handling, and triaging. The system includes a centralized metadata repository for data discovery, a data auditing system to track data loss and duplication across all of Uber's data centers, and an automated onboarding process that provisions resources as required.

# Conclusion

The real-time data infrastructure at Uber has been critical in supporting various mission-critical use cases. The system, processing multiple petabytes of data per day, has been optimized for flexibility and scale. The adoption of open source technologies has both reduced costs and time to market.

Uber’s engineering teams have contributed unique enhancements to these technologies, helping overcome three fundamental scaling challenges:

1. **Scaling Data:** Techniques such as Kafka thin client libraries, cluster federation, etc., allowed every service in Uber to adopt Kafka. This robust foundation enabled orchestrating Flink and Pinot data pipelines leveraged for mission-critical use cases across all business units. Flink job automation promoted widespread adoption with low operational overhead.
2. **Scaling Use Cases:** Investments were made in the flexibility of individual technologies to power various use cases. For instance, Kafka was used for a wide spectrum of use cases from logging to disseminating financial data. Pinot provided a low latency OLAP layer for mission-critical use cases as well as enabled real-time data exploration via Presto integration.
3. **Scaling Users:** Layers of indirection between users and the underlying technologies were added using abstractions and standard interfaces, greatly reducing the user support cost. FlinkSQL and PrestoSQL enabled users with basic SQL knowledge to spin up complex Flink pipelines or query data across multiple data systems in a seamless manner.

# Real-time Data Infrastructure at Uber
## 11 FUTURE WORK
### Streaming and Batch Processing Unification
* Currently, users have to express the same processing logic twice in different languages and run them on different compute frameworks for use cases that demand both batch and stream processing.
* To simplify this process, Uber is looking into a unified processing solution which would ease development and pipeline management.

### Multi-region and Multi-zone Deployments
* Uber is pushing towards a multi-region-multi-zone strategy to improve the scalability and reliability of their real-time data infrastructure.
* The challenge here is to optimize data placement to balance data redundancy (for reliability) and storage cost due to excessive copies.

### On-prem and Cloud Agnostic
* Uber is planning to convert its systems to be agnostic of data centers or cloud, facilitating a seamless transition from on-prem to cloud.

### Tiered Storage
* Uber is investigating tiered storage solutions for Kafka and Pinot to improve cost efficiency (by storing colder data in a cheaper storage medium) and elasticity (by separating data storage and serving layers).

## 12 ACKNOWLEDGEMENT
* Uber acknowledges the contributions of several engineers, PMs, and management leaders in the evolution of their real-time data infrastructure.

## REFERENCES
* The authors have referenced various online resources, technical papers, and conference presentations related to real-time data processing, distributed systems, and other relevant technologies.
* Some of the notable references include resources on Apache Kafka, Amazon Kinesis, Apache Samza, Apache Storm, Google Cloud Storage, MySQL, RabbitMQ, Apache Pulsar, and several papers and articles published by Uber Engineering.
* The referenced works also include research papers on Google’s globally distributed database - Spanner, Twitter's real-time processing engine - Heron, and other prominent technologies in the field of real-time data processing.
* Apart from these, they have also referred to a number of research papers and technical blogs that discuss various aspects of real-time data processing, such as the Lambda and Kappa architectures, dynamic pricing on the Uber platform, and open source OLAP systems for big data. 

---

(Following is the abstract summary of some important references)

### References

* [[1]] Amazon Kinesis: Amazon's streaming data service that makes it easy to collect, process, and analyze real-time, streaming data.
* [[3 Relaxed Ordering]] Apache Beam: An open-source, unified model for defining both batch and streaming data-parallel processing pipelines.
* [[8]] Apache HDFS: A distributed file system that provides high-throughput access to application data.
* [[15]] Benchmarking Apache Kafka, Apache Pulsar, and RabbitMQ: A blog post on the performance of these three messaging systems.
* [[16]] ClickHouse: An open-source column-oriented database management system that allows generating analytical data reports in real time.
* [[32]] Spanner: Google’s globally-distributed database that provides strong consistency, high availability, and durability guarantees.
* [[33]] MapReduce: A programming model and an associated implementation for processing and generating big data sets with a parallel, distributed algorithm on a cluster.
* [[40]] Michelangelo: Uber’s Machine Learning Platform.
* [[45]] Kafka: A distributed messaging system for log processing developed by LinkedIn.
* [[46]] Twitter Heron: Twitter's real-time stream processing engine.
* [[51]] Peloton: Uber’s unified resource scheduler for diverse cluster workloads.
* [[52]] Moving from Lambda and Kappa Architectures to Kappa+ at Uber: A talk on Uber's evolution of data architecture.
* [[57]] Presto: A distributed SQL query engine designed to query large data sets distributed over one or more heterogeneous data sources.