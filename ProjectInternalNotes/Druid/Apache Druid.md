https://juejin.cn/post/6844904083506069517?searchId=20230817073455A41F9D0BBBEE32CAD7C2
---
#### I. Introduction to Apache Druid
- **Nature**: Apache Druid is an analytical data platform combining features of time series databases, data warehouses, and full-text retrieval systems.
- **Purpose**: Designed for fast data ingestion and quick query response, especially suitable for modern cloud-native, stream-native, analytical database requirements.
#### II. Key Features of Druid
- **Fast Query and Data Ingestion**: Optimized for speedy data handling and query response.
- **UI and Runtime Queries**: Offers a powerful user interface with operable queries at runtime.
- **High-Performance Processing**: Capable of handling concurrent data processing efficiently.
#### III. Integration Capabilities
- **Data Streaming**: Integrates seamlessly with message buses like Kafka and Amazon Kinesis.
- **Batch Loading**: Supports batch data loading from data lakes (HDFS, Amazon S3, etc.).
#### IV. Performance Benchmark
- **Speed Comparison**: Benchmark tests indicate Druid's data ingestion and querying are significantly faster than traditional solutions.
#### V. Architectural Highlights
- **Hybrid Design**: Combines the best attributes of data warehouses, time series databases, and retrieval systems.
- **Deployment Flexibility**: Compatible with various environments including AWS/GCP/Azure, hybrid cloud, Kubernetes, and on-premise servers.
#### VI. Use Cases and Applications
- **Real-time Data Extraction**: Suitable for scenarios demanding high-performance and real-time data queries.
- **Common Applications**: 
  - Clickstream analysis (web/mobile).
  - Risk control analysis.
  - Network telemetry (network performance monitoring).
  - Server metrics storage.
  - Supply chain analysis.
  - Application performance metrics.
  - Business intelligence and real-time OLAP.
#### VII. Technical Features
- **Column Storage**: Individual column storage and compression for efficient data handling.
- **Native Search Index**: Inverted indexes for string values to enhance search and filtering.
- **Flexible Data Ingestion**: Supports both streaming and batch data ingestion.
- **Adaptable Data Schema**: Easily handles changing data schemas and nested data types.
- **Time-based Partitioning**: Optimized partitioning based on time for faster queries.
- **SQL Support**: Allows querying using SQL in addition to native JSON-based queries.
- **Scalability**: Capable of handling high data ingestion rates and massive data storage.
- **Operational Simplicity**: Easy to scale, rebalance, and recover.
#### VIII. Druid's Architecture
- **Microservice-based**: Comprises multiple microservices for ingestion, querying, and coordination.
- **Scalable and Failover Capable**: Supports independent scaling and failure of individual services.
- **Data Segmentation**: Utilizes segments for query-optimized data storage.
- **Indexing**: Employs various indexing techniques for different column types.
#### IX. Operational Aspects
- **Data Redundancy**: Creates multiple data copies for robustness.
- **Service Independence**: Each core service can be independently adjusted and scaled.
- **Backup and Recovery**: Automated data backup to filesystems for rapid recovery.
- **Update Management**: Supports rolling updates for seamless version transitions.
#### X. Druid in Business Intelligence
- **BI Usage**: Frequently used in business intelligence for its fast query capabilities and high concurrency.
- **Enhanced Applications**: Improves applications by enabling real-time data analysis and visualization.
---
