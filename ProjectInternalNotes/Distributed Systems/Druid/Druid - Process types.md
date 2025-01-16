Druid has several process types:

- [Coordinator](file:///Users/venkatamunnangi/Programming/oss/streaming/java_proj/druid/docs/design/coordinator.md)
- [Overlord](file:///Users/venkatamunnangi/Programming/oss/streaming/java_proj/druid/docs/design/overlord.md)
- [Broker](file:///Users/venkatamunnangi/Programming/oss/streaming/java_proj/druid/docs/design/broker.md)
- [Historical](file:///Users/venkatamunnangi/Programming/oss/streaming/java_proj/druid/docs/design/historical.md)
- [MiddleManager](file:///Users/venkatamunnangi/Programming/oss/streaming/java_proj/druid/docs/design/middlemanager.md) and [Peons](file:///Users/venkatamunnangi/Programming/oss/streaming/java_proj/druid/docs/design/peons.md)
- [Indexer (Optional)](file:///Users/venkatamunnangi/Programming/oss/streaming/java_proj/druid/docs/design/indexer.md)
- [Router (Optional)](file:///Users/venkatamunnangi/Programming/oss/streaming/java_proj/druid/docs/design/router.md)

## Server types

Druid processes can be deployed any way you like, but for ease of deployment we suggest organizing them into three server types:

- **Master**
- **Query**
- **Data**

![Druid architecture](http://localhost:63343/markdownPreview/1001161419/fileSchemeResource/befa7eadb386becc3bf98e77f6d90d1f-druid-architecture.png?_ijt=admjumghfsqk5sjd2ja3vfknsb)

This section describes the Druid processes and the suggested Master/Query/Data server organization, as shown in the architecture diagram above.

### Master server

A Master server manages data ingestion and availability: it is responsible for starting new ingestion jobs and coordinating availability of data on the "Data servers" described below.

Within a Master server, functionality is split between two processes, the Coordinator and Overlord.

#### Coordinator process

[**Coordinator**](file:///Users/venkatamunnangi/Programming/oss/streaming/java_proj/druid/docs/design/coordinator.md) processes watch over the Historical processes on the Data servers. They are responsible for assigning segments to specific servers, and for ensuring segments are well-balanced across Historicals.

#### Overlord process

[**Overlord**](file:///Users/venkatamunnangi/Programming/oss/streaming/java_proj/druid/docs/design/overlord.md) processes watch over the MiddleManager processes on the Data servers and are the controllers of data ingestion into Druid. They are responsible for assigning ingestion tasks to MiddleManagers and for coordinating segment publishing.

### Query server

A Query server provides the endpoints that users and client applications interact with, routing queries to Data servers or other Query servers (and optionally proxied Master server requests as well).

Within a Query server, functionality is split between two processes, the Broker and Router.

#### Broker process

[**Broker**](file:///Users/venkatamunnangi/Programming/oss/streaming/java_proj/druid/docs/design/broker.md) processes receive queries from external clients and forward those queries to Data servers. When Brokers receive results from those subqueries, they merge those results and return them to the caller. End users typically query Brokers rather than querying Historicals or MiddleManagers processes on Data servers directly.

#### Router process (optional)

[**Router**](file:///Users/venkatamunnangi/Programming/oss/streaming/java_proj/druid/docs/design/router.md) processes are _optional_ processes that provide a unified API gateway in front of Druid Brokers, Overlords, and Coordinators. They are optional since you can also simply contact the Druid Brokers, Overlords, and Coordinators directly.

The Router also runs the [web console](file:///Users/venkatamunnangi/Programming/oss/streaming/java_proj/druid/docs/operations/web-console.md), a management UI for datasources, segments, tasks, data processes (Historicals and MiddleManagers), and coordinator dynamic configuration. The user can also run SQL and native Druid queries within the console.

### Data server

A Data server executes ingestion jobs and stores queryable data.

Within a Data server, functionality is split between two processes, the Historical and MiddleManager.

### Historical process

[**Historical**](file:///Users/venkatamunnangi/Programming/oss/streaming/java_proj/druid/docs/design/historical.md) processes are the workhorses that handle storage and querying on "historical" data (including any streaming data that has been in the system long enough to be committed). Historical processes download segments from deep storage and respond to queries about these segments. They don't accept writes.

### Middle Manager process

[**MiddleManager**](file:///Users/venkatamunnangi/Programming/oss/streaming/java_proj/druid/docs/design/middlemanager.md) processes handle ingestion of new data into the cluster. They are responsible for reading from external data sources and publishing new Druid segments.

#### Peon processes

[**Peon**](file:///Users/venkatamunnangi/Programming/oss/streaming/java_proj/druid/docs/design/peons.md) processes are task execution engines spawned by MiddleManagers. Each Peon runs a separate JVM and is responsible for executing a single task. Peons always run on the same host as the MiddleManager that spawned them.

### Indexer process (optional)

[**Indexer**](file:///Users/venkatamunnangi/Programming/oss/streaming/java_proj/druid/docs/design/indexer.md) processes are an alternative to MiddleManagers and Peons. Instead of forking separate JVM processes per-task, the Indexer runs tasks as individual threads within a single JVM process.

The Indexer is designed to be easier to configure and deploy compared to the MiddleManager + Peon system and to better enable resource sharing across tasks. The Indexer is a newer feature and is currently designated [experimental](file:///Users/venkatamunnangi/Programming/oss/streaming/java_proj/druid/docs/development/experimental.md) due to the fact that its memory management system is still under development. It will continue to mature in future versions of Druid.

Typically, you would deploy either MiddleManagers or Indexers, but not both.

## Pros and cons of colocation

Druid processes can be colocated based on the Master/Data/Query server organization as described above. This organization generally results in better utilization of hardware resources for most clusters.

For very large scale clusters, however, it can be desirable to split the Druid processes such that they run on individual servers to avoid resource contention.

This section describes guidelines and configuration parameters related to process colocation.

### Coordinators and Overlords

The workload on the Coordinator process tends to increase with the number of segments in the cluster. The Overlord's workload also increases based on the number of segments in the cluster, but to a lesser degree than the Coordinator.

In clusters with very high segment counts, it can make sense to separate the Coordinator and Overlord processes to provide more resources for the Coordinator's segment balancing workload.

#### Unified Process

The Coordinator and Overlord processes can be run as a single combined process by setting the `druid.coordinator.asOverlord.enabled` property.

Please see [Coordinator Configuration: Operation](file:///Users/venkatamunnangi/Programming/oss/streaming/java_proj/druid/docs/configuration/index.md#coordinator-operation) for details.

### Historicals and MiddleManagers

With higher levels of ingestion or query load, it can make sense to deploy the Historical and MiddleManager processes on separate hosts to to avoid CPU and memory contention.

The Historical also benefits from having free memory for memory mapped segments, which can be another reason to deploy the Historical and MiddleManager processes separately.