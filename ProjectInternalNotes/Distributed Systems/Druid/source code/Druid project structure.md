1. druid-processing - related to Druid segments
2. druid-server - ZK curator discoveries.
3. druid-services - cli script to trigger / start various services. 
4. druid-sql
5. indexing-service
- [Coordinator](https://druid.apache.org/docs/latest/design/coordinator) manages data availability on the cluster.
- [Overlord](https://druid.apache.org/docs/latest/design/overlord) controls the assignment of data ingestion workloads.
- [Broker](https://druid.apache.org/docs/latest/design/broker) handles queries from external clients.
- [Router](https://druid.apache.org/docs/latest/design/router) routes requests to Brokers, Coordinators, and Overlords.
- [Historical](https://druid.apache.org/docs/latest/design/historical) stores queryable data.
- [MiddleManager](https://druid.apache.org/docs/latest/design/middlemanager) and [Peon](https://druid.apache.org/docs/latest/design/peons) ingest data.
- [Indexer](https://druid.apache.org/docs/latest/design/indexer) serves an alternative to the MiddleManager + Peon task execution system

| Service        | Port address |
| -------------- | ------------ |
| Coordinator    | 8081         |
| Overlord       | 8081         |
| Router         | 8888         |
| Broker         | 8082         |
| Historical     | 8083         |
| Middle Manager | 8091         |
https://druid.apache.org/docs/latest/api-reference/service-status-api