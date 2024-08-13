https://www.figma.com/blog/how-figmas-databases-team-lived-to-tell-the-scale/
## Introduction
**Objective:** To scale Figma's database to handle the growing user base and data volume by transitioning from vertical partitioning to horizontal sharding.
## Table of Contents
1. **Background and Challenges**
2. **Scaling Strategies**
3. **Vertical Partitioning**
4. **Scaffolding for Scale**
5. **Exploring Options**
6. **Horizontal Sharding**
7. **Implementation Details**
8. **Query Routing with DBProxy**
9. **Physical Sharding Operations**
10. **Future Directions**

## Background and Challenges
Figma's database stack experienced a 100x growth since 2020, necessitating a transition from a single Postgres instance to a distributed architecture.

### Key Challenges
1. **High Write Loads:** Tables with billions of rows caused reliability issues.
2. **IOPS Limits:** Amazon RDS's IO operations per second (IOPS) constraints.
3. **Scaling Beyond Vertical Partitioning:** Single tables became too large to manage.

## Scaling Strategies
### Vertical Partitioning
- **Definition:** Splitting related tables into separate databases.
- **Impact:** Quick scaling gains by distributing load across multiple databases.
#### ASCII Visualization
```
+------------------+
|    Main DB       |
|                  |
|  +------------+  |
|  | Figma Files|  |
|  +------------+  |
|                  |
+------------------+
```
```
+------------------+   +------------------+
| Figma Files DB   |   | Organization DB  |
|                  |   |                  |
| +------------+   |   | +------------+   |
| | Figma Files|   |   | | Organizations| |
| +------------+   |   | +------------+   |
|                  |   |                  |
+------------------+   +------------------+
```
## Scaffolding for Scale
**Goals:**
1. **Minimize Developer Impact:** Allow developers to focus on features rather than refactoring code.
2. **Transparent Scaling:** Future scaling should not require application layer changes.
3. **Avoid Backfills:** Skip large table backfills to save time.
4. **Incremental Progress:** Roll out changes incrementally to reduce risks.
5. **Avoid One-way Migrations:** Ensure the ability to roll back changes.
6. **Strong Data Consistency:** Maintain consistency without complex solutions like double-writes.
## Exploring Options
- **Alternative Databases:** CockroachDB, TiDB, Spanner, Vitess.
  - **Issue:** Migration complexity and loss of in-house expertise with RDS Postgres.
- **NoSQL Databases:** Not suitable due to Figma's complex relational data model.
- **In-house Horizontal Sharding:** Tailoring horizontal sharding to Figma's architecture was the chosen path.
## Horizontal Sharding
### Path to Horizontal Sharding
- **Process:** Breaking up a single table across multiple database instances.
- **Goals:** Achieve nearly infinite scalability, maintain data consistency, and ensure minimal downtime.
#### ASCII Visualization
**Vertical Partitioning**
```
+------------------+
|      DB          |
|                  |
| +------------+   |
| |  Files     |   |
| +------------+   |
|                  |
+------------------+
```

**Horizontal Sharding**
```
+------------------+   +------------------+
| Shard 1          |   | Shard 2          |
|                  |   |                  |
| +------------+   |   | +------------+   |
| |  Files     |   |   | |  Files     |   |
| +------------+   |   | +------------+   |
|                  |   |                  |
+------------------+   +------------------+
```

## Implementation Details
### Colos and Shard Keys
- **Colos:** Grouping related tables with the same sharding key.
- **Shard Keys:** UserID, FileID, OrgID for distributing data evenly.
### Logical Sharding vs. Physical Sharding
- **Logical Sharding:** Preparing tables for sharding at the application layer.
- **Physical Sharding:** Actual data split across multiple database instances.
**Logical Sharding**
```
+--------------------+
|   Logical Shard 1  |
|   +------------+   |
|   |   Data     |   |
|   +------------+   |
+--------------------+
```
**Physical Sharding**
```
+--------------------+   +--------------------+
|   Physical Shard 1 |   |   Physical Shard 2 |
|   +------------+   |   |   +------------+   |
|   |   Data     |   |   |   |   Data     |   |
|   +------------+   |   |   +------------+   |
+--------------------+   +--------------------+
```
### DBProxy
- **Purpose:** Intercepts SQL queries and routes them to the correct shards.
- **Components:**
  - **Query Parser:** Converts SQL to an Abstract Syntax Tree (AST).
  - **Logical Planner:** Extracts query type and logical shard IDs.
  - **Physical Planner:** Maps queries to physical shards.
**DBProxy Query Engine**
```
+-----------------+
|  Query Parser   |
+-----------------+
        |
        v
+-----------------+
| Logical Planner |
+-----------------+
        |
        v
+-----------------+
| Physical Planner|
+-----------------+
```

### Scatter-Gather Queries
- **Scatter:** Send query to all shards.
- **Gather:** Aggregate results from all shards.

**Scatter-Gather**
```
+---------+   +---------+   +---------+
| Shard 1 |   | Shard 2 |   | Shard 3 |
+---------+   +---------+   +---------+
      \         |         /
       \        |        /
        \       |       /
         +--------------+
         |  Aggregation |
         +--------------+
```
## Physical Sharding Operations
### Process
1. **Preparation:** Ensure logical sharding is working correctly.
2. **Execution:** Copy data from single to multiple databases.
3. **Failover:** Switch read and write traffic to new shards.
**Unsharded to Sharded**
```
Unsharded
+-----------------+
|    Database     |
+-----------------+

Sharded
+---------+   +---------+   +---------+
| Shard 1 |   | Shard 2 |   | Shard 3 |
+---------+   +---------+   +---------+
```
## Future Directions
### Goals
1. **Support Sharded Schema Updates:** Manage schema changes across shards.
2. **Globally Unique ID Generation:** Ensure unique IDs for sharded primary keys.
3. **Atomic Cross-shard Transactions:** Implement for business-critical use cases.
4. **Distributed Unique Indexes:** Support indexes including sharding keys.
5. **ORM Compatibility:** Enhance developer productivity.
6. **Automated Reshard Operations:** Enable seamless shard splits.
**Globally Unique ID Generation**
```
+-----------------+
| ID Generation   |
+-----------------+
        |
        v
+---------+   +---------+   +---------+
| Shard 1 |   | Shard 2 |   | Shard 3 |
+---------+   +---------+   +---------+
```

**Atomic Cross-shard Transactions**
```
+-------------------+
|   Transaction     |
+-------------------+
        |
        v
+---------+   +---------+   +---------+
| Shard 1 |---| Shard 2 |---| Shard 3 |
+---------+   +---------+   +---------+
```

**ORM Compatibility**
```
+-----------+
|   ORM     |
+-----------+
        |
        v
+---------+   +---------+   +---------+
| Shard 1 |   | Shard 2 |   | Shard 3 |
+---------+   +---------+   +---------+
```
## Conclusion
The journey to horizontally shard Figma's Postgres stack was a significant technical challenge. By incrementally scaling and de-risking the process, Figma achieved nearly infinite scalability while maintaining data consistency and minimizing developer impact. The future roadmap includes further enhancements to ensure continued growth and efficiency.