https://jack-vanlightly.com/analyses/2024/7/30/understanding-apache-icebergs-consistency-model-part1
## Introduction
Apache Iceberg is a widely adopted table format designed for large-scale, high-performance analytics on data lakes. It is known for its support of high-throughput, parallel data processing, and robust consistency model. This detailed note dives into the internals of Apache Iceberg, focusing on its consistency model, write path, and metadata management.
## Basic Mechanics of Apache Iceberg
### Table Representation
An Iceberg table consists of a metadata layer and a data layer, both stored in an object store such as S3. A key feature is that commits are performed against a catalog component.
- **Metadata Layer**: Contains information about the table structure, schema, and versions (snapshots).
- **Data Layer**: Stores the actual data files (Parquet, ORC, Avro).
Each write operation creates a new snapshot representing the table's state at that time. The snapshot includes:
- **Manifest List File**: Contains entries for each manifest file, indicating its file path and metadata.
- **Manifest Files**: Each manifest file references one or more data files.
- **Data Files**: Actual storage files in formats like Parquet, ORC, or Avro.
### Snapshot Management
Snapshots are stored in a log within the table's metadata file, referenced by the catalog. Each write creates a new metadata file, including the new snapshot.
```plaintext
Snapshot Log:
metadata-1 -> snapshot-1
metadata-2 -> snapshot-2
metadata-3 -> snapshot-3
```
### Atomic Compare-and-Swap (CAS) Commit
A commit operation in Iceberg is an atomic compare-and-swap (CAS) where the writer provides the current and new metadata file locations. This ensures consistency even with concurrent writes.
```plaintext
CAS Operation:
current_metadata_location -> new_metadata_location
```
## Example of Writing Snapshots
### Simple Insert Operations
Consider a table with a single column "fruit". Three insert operations create three snapshots:
```plaintext
Insert Operations:
snapshot-1: data-1
snapshot-2: data-2
snapshot-3: data-3
```
### Tracking File Additions and Deletions
Iceberg uses manifest files to track added and deleted files. Each manifest entry can have a status of ADDED, EXISTING, or DELETED. These statuses help compute engines understand the state of the data files.
### Example Across Multiple Snapshots
```plaintext
Snapshot-1:
manifest-1: data-1 (ADDED), data-2 (ADDED)

Snapshot-2:
manifest-2: data-1 (DELETED), data-2 (EXISTING)

Snapshot-3:
manifest-3: data-3 (ADDED)
manifest-2: data-1 (DELETED), data-2 (EXISTING)

Snapshot-4:
manifest-4: data-2 (DELETED)

Snapshot-5:
manifest-5: data-4 (ADDED)
```
## Copy-on-Write (COW) vs. Merge-on-Read (MOR)
### Copy-on-Write
In COW mode, any row-level modifications cause the entire data file to be rewritten.
```plaintext
COW Example:
snapshot-1: data-1 (original)
snapshot-2: data-2 (rewritten due to update)
```
### Merge-on-Read
In MOR mode, updates and deletes only add new files without rewriting existing ones. There are two types:
- **Position Deletes**: Logically delete rows by adding delete files referencing the data file and row position.
- **Equality Deletes**: Use equality predicates to filter rows based on specific conditions.
## Compaction
Compaction is the process of rewriting many small files into fewer, larger files to improve read efficiency. There are two types:

- **Data Compaction**: Reduces the number of data and delete files.
- **Manifest Compaction**: Optimizes the manifest files for better performance.

## Partitioning and Partition Evolution
### Partitioning
Iceberg supports transform-based partitioning, known as hidden partitioning. This allows partitioning based on transformations like `day(col)`, `hour(col)`, etc., without explicitly adding partitioning columns.
### Partition Evolution
Partition specs can evolve over time, allowing changes in partitioning strategy without rewriting data files.
```plaintext
Initial Partition Spec: hour(col)
Evolved Partition Spec: day(col)
```
## Write Path and Concurrency Control
### Write Path
The write path in Iceberg involves writing data files first, followed by committing metadata files.
### Concurrency Control
Iceberg supports multiple concurrent writers with Serializable or Snapshot isolation levels. Data conflict checks are performed before committing, ensuring consistency.
## Mapping COW and MOR Operations to Iceberg Code
### Table Interface
The `Table` interface in the Iceberg API module offers various write operations:
- **AppendFiles**: Adds new files (FastAppend and MergeAppend implementations).
- **OverwriteFiles**: Performs COW operations.
- **RowDelta**: Performs MOR operations.
- **RewriteFiles**: Handles compaction.
### Critical Commit Process
The commit process is handled by the `SnapshotProducer` class, ensuring atomicity and consistency.
## Example Code Snippets
### Writing Data to Iceberg Tables

```java
import org.apache.iceberg.Table;
import org.apache.iceberg.catalog.TableIdentifier;
import org.apache.iceberg.flink.TableLoader;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

// Initialize Flink and Iceberg Table
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
StreamTableEnvironment tableEnv = StreamTableEnvironment.create(env);
TableLoader tableLoader = TableLoader.fromHadoopTable("path/to/iceberg/table");
Table table = tableLoader.loadTable(TableIdentifier.of("namespace", "table_name"));

// Define Flink Job to Write Data
tableEnv.executeSql("INSERT INTO iceberg_table SELECT * FROM kafka_source");

// Execute the Job
env.execute("Flink to Iceberg");
```
### Compaction Strategy
```java
import org.apache.iceberg.Actions;
import org.apache.iceberg.spark.SparkActions;
import org.apache.spark.sql.SparkSession;

// Initialize Spark and Iceberg
SparkSession spark = SparkSession.builder().appName("Iceberg Compaction").getOrCreate();
Actions actions = SparkActions.get();

// Perform Compaction
actions.rewriteDataFiles("path/to/iceberg/table")
  .option("target-file-size-bytes", 512 * 1024 * 1024)
  .execute();
```
## Conclusion
Apache Iceberg provides a robust and scalable consistency model for managing large-scale data lakes. Its support for both COW and MOR modes, combined with efficient metadata management and partitioning strategies, makes it a powerful tool for high-performance analytics. Understanding the internals of Iceberg helps in designing efficient and reliable data processing pipelines.