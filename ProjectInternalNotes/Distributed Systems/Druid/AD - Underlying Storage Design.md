https://juejin.cn/post/6844904110626439175
#### I. Introduction to Druid's Storage Method
- **Integration of Features**: Druid combines elements of data warehouses, full-text retrieval systems, and time series databases.
- **Core Component**: The underlying data storage method is key to Druid's multifaceted capabilities.
#### II. Druid's Data Model
- **Three Basic Column Types**: Druid's data model includes timestamp columns, dimension columns, and metric columns.
- **Segment Files**: Druid stores data in segment files, partitioned by time.
    - **Granularity**: Configurable through `granularitySpec` parameters like `segmentGranularity`.
    - **Recommended File Size**: Ideally between 300mb-700mb for optimal performance.
#### I. Druid's Data Structures in Detail
- **Timestamp and Metric Columns**:
  - Simple arrays of integers or floats, compressed using LZ4.
  - When queried, relevant rows are unpacked for aggregation operations.
  - Skipped if not required by the query.
#### II. Dimension Columns and Their Data Structures
- **Dictionary Encoding**: Maps string values to integer IDs for compact representation.
  ```json
  {
    "Justin Bieber": 0,
    "Ke$ha": 1
  }
  ```
- **Column Data Structure**:
  - An array representing the column values.
  - Example: `[0, 0, 1, 1]` (simple values) or `[0, [0, 1], 1, 1]` (multi-value columns).
- **Bitmaps (Inverted Indexes)**:
  - Used for fast filtering operations.
  - Bitmaps are especially effective for AND and OR operations.
  - Example: 
    ```json
    {
      "value='Justin Bieber'": [1, 1, 0, 0],
      "value='Ke$ha'": [0, 0, 1, 1]
    }
    ```
  - Roaring bitmap compression is used for efficient bitmap storage.
#### III. Segment File Identification and Structure
- **Segment File Composition**:
  - `version.bin`: Contains the version of the segment (e.g., 0x0, 0x0, 0x0, 0x9 for v9 segments).
  - `meta.smoosh`: Metadata file storing information about other smoosh files.
  - `XXXXX.smoosh`: Binary data storage files, up to 2GB each, containing data for each column.
  - `index.drd`: Contains additional metadata about the segment.
  - Special column `__time`: Represents the time column of the segment.

- **Naming Conventions**:
  - Segment identifiers consist of data source, start and end times in ISO 8601 format, version number, and partition number if sharded.
  - Example: `datasource_intervalStart_intervalEnd_version_partitionNum`.
#### IV. Data Fragmentation and Mode Changes
- **Sharded Data Handling**:
  - Druid handles multiple segments within the same interval as part of a block.
  - Queries complete only when all shards of a block are loaded.
- **Segment Replacement and Versioning**:
  - Newer versions of segments replace older ones atomically.
  - Batch indexing ensures that newer version segments are used once fully loaded into the cluster.
#### V. Column Format and Sharded Data
- **Column Storage Breakdown**:
  - Each column is stored in two parts: a Jackson serialized `ColumnDescriptor` and binary data.
  - The `ColumnDescriptor` contains metadata and serialization/deserialization logic.
- **Handling Multi-Value Columns**:
  - Multi-value columns have a unique data structure representation.
  - Example: A row with multiple values for a column will have an array of values in "column data."
#### VI. Summary
- **In-Depth Understanding**: These expanded notes offer a more detailed understanding of Druid's storage design, specifically its segment file structure, data partitioning, and handling of different data types.
- **Advanced Features**: The unique approach to dictionary encoding, bitmap indexing, and segment versioning are crucial to Druid's efficiency in data processing.
---
