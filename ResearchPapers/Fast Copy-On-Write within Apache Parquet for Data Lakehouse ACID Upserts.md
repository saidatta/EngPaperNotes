## Overview
With the evolution of storage table formats such as **Apache Hudi®**, **Apache Iceberg®**, and **Delta Lake™**, more and more companies are building their lakehouses on top of these formats for various use cases, including incremental ingestion. However, the speed of upserts can become a significant issue when data volumes increase.

In storage tables, **Apache Parquet** is used as the main file format. This article discusses how we built a row-level secondary index and introduced innovations in Apache Parquet to speed up upsert operations within Parquet files. We also present benchmarking results that demonstrate much faster speeds compared to traditional copy-on-write methods in Delta Lake and Hudi.

## Motivation
Efficient table ACID upserts are critical for modern lakehouses. Use cases such as **Data Retention** and **Change Data Capture (CDC)** heavily rely on these operations. While Apache Hudi, Apache Iceberg, and Delta Lake are widely adopted, the performance of upserts can degrade when data volumes scale up, particularly in copy-on-write mode. This can lead to time-consuming and resource-intensive tasks or even block task completion.

To address this issue, we introduced partial copy-on-write within Apache Parquet files using a row-level index. This method skips unnecessary data pages, reading and writing efficiently. Generally, only a small portion of the file needs updating, and most data pages can be skipped, resulting in increased speed compared to traditional methods.

## Copy-On-Write in LakeHouse
We use **Apache Hudi** as an example, but similar concepts apply to Delta Lake and Apache Iceberg. Apache Hudi supports two types of upserts: **copy-on-write** and **merge-on-read**.

- **Copy-on-write**: All files containing records within the scope of updates are rewritten to new files, and new snapshot metadata is created.
- **Merge-on-read**: Delta files are added for updates, and readers merge them as needed.

### Example
Consider a partitioned table where one field (e.g., email) is updated. Logically, only the email field for User ID1 is replaced with a new value. Physically, the table data is stored in individual files grouped into partitions. Apache Hudi uses an indexing system to locate the impacted files, reads them completely, updates the email fields in memory, and writes the changes to disk as new files.

![Figure 1: Logic and physical file views of table upserts](path_to_image/figure1.png)

For large-scale tables, this process can be resource-intensive. For example, some tables at Uber required updates across 90% of the files, resulting in data rewrites of around 100 TB for any given large-scale table. This highlights the critical need for efficient copy-on-write operations.

## Introduce Row-Level Secondary Index
To improve copy-on-write within Apache Parquet, we introduced a **row-level secondary index** to locate data pages and accelerate the process.

### How It Works
The row-level secondary index is built when a Parquet file is first written or through offline reading. It maps records to [file, row-id] instead of just [file]. For instance, the RECORD_ID can be used as the index key, with FILE and Row_IDs pointing to files and their offsets.

![Figure 2: Row-level index for Apache Parquet](path_to_image/figure2.png)

Inside a Parquet file, data is partitioned into multiple row groups. Each row group consists of one or more column chunks corresponding to a column in the dataset. The data for each column chunk is written as pages, which are the smallest unit that must be read fully to access a single record. With the row-level index, updates can quickly locate the specific pages that need to be changed, skipping all other pages.

## Copy-On-Write within Apache Parquet
We introduced a new method to perform copy-on-write within Apache Parquet for fast upserts in Lakehouse. This method only updates related data pages and skips unrelated ones by copying them as bytebuffers without changes. This reduces the amount of data updated during an upsert operation, improving performance.

### Comparison with Traditional Methods
In traditional Apache Hudi upserts:
1. The record index locates files that need to be changed.
2. Files are read record by record into memory.
3. Records to be changed are searched and updated.
4. Data is written to disk as a whole new file.

This process involves expensive tasks such as de(re)-compression, de(re)-encoding, and record de(re)-assembling, consuming significant CPU cycles and memory.

To improve this, we use the row-level index and Parquet metadata to locate pages that need changes. For unrelated pages, we simply copy the data to the new file as bytebuffers without any modifications. We call this the "copy & update" process.

![Figure 3: Comparison of traditional copy-on-write in Apache Hudi and new copy-on-write](path_to_image/figure3.png)

### Detailed Process
1. **Locate Pages**: Use the row-level index to identify pages that need updates.
2. **Copy Unrelated Pages**: Copy unrelated pages as bytebuffers without changes.
3. **Update Related Pages**: Read, update, and write related pages.
4. **Write New File**: Combine copied and updated pages into a new file.

![Figure 4: The new copy-on-write within Parquet file](path_to_image/figure4.png)

## Benchmarking Results
We conducted benchmarking tests to compare the performance of our fast copy-on-write approach with traditional methods using TPC-DS data. We set up the test with out-of-box configurations, using the same number of vCores and memory settings for Spark jobs. We updated 5% to 50% of the data and compared the time consumed by Delta Lake and the new copy-on-write method.

The results show that our approach is significantly faster, with consistent performance gains across different percentages of updated data.

![Figure 5: Benchmarking results of new copy-on-write comparing with traditional Delta Lake](path_to_image/figure5.png)

**Disclaimer**: The benchmark on DeltaLake used default out-of-box configurations.

## Conclusion
Efficient ACID upserts are crucial for modern data lakehouses. While Apache Hudi, Delta Lake, and Apache Iceberg are widely adopted, the performance of upserts can degrade with increasing data volumes. To address this, we introduced partial copy-on-write within Apache Parquet files using a row-level index. This method skips unnecessary data pages, improving read and write efficiency. Our approach significantly speeds up upsert operations, enabling companies to efficiently perform data deletion, CDC, and other important use cases in the lakehouse.

## Code Snippets
### Example of Row-Level Index Creation
```python
import pyarrow.parquet as pq

def create_row_level_index(parquet_file_path):
    # Read Parquet file
    table = pq.read_table(parquet_file_path)
    
    # Create an empty index dictionary
    row_level_index = {}
    
    # Iterate over rows and build the index
    for i, row in enumerate(table.to_pandas().itertuples()):
        record_id = row.RECORD_ID
        if record_id not in row_level_index:
            row_level_index[record_id] = []
        row_level_index[record_id].append((parquet_file_path, i))
    
    return row_level_index

# Example usage
index = create_row_level_index('path_to_parquet_file.parquet')
print(index)
```

### Example of Partial Copy-On-Write
```python
import pyarrow as pa
import pyarrow.parquet as pq

def partial_copy_on_write(parquet_file_path, updates):
    # Read Parquet file
    table = pq.read_table(parquet_file_path)
    
    # Create a new table for the updated data
    new_data = []
    
    # Iterate over rows and apply updates
    for i, row in enumerate(table.to_pandas().itertuples()):
        record_id = row.RECORD_ID
        if record_id in updates:
            # Apply update to the specific field
            updated_row = row._asdict()
            updated_row['email'] = updates[record_id]
            new_data.append(updated_row)
        else:
            # Copy the row as is
            new_data.append(row._asdict())
    
    # Create a new table from the updated data
    new_table = pa.Table.from_pandas(pd.DataFrame(new_data))
    
    # Write the new table to a new Parquet file
    pq.write_table(new_table, 'path_to_new_parquet_file.parquet')

# Example usage
updates = {1: 'new_email@example.com'}
partial_copy_on_write('path_to_parquet_file.parquet', updates)
```

## Visualizations
### Figure 1: Logic and physical file views of table upserts
![Figure 1](path_to_image/figure1.png)

### Figure 2: Row-level index for Apache Parquet
![Figure 2](path_to_image/figure2.png)

### Figure 3: Comparison of traditional copy-on-write in Apache Hudi and new copy-on-write
![Figure 3](path_to_image/figure3.png)

### Figure 4: The new copy-on-write within Parquet file
![Figure 4](path_to_image/figure4.png)

### Figure 5: Benchmarking results of new copy-on-write comparing with traditional Delta Lake
![Figure 5](path_to_image/figure5.png)
```

This note provides a comprehensive overview of the topic, including detailed explanations, code snippets, and visualizations to help you understand and implement fast copy-on-write within Apache Parquet for data lakehouse ACID upserts.