https://www.uber.com/blog/selective-column-reduction-for-datalake-storage-cost-efficiency/?uclick_id=336a56a2-cc2d-40c9-b908-10ed680e0404

**Date**: September 20  
**Focus**: Reducing data size within the Apache Parquet™ file format by selectively deleting unused large size columns.

## Background

- **Problem**: Rapid growth of Uber's data led to escalating storage and compute resource costs.
- **Challenges**:
    - Increased hardware requirements
    - Heightened resource consumption
    - Performance issues: out-of-memory errors, prolonged garbage collection pauses
- **Initiatives**:
    - Implementation of Time to Live (TTL) policies for aging partitions.
    - Adoption of tiered storage strategies (hot/warm to cold storage tiers).
    - Optimization of data size at the file format level.

## Apache Parquet™ Data Structure

- **What**: Main data lake storage file format at Uber.
- **Nature**: Columnar file format. All data of a column is stored together.
- **Structure**:
    - Data is partitioned into multiple row groups.
    - Each row group contains column chunks corresponding to dataset columns.
    - Data in each column chunk is stored in pages.
    - A block consists of pages. A block is the smallest unit that must be entirely read to access a record.
    - Within a page: Each field is appended with its value, repetitive level, and definition level.
- **References**: [Official Parquet™ file format documentation](https://chat.openai.com/c/e17f4791-650b-4805-b569-9a7488f899ae#).

## Possible Solution

- **Traditional Approach**:
    - Use Spark™ to read data, eliminate unnecessary columns, and write back the dataset.
    - Involves decryption, decompression, decoding during reading. Encoding, compression, and encryption during writing.
    - These steps are resource-intensive and time-consuming.
- **Need**: A solution to minimize reliance on costly steps for efficiency.

## Selective Column Reduction Solution

- **Idea**: Utilize the columnar nature of Parquet™. Just copy the original data for required columns and ignore/skip the columns to be removed.
- **Process**:
    - Copy data for a given column and send back to disk.
    - Skip columns that need to be removed.
    - Update metadata to reflect changes.
    - No data reshuffling needed.
- **Advantages**:
    - Vast improvement by bypassing expensive steps.
    - No reshuffling required, which is another cost-saver.

## Benchmarking

- **Tests**:
    - Conducted on files of sizes 1G, 100M, and 1M.
    - Used Apache Spark™ as the baseline.
- **Results**:
    - Selective Column Pruner was ~27x faster for ~1G file and ~9x faster for 1M file.

## Parallelization of Execution

- **Need**: Process multiple files in large datasets.
- **Solution**: Use Apache Spark™ Core to parallelize execution, but not its Data Source.
- **Execution Process**:
    - Handle dataset as the unit.
    - List all partitions for partitioned tables.
    - List all files within each partition.
    - Assign tasks to each file to execute column reduction using the Selective Column Pruner.

## Conclusion

The selective column reduction approach in Parquet™:

- Efficiently optimizes data storage.
- Reduces storage costs and resource consumption.
- Enhances system performance.
- Uber's ongoing efforts: Improve data management and minimize infrastructure costs.

## Action Items / Next Steps

- Consider implementing this strategy in similar environments with large data storage needs.
- Monitor and adjust based on future data growth and storage demands.