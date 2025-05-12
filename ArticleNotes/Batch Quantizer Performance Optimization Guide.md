  
## Current Performance  
  
Currently, the Batch Quantizer application processes data with the following characteristics:  
- Running on a 15-worker cluster (360 cores)  
- Taking approximately 13 minutes to complete  
- Seeing minimal improvements when adding more executors  
  
This document outlines strategies to significantly improve performance beyond simply adding more resources.  
  
## 1. Spark Configuration Optimizations  
  
### Memory Management  
  
```bash  
# Increase memory allocation with proper overhead  
--conf spark.executor.memory=100g  
--conf spark.executor.memoryOverhead=25g  # Increase from current 16g to 25% of executor memory  
--conf spark.driver.memory=16g  
--conf spark.driver.memoryOverhead=4g  
  
# Reduce GC pressure  
--conf spark.memory.fraction=0.8  # Increase from default 0.6  
--conf spark.memory.storageFraction=0.3  # Adjust based on caching needs  
--conf spark.cleaner.periodicGC.interval=30  # Run GC periodically every 30s  
```  
  
### Shuffle and Serialization  
  
```bash  
# Optimize shuffle performance  
--conf spark.shuffle.file.buffer=1m  # Increase from default 32k  
--conf spark.shuffle.unsafe.file.output.buffer=5m  # Increase buffer size  
--conf spark.shuffle.service.index.cache.size=2048  # Increase index cache  
--conf spark.shuffle.registration.timeout=120000  # Increase timeout for large shuffles  
  
# Improve serialization  
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer  
--conf spark.kryoserializer.buffer.max=128m  
--conf spark.sql.inMemoryColumnarStorage.compressed=true  
--conf spark.sql.inMemoryColumnarStorage.batchSize=20000  # Increase batch size  
```  
  
### Parallelism and Partitioning  
  
```bash  
# Optimize task parallelism  
--conf spark.sql.shuffle.partitions=1000  # Adjust based on data volume  
--conf spark.default.parallelism=1000  # Match with shuffle partitions  
--conf spark.sql.adaptive.enabled=true  # Enable adaptive query execution  
--conf spark.sql.adaptive.coalescePartitions.enabled=true  
--conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128m  
```  
  
### I/O Optimization for S3  
  
```bash  
# S3 specific optimizations  
--conf spark.hadoop.fs.s3a.connection.maximum=1000  # More concurrent connections  
--conf spark.hadoop.fs.s3a.connection.timeout=1200000  
--conf spark.hadoop.fs.s3a.attempts.maximum=20  
--conf spark.hadoop.fs.s3a.connection.establish.timeout=5000  
--conf spark.hadoop.fs.s3a.threads.max=20  # More threads for S3 operations  
```  
  
## 2. Code Modifications for Performance  
  
### Dataset Caching  
  
Add strategic caching of frequently used datasets:  
  
```java  
// After reading the repartitioned data  
Dataset<Row> repartitionedData = readRepartitionedData(startTime, batchDuration, repartitionedDataTableName, spark)  
    .cache();  // Cache after reading but before multiple uses  
// Use optimized storage level if memory is a concern  
.persist(StorageLevel.MEMORY_AND_DISK_SER())  
```  
  
### Partition Optimization  
  
Modify data reading to leverage partitioning:  
  
```java  
private static Dataset<Row> readIcebergData(long startTime, int interval, String repartitionTableName, SparkSession spark) {  
    // Existing code...        // Add partition hint based on data distribution  
    return spark.read()            .table(icebergTableName)            .filter(functions.col(Constants.WALL_TIMESTAMP).gt(startTime)                    .and(functions.col(Constants.WALL_TIMESTAMP).leq(startTime + interval)))            .repartition(200, functions.col("org_id"))  // Repartition by primary key            .cache();  // Cache after filtering}  
```  
  
### Optimized Writing  
  
Modify write operations to optimize performance:  
  
```java  
private static void writeToS3UsingIceberg(Dataset<Row> rollups) {  
    // Existing code...        // Repartition before writing to optimize file sizes  
    transformedDF = transformedDF            .repartition(200, col("org_id"), col("resolution"))  // Match partitioning scheme            .sortWithinPartitions("metric_name");  // Sort within partitions                // Use bulk insert for better performance  
    transformedDF.writeTo(concatTableName)            .option("write-format", "data")  // Use data files, not delete files            .option("fanout-enabled", "true")  // Enable fanout writers            .option("target-file-size-bytes", "134217728")  // Target 128MB files            .append();}  
```  
  
## 3. Algorithmic Improvements  
  
### Batch Processing Optimization  
  
Split large batches into smaller chunks to improve parallelism:  
  
```java  
// In main method  
if (batchDuration > 600000) {  
// If batch is larger than 10 minutes  
    // Process in 5-minute chunks for better parallelism    
    int chunkSize = 300000;  // 5 minutes in ms  
     List<Dataset<Row>> results = new ArrayList<>();       
     for (long chunkStart = startTime; chunkStart < startTime + batchDuration; chunkStart += chunkSize) {  
	long chunkEnd = Math.min(chunkStart + chunkSize, startTime + batchDuration);        Dataset<Row> chunkData = readRepartitionedData(chunkStart, chunkEnd - chunkStart, repartitionedDataTableName, spark);        // Process chunk...        results.add(processedChunkData);    }        // Union the results  
    Dataset<Row> combinedResults = results.get(0);    for (int i = 1; i < results.size(); i++) {        combinedResults = combinedResults.unionAll(results.get(i));    }}  
```  
  
### Optimized Aggregation  
  
Replace the current rollup computation with a more efficient approach:  
  
```java  
static RollupResult computeRollups(SparkSession spark, Dataset<Row> repartitionedData,  
                                     int outputResolution, String partialFilePath) {    // Cache after initial projection    Dataset<Row> batchDatapoints = repartitionedData.selectExpr(REQUIRED_COLUMNS_INPUT).cache();        // Read partials efficiently  
    PartialUtil.PartialDatapointsResult partialDatapoints = readPartial(spark, partialFilePath);        // Optimize union with broadcast if partials are small  
    Dataset<Row> datapoints;    if (partialDatapoints.partialPoints().count() < 10000) {        Dataset<Row> broadcastPartials = broadcast(partialDatapoints.partialPoints());        datapoints = batchDatapoints.unionAll(broadcastPartials);    } else {        datapoints = batchDatapoints.unionAll(partialDatapoints.partialPoints());    }        // Add repartitioning before heavy computation  
    datapoints = datapoints.repartition(col("org_id"), col("metric_name"));        // Rest of the method...  
}  
```  
  
## 4. Monitor and Profile Performance  
  
Add performance monitoring to identify bottlenecks:  
  
```java  
// At key points in your code  
long startTimeMs = System.currentTimeMillis();  
// ... operation ...  
long durationMs = System.currentTimeMillis() - startTimeMs;  
logMessage(String.format("Operation X took %d ms (%.2f seconds)", durationMs, durationMs / 1000.0));  
```  
  
Set up Spark UI monitoring to identify:  
- Skewed partitions  
- Spill to disk  
- Task duration anomalies  
- Shuffle read/write volumes  
  
## 5. Iceberg-Specific Optimizations  
  
### Table Properties  
  
```sql  
-- Set these properties on your Iceberg tables  
ALTER TABLE metrics_repartitioner_catalog.batch_quantizer_db_lab0.repartitioned_data_lab0_new_v7  
SET TBLPROPERTIES (  
  'read.split.target-size'='268435456',  -- 256MB split size  'write.distribution-mode'='hash',  'write.metadata.delete-after-commit.enabled'='true',  'write.metadata.previous-versions-max'='10',  'format-version'='2'  -- Use Iceberg v2 format for better performance);  
```  
  
### Data Organization  
  
```sql  
-- Optimize your table periodically  
CALL metrics_repartitioner_catalog.system.rewrite_data_files(  
  table => 'batch_quantizer_db_lab0.repartitioned_data_lab0_new_v7',  strategy => 'binpack',  options => map('target-file-size-bytes', '536870912')  -- 512MB files);  
```  
  
## 6. Executor Resource Allocation  
  
For a 15-worker cluster, consider:  
  
```bash  
# Adjust number of executors and cores per executor  
--num-executors=30  
--executor-cores=10  # Match to available CPU cores per node  
--executor-memory=100g  
```  
  
The ideal balance depends on your hardware:  
- On nodes with 24 cores and 256GB RAM: 2 executors per node, 10-12 cores each  
- On nodes with 16 cores and 128GB RAM: 1-2 executors per node, 8 cores each  
  
## 7. Java & JVM Tuning  
  
```bash  
# Add to spark.executor.extraJavaOptions and spark.driver.extraJavaOptions  
--conf "spark.executor.extraJavaOptions=-XX:+UseG1GC -XX:+UnlockDiagnosticVMOptions -XX:+G1SummarizeConcMark -XX:InitiatingHeapOccupancyPercent=35 -XX:G1HeapRegionSize=16m -XX:ConcGCThreads=4"  
```  
  
## 8. Implementation Checklist  
  
For best results, implement these changes in this order:  
  
1. **Basic Spark Configuration**:  
   - Memory settings  
   - Serialization improvements  
   - Adaptive query execution  
  
2. **Code Improvements**:  
   - Add strategic caching  
   - Optimize reading with partition pruning  
   - Improve write operations  
  
3. **Advanced Optimizations**:  
   - Tune shuffle parameters  
   - Implement batch chunking if needed  
   - Optimize Iceberg table properties  
  
4. **Resource Tuning**:  
   - Adjust executor/core ratios  
   - Fine-tune memory allocation  
   - JVM garbage collection settings  
  
## 9. Expected Performance Gains  
  
With these optimizations properly implemented, you can expect:  
  
- **30-50% reduction in processing time** from current 13 minutes to ~6-9 minutes  
- **Better scaling** with additional resources  
- **More efficient resource utilization**  
- **Reduced S3 data transfer costs**  
- **More consistent performance** across varying data volumes  
  
Monitor key metrics like shuffle read/write sizes, GC time, and task skew to continuously refine your configuration.