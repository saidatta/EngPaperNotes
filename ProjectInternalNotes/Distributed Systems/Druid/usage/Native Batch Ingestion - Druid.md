#### 1. **Overview of Native Batch Ingestion**
   - Apache Druid supports **native batch ingestion** to load data from various sources like S3, HDFS, and local files.
   - Two primary task types:
     - **Index Parallel Task:** Runs multiple subtasks concurrently to speed up ingestion, ideal for large datasets.
     - **Index Task:** Single-threaded ingestion, used for testing or smaller datasets.

```
+-------------------------------+
|     Apache Druid Ingestion    |
|   +-----------------------+   |
|   |    Index Parallel     |   | <-- Multi-threaded
|   +-----------------------+   |
|   |       Index           |   | <-- Single-threaded
|   +-----------------------+   |
+-------------------------------+
```
#### 2. **Parallelism in Native Batch Ingestion**
   - **Index Parallel Task:** Defines the degree of parallelism by specifying the number of concurrent subtasks.
     - Subtasks can run in parallel, each responsible for ingesting a subset of data, increasing ingestion speed.
   - **Index Task:** Processes the entire ingestion serially, suitable for development or test environments.
#### 3. **Setting Up S3 Ingestion in Druid**
   - To load data from S3, ensure the **Druid S3 extension** is enabled.
   - Add the extension to the **common runtime properties file**:
     - **Path:** `conf/druid/single-server/micro-quickstart/common.runtime.properties`
     - **Configuration:**
    ```properties
       druid.extensions.loadList=["druid-s3-extensions"]
       ```
   - **Restart Apache Druid** to load the extension.

#### 4. **Practical Example: Loading Data from S3**
   - **Data:** COVID-19 worldwide aggregated data in CSV format stored in an S3 bucket.
   - **CSV File Example:**
     ```
     date,confirmed,recovered,deaths,increase_rate
     2024-04-14,100000,95000,5000,0.1
     2024-04-15,105000,98000,7000,0.05
     ```
   
   - **Specify S3 Details in the Druid Console:**
     - **Source Type:** S3
     - **S3 URI:** Path to the CSV file in the S3 bucket.
     - **S3 Access Key & Secret Key:** Required for authentication.
   
   - **Step-by-Step Configuration:**
     1. **Click "Load Data" → "Batch Classic Ingestion" → "AWS S3".**
     2. **Connect to S3:**
        - Specify S3 URI, access key, and secret key.
        - Verify the connection to ensure Druid can read the file.

#### 5. **Creating `spec.json` for Index Parallel Task**
   - The ingestion spec defines how data is loaded into Druid, including task type, data source, schema, partitioning, tuning, and publishing.
   - **Example `spec.json` for Parallel Index Task:**
     ```json
     {
       "type": "index_parallel",
       "spec": {
         "dataSchema": {
           "dataSource": "covid_stats",
           "timestampSpec": {
             "column": "date",
             "format": "yyyy-MM-dd"
           },
           "dimensionsSpec": {
             "dimensions": ["confirmed", "recovered", "deaths", "increase_rate"]
           },
           "granularitySpec": {
             "segmentGranularity": "day",
             "queryGranularity": "none",
             "rollup": false
           }
         },
         "ioConfig": {
           "type": "index_parallel",
           "inputSource": {
             "type": "s3",
             "uris": ["s3://druid-batch-ingestion/covid_data.csv"]
           },
           "inputFormat": {
             "type": "csv",
             "columns": ["date", "confirmed", "recovered", "deaths", "increase_rate"]
           }
         },
         "tuningConfig": {
           "type": "index_parallel",
           "maxNumConcurrentSubTasks": 2
         }
       }
     }
     ```
   
   - **Explanation:**
     - `"type": "index_parallel"`: Uses parallel ingestion for faster processing.
     - `"maxNumConcurrentSubTasks": 2`: Runs two concurrent tasks.
     - `"segmentGranularity": "day"`: Creates daily segments.
     - `"rollup": false`: No data aggregation during ingestion.

#### 6. **Submitting Ingestion Task via HTTP API**
   - **Using Postman to Submit the Ingestion Task:**
     1. **Set the URL to the Coordinator/Overlord API:** `http://localhost:8081/druid/indexer/v1/task`
     2. **Set HTTP Method to POST** and paste the `spec.json` into the body.
   
   - **Example API Call:**
     ```bash
     curl -X POST -H 'Content-Type: application/json' -d @spec.json http://localhost:8081/druid/indexer/v1/task
     ```
   
   - **Response:**
     ```json
     {
       "task": "index_parallel_covid_stats_2024-04-14T10:00:00.000Z"
     }
     ```
   
   - **Monitor the Status of the Ingestion Task:**
     - Use the task ID from the response to check the status.
     - **Status API URL:** 
       ```bash
       curl -X GET http://localhost:8081/druid/indexer/v1/task/index_parallel_covid_stats_2024-04-14T10:00:00.000Z/status
       ```

#### 7. **Task Management and Parallelism in Druid**
   - **Subtask Management:** The supervisor task manages parallel subtasks, splitting the data ingestion workload across multiple workers.
   - **Parallel Task Execution:**
     - Each subtask processes a subset of the data, writing segments independently.
     - Subtasks use distributed coordination to ensure no data overlap.
   
   - **Append vs. Overwrite Modes:**
     - **Append Mode:** New rows are added to existing segments.
       - Suitable for adding new data, creating multiple versions of segments.
     - **Overwrite Mode:** Replaces existing rows with new rows for the same timestamp.
       - Useful for updating records or correcting data inconsistencies.

#### 8. **ASCII Diagram for Parallel Ingestion**
   ```
   +--------------------+
   |    Coordinator     | 
   +--------------------+
           |      
           v
   +------------------------+
   |  Parallel Supervisor   |
   +------------------------+
      |     |          |
      v     v          v
   +------+ +------+  +------+
   |Task 1| |Task 2|  |Task 3|  <-- Subtasks
   +------+ +------+  +------+
      |       |         |
      v       v         v
   [ Seg 1 ] [ Seg 2 ] [ Seg 3 ]  <-- Segments Created
   ```

#### 9. **Handling Failure and Retrying in Parallel Tasks**
   - **Task Failure Handling:**
     - If a subtask fails, the supervisor retries it based on configured retry policies.
     - **Retry Configuration:** Specify max retries in the `tuningConfig`.
   
   - **Example Retry Configuration:**
     ```json
     "tuningConfig": {
       "type": "index_parallel",
       "maxNumConcurrentSubTasks": 2,
       "maxRetry": 3
     }
     ```
   
   - **Outcome:** If any subtask fails, the supervisor retries it up to 3 times.

#### 10. **Querying Ingested Data**
   - **Query Example:**
     - Use Druid SQL to verify data ingestion:
       ```sql
       SELECT date, confirmed, recovered, deaths, increase_rate 
       FROM covid_stats 
       WHERE date >= '2024-04-14';
       ```
   
   - **Output:**
     ```
     date        | confirmed | recovered | deaths | increase_rate
     ------------+-----------+-----------+--------+--------------
     2024-04-14  | 100000    | 95000     | 5000   | 0.1
     2024-04-15  | 105000    | 98000     | 7000   | 0.05
     ```

#### 11. **Automation and ETL Integration**
   - **Automating Batch Ingestion with Airflow:**
     - Use Airflow’s PythonOperator to automate API calls:
       ```python
       from airflow.operators.python_operator import PythonOperator

       def start_druid_ingestion():
           import requests
           url = 'http://localhost:8081/druid/indexer/v1/task'
           spec = open('spec.json').read()
           headers = {'Content-Type': 'application/json'}
           response = requests.post(url, data=spec, headers=headers)
           return response.json()

       task = PythonOperator(
           task_id='start_druid_ingestion',
           python_callable=start_druid_ingestion,
           dag=dag
       )
       ```

#### 12. **Conclusion**
   - **Parallel ingestion** in Druid enables faster data loading by distributing tasks

 across multiple subtasks.
   - By using **index_parallel**, you can achieve efficient scaling and handle large datasets in distributed environments.
   - Understanding task management, parallelism, and REST API integration is critical for efficient ETL pipelines in distributed systems like Apache Druid.

Let me know if you need any additional details or specific deep dives!