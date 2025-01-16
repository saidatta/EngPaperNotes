aliases: [Incremental Processing, Netflix Maestro, Apache Iceberg, IPS, Big Data Workflows]
tags: [big data, incremental processing, iceberg, maestro, data pipelines, Netflix]
## Overview
This note provides a **technical deep dive** into how Netflix uses **Incremental Processing** via **Maestro** (Netflix’s workflow orchestrator) and **Apache Iceberg** to address challenges in **data freshness**, **data accuracy**, and **backfill** support for large-scale data pipelines.
## Table of Contents
1. [Introduction](#introduction)  
2. [Challenges in Big Data Workflows](#challenges-in-big-data-workflows)  
3. [Incremental Processing Approach](#incremental-processing-approach)  
4. [Key Components](#key-components)  
   - [Netflix Maestro](#netflix-maestro)  
   - [Apache Iceberg](#apache-iceberg)  
5. [Incremental Change Capture Design](#incremental-change-capture-design)  
6. [Advantages of This Solution](#advantages-of-this-solution)  
7. [Incremental Processing Patterns](#incremental-processing-patterns)  
8. [Use Cases and Example Pipeline](#use-cases-and-example-pipeline)  
   - [Original Pipeline with Lookback Window](#original-pipeline-with-lookback-window)  
   - [New Pipeline with Incremental Processing](#new-pipeline-with-incremental-processing)  
9. [Looking Forward](#looking-forward)  
10. [References & Acknowledgements](#references--acknowledgements)

---
## Introduction
Netflix relies on data-driven insights for **A/B tests**, **content recommendation**, **studio production**, **security**, and **billing**. As Netflix’s scale grows, so do the **low-latency** and **accurate** processing needs:
- **Data Freshness**: Quickly process newly arrived or changed data to drive near-real-time analytics or operational insights.
- **Data Accuracy**: Accommodate **late arriving** data to ensure computations remain correct and consistent, without repetitive large reprocessing.
- **Backfill**: Easily reprocess historical data (e.g., for newly introduced metrics or fixed logic) without ad hoc manual pipelines.
Incremental Processing (IPS) solves these needs by processing **only new or changed** data rather than large reprocessing of entire datasets.

---
## Challenges in Big Data Workflows

| Challenge                         | Description                                                                                                                                                                                                                 |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Data Freshness**               | Large Iceberg tables must be processed quickly. Hourly jobs with coarse triggers often reprocess too much data.                                                                      |
| **Data Accuracy**                | Datasets can become **incomplete** due to late arriving data. Traditional solutions rely on **lookback windows** (e.g., reprocess last *N* days). This is costly in time and compute.   |
| **Backfill**                     | Historical reprocessing for changes in upstream data logic or new metrics can be laborious. Requires building “foreach” backfill workflows, leading to **manual orchestration**.        |

### Common Suboptimal Approaches
1. **Lookback Windows**  
   - *Pro*: Simple to implement.  
   - *Con*: Recomputes large swaths of already processed data, wasting time and compute resources.
2. **Manual Backfill Workflows**  
   - *Pro*: Flexible for single pipeline scenarios.  
   - *Con*: Doesn’t scale to multi-stage pipelines. Requires significant manual configuration.

---
## Incremental Processing Approach
**Incremental processing** only processes **new or changed** data and merges or appends these results to target tables. This includes:
1. **State Tracking** of new data or late arriving data.  
2. **Change Capture**: Identifies partitions/rows that need recomputation.  
3. **Merge/Append Logic**: Efficiently update the target table without duplicating older data.

**ASCII Visualization**:
```
Sources (Table A, Table B) --> Capture new data changes --> Incremental Pipeline --> Updated Target
```
By focusing on incremental data, pipelines become faster, more cost-efficient, and more accurate.
![[Screenshot 2025-01-08 at 1.48.54 PM.png]]
---
## Key Components

### Netflix Maestro
**Maestro** is Netflix’s next-gen workflow orchestrator:
- **General-purpose** orchestration with a fully managed service.
- Serves thousands of **data scientists**, **engineers**, **analysts**.
- Provides building blocks such as:
  - **Trigger mechanisms** (time-based, data-based, or now incremental-based).
  - **Step job types** (Spark jobs, custom tasks, etc.).
  - **Foreach patterns** for distributed parallel sub-jobs.
**Incremental Processing (IPS)** extends Maestro by introducing:
- A **new trigger mechanism** for incremental changes.
- A **new step job type** that captures only new/changed data.
#### Maestro Example Configuration
```yaml
workflow:
  name: "playback_daily_workflow"
  steps:
    - name: "capture_icdc"
      type: "ICDC"
      sourceTable: "db.playback_table"
      targetTable: "db.playback_icdc_table"
    - name: "transform_new_data"
      type: "spark"
      script: "merge_incremental_spark_job.py"
      dependsOn:
        - "capture_icdc"
```
---
### Apache Iceberg
**Iceberg** is an open table format for huge analytic datasets:
- **Immutable Snapshots**: Each commit references new data files.
- **Hidden Partitioning** and **Schema Evolution**.
- **Time Travel & Rollback**: Maintains history of table changes.

**Incremental Processing** leverages Iceberg’s snapshot metadata to track only **new files** or **late data** without copying them. This approach, sometimes called **lightweight ICDC** (Incremental Change Data Capture), references existing data files rather than rewriting them.

---
## Incremental Change Capture Design
By combining **Netflix Maestro** and **Apache Iceberg**, a **lightweight** approach is developed:
1. **ICDC Table**: Create a special *“ICDC”* Iceberg table that references **only** the newly added data files from the original source table.
2. **Metadata-based**: Rely on Iceberg’s snapshot metadata to identify which partitions or rows have changed.
3. **Granular Tracking**: Maestro tracks at the data-file or partition level to know what needs reprocessing.

```mermaid
flowchart LR
    A[Iceberg Source Table (Snapshots)] --> B{Identify New Data Files}
    B --> C[ICDC Table (Lightweight references)]
    C --> D[Workflow Process (Spark Merge/Append)]
    D --> E[Target Table Updated]
```

**Key Insight**:  
No data duplication. The ICDC table reuses the **same data files** of the source table, drastically lowering cost.

---

## Advantages of This Solution

1. **Cost Efficiency**:  
   - Only process changed partitions/rows.  
   - No large-scale re-scans for every run.

2. **Simplicity**:  
   - No complicated manual logic to detect changed partitions.  
   - Minimal code changes from existing ETL frameworks.

3. **Integration with Maestro**:  
   - Thousands of existing workflows can gradually adopt IPS.  
   - Consistent patterns for **triggers**, **dependencies**, and **step** definitions.

4. **Scalable for Multi-Stage**:  
   - Downstream workflows seamlessly ingest ICDC changes from upstream.  
   - Eliminates the repeated backfill chores in multi-step DAGs.

---

## Incremental Processing Patterns
During onboarding, engineers discovered various usage patterns:

1. **Direct Append**  
![[Screenshot 2025-01-08 at 1.49.12 PM.png]]
   - If the changed data alone is sufficient, the workflow directly appends new results to the target table.  
   ```sql
   -- Pseudo Spark SQL example
   INSERT INTO target_table
   SELECT * FROM icdc_source_table
   ```

2. **Merge Pattern**  
   - If partial updates or idempotency are needed, **MERGE INTO** is used.  
   ```sql
   MERGE INTO target_table AS T
   USING icdc_source_table AS S
   ON T.key = S.key
   WHEN MATCHED THEN UPDATE SET ...
   WHEN NOT MATCHED THEN INSERT ...
   ```
![[Screenshot 2025-01-08 at 1.49.26 PM.png]]
3. **ICDC as Filter**  
   - The ICDC table is joined on some grouping key to prune the original source table for re-aggregation. Only keys with changes are processed.  
   ```sql
   SELECT main.*
   FROM main_source_table main
   JOIN icdc_table icdc
       ON main.group_id = icdc.group_id
   ```

4. **Range-based Processing**  
![[Screenshot 2025-01-08 at 1.49.35 PM.png]]
   - If transformations require scanning broader data (e.g., re-calculating rolling windows), the ICDC metadata reveals the **range** of partitions to refresh.  
   ```txt
   # Example: If partition 'day=2024-10-01' received new data, 
   # reprocess only from 'day=2024-10-01' onward.
   ```

---

## Use Cases and Example Pipeline

### Original Pipeline with Lookback Window
**Scenario**:  
- A pipeline that processes `14 days` of data in every run to handle late arrivals.  
- Two stages: 
  1. `playback_daily_workflow` => `playback_daily_table`  
  2. `playback_daily_agg_workflow` => `playback_daily_agg_table`

**Diagram**:
![[Screenshot 2025-01-08 at 1.49.45 PM.png]]
**Performance** (example):
- Stage 1: ~7 hours to reprocess 14 days.  
- Stage 2: ~3.5 hours to re-aggregate 14 days.  
- ~10.5 hours total. Large resource usage.

---
### New Pipeline with Incremental Processing
By incorporating **ICDC tables** and **MERGE** operations:
1. `ips_playback_daily_workflow`  
   - Reads only **new** data from `playback_icdc_table` referencing `playback_table`.  
   - Merges new or changed data into `playback_daily_table`.
2. `ips_playback_daily_agg_workflow`  
   - Captures changes in `playback_daily_table` via `playback_daily_icdc_table`.  
   - Joins these changes with the main dataset to re-aggregate only necessary partitions.
![[Screenshot 2025-01-08 at 1.49.55 PM.png]]
**Performance Gains**:
- Stage 1: ~30 minutes (instead of 7 hours).  
- Stage 2: ~30 minutes total (~15 min for previous partitions + 15 min for current day).  
- **~1 hour total** vs. ~10.5 hours before. Over **80%** cost reduction.
---
## Looking Forward
1. **Beyond Append Cases**:  
   - Track data changes for **overwrite** or **delete** operations.  
   - Support partial or complex transformations.
2. **Managed Backfill Support**:  
   - Automatic reprocessing of historical data.  
   - Propagate changes through multi-stage DAGs seamlessly.
3. **Enhanced Orchestration**:  
   - Provide advanced analytics on incremental data.  
   - Integrate more deeply with Netflix’s data governance and lineage tools.

---
## References & Acknowledgements
- **Apache Iceberg**: [Official Project Page](https://iceberg.apache.org/)  
- **Netflix Maestro**: [Previous Netflix TechBlog on Maestro](https://netflixtechblog.com/)  
- **Incremental Processing** concepts have parallels in CDC (Change Data Capture) solutions.

**Acknowledgements**:  
- **Authors**: Jun He, Yingyi Zhang, and Pawan Dixit (Netflix)
- **Contributors**: Andy Chu, Kyoko Shimada, Abhinaya Shetty, Bharath Mummadisetty, John Zhuge, Rakesh Veeramacheneni, etc.
- **Leaders**: Prashanth Ramdas, Eva Tse, Charles Smith for strategic feedback on architecture.

> **Key Takeaway**: **Incremental Processing** in Netflix Maestro + Apache Iceberg significantly **reduces cost**, **accelerates data freshness**, and **improves accuracy** by only processing changed data. Its modular design allows minimal business-logic changes, enabling large-scale adoption across Netflix’s data ecosystem.  