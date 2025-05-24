
This document provides a *comprehensive*, *PhD-level* exploration of **dynamic filtering** in Trino. Dynamic filtering optimizes queries—particularly *star schema* or *fact-dimension* patterns—by pushing down **runtime filters** from a selective join side into the scan of a large fact table, thereby reducing I/O and boosting performance.

---

## 1. High-Level Concept

**Dynamic Filtering**:
- Collects actual **join key values** from the smaller (build) side of a join at query runtime.
- Uses those values to prune splits and/or filter rows for the larger (probe) side.
- Applies to **inner** and **right** joins with `=`, `<`, `<=`, `>`, `>=`, `IS NOT DISTINCT FROM`, and *semi-joins* with `IN` conditions.

**Result**: Trino avoids reading irrelevant partitions or row groups from the large fact table, significantly **reducing** data scanned and overall runtime.

### 1.1 Example Query

```sql
SELECT count(*)
FROM store_sales
JOIN date_dim ON store_sales.ss_sold_date_sk = date_dim.d_date_sk
WHERE d_following_holiday='Y'
  AND d_year=2000;
```

- `date_dim` is **selective**: only a small fraction of rows match the holiday/year conditions.
- **Without** dynamic filtering: 
  - `store_sales` is scanned in entirety.
  - The join discards most rows because they don’t match the dimension table keys.
- **With** dynamic filtering:
  - Trino collects the actual `d_date_sk` values from `date_dim` (the smaller side).
  - Prunes partitions in `store_sales` that do not match these date keys.

---

## 2. How Dynamic Filtering Works

1. **Identify** dimension table columns used in the join’s condition.
2. **Scan** the dimension table (build side), collecting a set or range of join keys.
3. **Push** these keys or ranges:
   - **Locally**, to the worker scanning the large table (probe side).  
   - **Globally**, to the coordinator to skip irrelevant splits at scheduling time (e.g., in the Hive connector).
4. **Probe** side scan is effectively restricted to partitions/rows that match the build side’s data.

### 2.1 Broadcast Join vs. Partitioned Join

- **Broadcast Join**: The smaller build side is replicated to all workers.
  - Dynamic filters are gathered before data distribution, so the worker receives them quickly.
  - The coordinator can prune splits as well.
- **Partitioned Join**: Both tables are partitioned by the join key.
  - The build side is read and hashed.  
  - The coordinator collects build side key ranges or sets and then pushes them to the probe side scans, or the local operator enforces the filter.  

### 2.2 Connectors

**Dynamic filter pushdown** depends on the connector’s ability to:
- Prune partitions, row-groups (ORC, Parquet stripes), or index-based skipping.
- For example, the **Hive connector** can skip entire partitions if the partition key is not in the dynamic filter.

---

## 3. Benefits

1. **Reduced Data Scanned**: Potentially skip entire partitions, file splits, or row groups.  
2. **Reduced Network Transfer**: Less data fetched from the data source.  
3. **Faster Queries**: Especially beneficial when dimension filters are highly selective.

---

## 4. Enabling/Disabling

**Enabled** by default. To disable:

- **Config**: `enable-dynamic-filtering=false` in `config.properties`
- **Session**: `SET SESSION enable_dynamic_filtering = false`

---

## 5. EXPLAIN Plan Insights

Run:

```sql
EXPLAIN
SELECT count(*)
FROM store_sales
JOIN date_dim ON store_sales.ss_sold_date_sk = date_dim.d_date_sk
WHERE d_following_holiday='Y' AND d_year = 2000;
```

A typical plan snippet might show:

```
InnerJoin[("ss_sold_date_sk" = "d_date_sk")][$hashvalue, $hashvalue_4]
...
 dynamicFilterAssignments = { d_date_sk -> #df_370 }
...
ScanFilterProject[ table = hive:default:store_sales, dynamicFilters = {"ss_sold_date_sk" = #df_370} ]
```

- `df_370` is a **dynamic filter** ID referencing `d_date_sk` from the build side.
- The `ScanFilterProject` on `store_sales` is told “only read rows that match #df_370”.

---

## 6. Runtime Stats (QueryInfo JSON)

During execution, Trino logs dynamic filter usage in the `QueryInfo` JSON or the Web UI. For example:

```json
"dynamicFiltersStats": {
  "dynamicFilterDomainStats" : [ {
    "dynamicFilterId" : "df_370",
    "simplifiedDomain" : "[ SortedRangeSet[type=bigint, ...]]",
    "collectionDuration" : "2.34s"
  } ],
  "lazyDynamicFilters" : 1,
  "replicatedDynamicFilters" : 1,
  "totalDynamicFilters" : 1,
  "dynamicFiltersCompleted" : 1
}
```

**Key fields**:
- `dynamicFilterDomainStats` → The final domain (set or range) used for filtering.
- `collectionDuration` → Time spent gathering build side data.
- `dynamicFiltersCompleted` → Number of dynamic filters that have finished collecting build side data.

Additionally, each **ScanFilterAndProjectOperator** can report:
```
"dynamicFilterSplitsProcessed" : 1
```
meaning at least one table split was processed after dynamic filters arrived.

---

## 7. Large vs. Small Build Sides

**Dynamic filtering** is especially potent when the build side is *much smaller* than the probe side. The cost-based optimizer (CBO) attempts to place the smaller table on the build side automatically—provided **table statistics** are up to date.

### 7.1 Thresholds

- If the build side is huge, collecting distinct values can be expensive.  
- Trino uses property-based **thresholds** to limit overhead (number of distinct keys, size in bytes, row limit, etc.).
- **Small** dynamic filters → fully collect distinct build side values.  
- **Large** dynamic filters → might fallback to **min/max** range-based filtering if the distinct set is too large.

**Properties**:

| Property                                                   | Description                                                                     |
|------------------------------------------------------------|---------------------------------------------------------------------------------|
| `enable-large-dynamic-filters`                             | Enables collecting dynamic filters even for large build sides.                  |
| `dynamic-filtering.large.*` and `dynamic-filtering.small.*`| Sets max distinct values, max size, row limit, etc. for broadcast join scenarios.  |
| `dynamic-filtering.large-partitioned.*` and `dynamic-filtering.small-partitioned.*` | For partitioned join scenarios.                  |

---

## 8. Dimension Table Layout Considerations

- Dynamic filtering works best when dimension keys correlate to build table columns.  
- A date dimension might have `d_date_sk` as a monotonic range, making range-based filtering effective.  
- Text-based dimension keys (e.g., `country-state-zip`) also benefit if the dimension key is frequently used for joining and can be pruned with either distinct sets or min/max ranges.

---
## 9. Limitations

1. **DOUBLE/REAL** types:
   - No min-max dynamic filter collection.
   - No `IS NOT DISTINCT FROM` dynamic filter.
2. **Data Type Casting**:
   - Build side key might be double, probe side integer, partial support for implicit casts.  
   - Some cast scenarios are not supported.
3. **Large Single Partition** in a window function or an unpartitioned scenario might limit dynamic filtering’s effect.
---
## 10. Example Visualization

```mermaid
flowchart LR
    A[Dimension Table: date_dim] -- Build side \n (small) --> B[Collect join keys\n e.g. {2451546..2451905}]
    A -->|Predicate: d_following_holiday='Y'| A
    B -->|Send dynamic filter\n df_370 to coordinator| C[Coordinator]
    C -->|Prune splits matching df_370| D[store_sales splits]
    D -->|Scan only relevant splits| W[Workers]
    W -->|Perform join with dimension| E[Final result]
```

1. The dimension side is read and filtered (`d_year=2000` etc.).
2. The key column values are collected into a dynamic filter `df_370`.
3. The coordinator prunes `store_sales` partitions that don’t match `df_370`.
4. Only relevant splits are sent to the workers for the big fact table scan.
---
## 11. Conclusion

**Dynamic filtering** in Trino is a powerful optimization for selective joins:

- Eliminates unnecessary reading of large fact tables.  
- Pushes runtime filters into connectors for partition pruning, row-group skipping, or index usage.  
- By default, dynamic filtering is on; it automatically applies if the optimizer chooses a smaller table on the build side.  
- For truly large build sides, property-based thresholds safeguard performance overhead, possibly switching to min-max range filters.

**Takeaways**:
- Keep dimension statistics updated; let the cost-based optimizer choose correct build side.  
- Confirm dynamic filtering is happening via **EXPLAIN** and `dynamicFiltersStats`.  
- Monitor overhead for large build side filters; tune via “large” or “small” dynamic filter config properties.

This optimization can dramatically reduce data scanning and speed up queries involving selective joins.