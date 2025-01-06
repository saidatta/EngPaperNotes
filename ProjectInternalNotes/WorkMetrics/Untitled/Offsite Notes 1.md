**Asked questions**
- Human is involved when alerting.
	- Cost effective, scalable >> realtime; performant.
- Grafana, chronosphere cost = 0.33 * our cost; Competitive advantage they have.

**My Questions**
- How is Prometheus, Alertmanager behind technically?
	- Prometheus uses head-linked list chunks which is different different how TSDB stores it. does it matter?
- What are archived metrics? historicals?
- What does **Repartitioner** do?
- Whats worng with having a buffer pool in the last 1 hr in the new metric data architecture.
- How to use Arrow for in-memory tier, since we are using Parquet for Disk.
- consider the lag of the batch quantization
- What are auto-complete queries? meatballs-light?
- What does quantize on-demand? is it quantizing on real time queries?

**Notes**
**Customer churn**
- lack of unified gdi
- lack of tightly coupling with Splunk core product.
- RBAC, unified identity, team mgmt?

**cost improvement**
per metric cost : 50 cents(current) -> 11 cents (Dec 25).
- 11 cents is the current competitive customer market cost.

**Competitor Notes**
- Grafana UI >> Our product visualization
- Grafana, chronosphere tracing << splunk.
- Customers dont want to maintain which metrics are useful as its decentralized to a variety of their own teams. They just want to hit our APIs.

**List of competitors we lost deals to**
1. Grafana
2. Chronosphere
3. DataDog
4. Cost >> we have data shaping capabilities.

Q - joins here are analytics joins? dont believe TSDB have joins.
![[Screenshot 2024-11-18 at 11.46.06 AM.png]]

The 60% cost savings will keep to ourselves unless market pressure from competitors.
![[Screenshot 2024-11-18 at 3.36.46 PM.png]]
- S3 ingest data - WAL

---------------------
**New metric info**
**API**
Metricname = jvm.cpu.load; clustername = sfx_realm:us0
org_id = <Org>
Time range = 30-now
Resolution = 15m.

**Benchmarks for Agamemnon**
 - peak - 36k QPS
- Avg - ~20k QPS

**TSDB benchmarks**
- 2000 QPS

Query benchmark
150k * 30 = 4.5million from S3. - 1min time 

if org & metric - 
Last 1 hour data could take along of time, since data will be in S3 and for lower granularities it will have exponential response times.

**Storage estimates.**
50 bytes per data point on disk.

new version of in-memory tier is partitioned by wall clock timestamp, but UI is showed based on logical timestamp.


**Uday**
Avro (queries) vs Parquet - repartitioner
- Parquet is too slow to encode 

Analytics->stream quantizer

**Terminologies**
Metric source service (MSS) = view server


Technologies discussed
- Trino
- Spark
- DuckDB - locally near process query.

With new architecture - Hot partitions will be more often since its partitioned by org_id, metric-name instead of mts_id.


with new architecture, batch quantizing everything - will be cost-effective.