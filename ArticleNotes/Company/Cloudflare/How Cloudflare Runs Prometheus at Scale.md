https://blog.cloudflare.com/how-cloudflare-runs-prometheus-at-scale
#### Overview
Cloudflare employs Prometheus extensively to monitor its expansive network infrastructure, running 916 Prometheus instances managing around 4.9 billion time series. This chapter explores the complexities and strategies Cloudflare employs to maintain this large-scale Prometheus deployment efficiently.
#### Cardinality and Metrics
- **Cardinality Definition**: In Prometheus, cardinality represents the number of unique combinations of all labels in a metric. High cardinality can lead to memory and performance issues, known as "cardinality explosion."
- **Metric vs. Time Series**: A metric defines an observable property, while a time series is an instance of a metric with a unique combination of labels and a series of timestamp-value pairs.
- **Samples**: Samples are individual instances of metrics at specific timestamps. Prometheus handles the timestamping of these samples during the scraping process.
#### Prometheus Data Model
- **Scraping Process**: Prometheus collects metrics through HTTP requests, parsing responses to extract samples.
- **Time Series Database (TSDB)**: Prometheus uses TSDB to store time series, identifying unique series by hashing labels.
- **Memory Usage**: Time series are stored in memory initially, with older chunks written to disk and memory-mapped to reduce memory usage. Memory usage patterns follow the creation, update, and garbage collection of time series.
#### Challenges with High Cardinality

- High cardinality can significantly increase memory usage in Prometheus, risking server crashes if physical memory is exhausted.
- Prometheus provides several configuration options to mitigate high cardinality risks, including limits on sample count, label count, and label length.
- Cloudflare employs additional measures, including custom patches and CI validation, to manage cardinality and prevent overload.

#### Cloudflare's Prometheus Strategy

- **Custom Patches**: Cloudflare maintains custom patches on Prometheus, including a TSDB limit patch to cap the total number of time series and a modified handling of `sample_limit` for graceful degradation instead of failing scrapes.
- **Configuration Limits**: Default scrape configurations impose limits on labels and sample counts to catch accidental high cardinality scenarios.
- **CI Validation**: Cloudflare's CI checks ensure sufficient capacity on Prometheus servers for any configuration changes that increase time series counts.
- **Documentation and Tools**: Internal documentation and tools assist engineers in managing metrics lifecycle, from definition to visualization, helping avoid common pitfalls.

#### Key Takeaways

- **Understanding Prometheus Internals**: A deep understanding of Prometheus's behavior and limitations is crucial for managing high cardinality and ensuring efficient operation at scale.
- **Strategic Measures**: Cloudflare's approach combines Prometheus's built-in protections with custom enhancements and operational practices to manage its large-scale deployment effectively.
- **Flexibility and Safety**: The custom patches and configuration strategies provide a safety net against cardinality explosions while allowing flexibility in metrics collection and analysis.

#### Conclusion

Cloudflare's experience with running Prometheus at scale highlights the importance of a comprehensive strategy that includes understanding the underlying system, implementing safeguards, and enabling engineers to deploy metrics confidently. Through custom patches, strict configuration controls, and a focus on education, Cloudflare has created a resilient and scalable observability platform with Prometheus at its core.

---
#### Technical Insights
- **Prometheus at Cloudflare**: Utilizes 916 instances, handling about 4.9 billion time series. This vast deployment serves to monitor performance and health across Cloudflare's global network infrastructure.
#### Deep Dive into Cardinality and Prometheus's Operation
- **Cardinality Challenges**: High cardinality, resulting from a vast number of label combinations, poses significant memory and performance challenges. Cardinality explosions, where the number of time series spikes dramatically, can crash Prometheus instances due to memory exhaustion.
- **Metrics vs. Time Series vs. Samples**:
  - **Metrics**: Defined observable quantities in an application, structured with names and labels.
  - **Time Series**: Derived from metrics, each with a unique label combination and associated with timestamped values, representing how metrics change over time.
  - **Samples**: The actual data points collected during scrapes, timestamped by Prometheus, forming the time series.
#### Prometheus: Storage and Memory Management
- **TSDB Operations**:
  - Prometheus's Time Series Database (TSDB) manages time series data, identifying unique series through label hashing.
  - Memory usage is initially high due to in-memory storage of time series, with older chunks offloaded to disk through memory-mapping and written to disk blocks every two hours to optimize for space and memory.
- **Handling High Cardinality**:
  - Prometheus configurations allow limits on sample counts, label counts, and lengths to mitigate high cardinality risks.
  - Cloudflare implements additional safeguards, including custom patches to enforce a global limit on the number of time series and modify `sample_limit` behavior for individual scrapes.
#### Cloudflare's Prometheus Strategies
- **Custom Solutions**:
  - **TSDB Limit Patch**: A custom patch that caps the total number of time series stored, preventing memory overload by selectively skipping the creation of new time series beyond the limit.
  - **Sample Limit Modification**: Alters Prometheus's handling of the `sample_limit` parameter to ensure graceful degradation rather than complete scrape failure when limits are exceeded.

- **Operational Practices**:
  - **Default Configuration Limits**: Sets baseline limits on labels and sample counts to preemptively catch potential cardinality issues.
  - **CI Validation for Scalability**: Ensures new or modified Prometheus scrape configurations do not exceed the capacity of Cloudflare's Prometheus servers, based on memory and time series metrics.

- **Documentation and Engineering Support**:
  - Internal documentation and tooling guide engineers through best practices for metric collection, configuration, and analysis, promoting understanding and efficient use of Prometheus within Cloudflare's infrastructure.

#### Enhanced Conclusion

Cloudflare's approach to running Prometheus at scale is a blend of leveraging Prometheus's built-in capabilities, custom technical solutions, and strategic operational practices. The focus on managing cardinality, understanding Prometheus's memory usage patterns, and implementing safeguards against potential overloads ensures the resilience and scalability of their observability platform. Through meticulous planning, technical innovation, and empowering engineers with knowledge and tools, Cloudflare maintains an efficient and reliable monitoring system that underpins their global network infrastructure.