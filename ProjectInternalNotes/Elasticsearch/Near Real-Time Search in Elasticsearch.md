### Understanding Near Real-Time Search
Elasticsearch enables **near real-time search**, meaning that when a document is stored in Elasticsearch, it is indexed and fully searchable within approximately 1 second. This capability is attributed to the underlying Lucene Java libraries, which introduced the concept of per-segment search.

### Lucene's Per-Segment Search
In Lucene terminology, a segment is akin to an inverted index. However, an index in Lucene represents a collection of segments, combined with a commit point. Post-commit, a fresh segment is appended to the commit point and the buffer is emptied.

### The Role of Filesystem Cache
Interfacing between Elasticsearch and the disk storage is the filesystem cache. Documents from the in-memory indexing buffer (as depicted in Figure 4) are first written to a new segment (Figure 5). This new segment is initially written to the filesystem cache (which is a cost-effective operation) before it gets flushed to the disk (which is costlier). Once a file enters the cache, it can be accessed and read just like any standard file.

### Refresh: Making Segments Searchable
Lucene permits new segments to be written and opened. As a result, documents within these segments become searchable without necessitating a full commit, making it a lightweight operation that can be performed frequently without performance degradation.

In the context of Elasticsearch, this process of writing and opening a new segment is known as a **refresh**. Refreshing renders all operations performed on an index since the last refresh available for search.

### Controlling Refreshes
There are several ways to control refreshes in Elasticsearch:

1. Wait for the refresh interval to elapse.
2. Set the `?refresh` option.
3. Use the Refresh API to explicitly execute a refresh (`POST _refresh`).

By default, Elasticsearch refreshes indices every second. However, this is applicable only to those indices that have received at least one search request within the last 30 seconds. This periodic refresh is why Elasticsearch is classified as having near real-time search capabilities - while document changes aren't immediately visible to search, they become visible within this predefined timeframe. 

### Figures
* **Figure 4** depicts a Lucene index with new documents in the in-memory buffer. ![[Screenshot 2023-06-14 at 4.09.04 PM.png]]
* **Figure 5** demonstrates that the buffer contents are written to a segment, which is searchable, but is not yet committed.![[Screenshot 2023-06-14 at 4.09.21 PM.png]]
### Summary
The near real-time search capabilities of Elasticsearch stem from the integration of the per-segment search concept from Lucene. These efficient search features are further supported by the usage of the filesystem cache and the refreshing process in Elasticsearch.