
https://www.redditinc.com/blog/view-counting-at-reddit/

---
## **Introduction**
- Reddit aimed to implement an efficient counting system for post views.
- Counting views at scale presents unique challenges, especially for popular posts.
---
## **Counting Methodology Requirements**:
1. **Real-time counts**: Can't depend on daily or hourly aggregates.
2. **Unique counts**: Users should be counted once within a short window.
3. **Accuracy**: Displayed count should closely reflect actual tally.
4. **Production-scale**: System must process events within seconds.
---
## **Why Not Exact Counts?**
- To get real-time exact counts, need to check if a specific user viewed the post before.
- Naive Solution: Store unique users in memory as a hash table (post ID as key).
- **Issue**: Difficult to scale for popular posts with massive unique views. 
---
## **Cardinality Estimation Algorithms**
1. **Linear Probabilistic Counting**: Accurate but needs more memory as set grows.
2. **HyperLogLog (HLL)**: Uses sub-linear memory but sacrifices some accuracy.
**Comparison**:
- Storing 1 million unique user IDs = 8MB (assuming 8-byte ID).
- Using HLL to count 1 million IDs = 12KB (massive reduction).
---
## **HLL Implementations**
- **Sparse HLL**: Linear counting for small sets.
- **Dense HLL**: Switches to HLL for larger sets.
- **Hybrid approach**: Combines both for balance in accuracy and memory use.

---
## **HLL Implementation Choices**
1. **Twitter’s Algebird (Scala)**: Good usage docs, but intricate implementation details.
2. **Stream-lib (Java)**: Well-documented but challenging to tune.
3. **Redis’s HLL**: Selected due to clear documentation, configurability, and performance benefits.
---
## **Reddit’s Data Pipeline**
- **Event Generation**: User views post → Event fired → Sent to event collector → Persisted in Kafka.
- **Counting System Components**:
  1. **Nazar**: A Kafka consumer, processes each event, decides if it's countable (protects against gaming).
  2. **Abacus**: Reads the events output by Nazar, counts views based on Nazar’s flag.
---
## **Event Flow**:
1. **Nazar’s Role**: 
    - Uses Redis for state maintenance.
    - Filters out repeat views.
    - Adds a Boolean flag (countable or not).
    - Sends the altered event back to Kafka.
![[Screenshot 2023-08-28 at 3.27.36 PM.png]]
2. **Abacus's Role**:
    - Reads events from Kafka.
    - Checks the HLL counter in Redis for the post.
    - Interacts with Cassandra for counter persistence.
    - Abacus first checks if there is an HLL counter already existing in Redis for the post corresponding to the event. If the counter is already in Redis, then Abacus makes a [PFADD](https://redis.io/commands/pfadd) request to Redis for that post. If the counter is not already in Redis, then Abacus makes a request to a Cassandra cluster, which we use to persist both the HLL counters and the raw count numbers, and makes a [SET](https://redis.io/commands/set) request into Redis to add the filter. This usually happens when people view older posts whose counters have been evicted from Redis.
1. **Data Storage**:
    - Abacus periodically saves HLL filters and counts to Cassandra.
    - Batches writes to avoid overloading Cassandra.
    - In order to allow for maintaining counts on older posts that might have been evicted from Redis, Abacus periodically writes out both the full HLL filter from Redis along with the count for each post to a Cassandra cluster. Writes to Cassandra are batched in 10-second groups per post in order to avoid overloading the cluster. Below is a diagram outlining this event flow at a high level.
---
## **Conclusion**:
- **Goal**: Better insight for content creators and moderators.
- **Future Plans**: Leverage real-time data for valuable feedback to Reddit users.
---
## **References**:
1. High Scalability article on counting algorithms.
2. Google's HyperLogLog++ paper.
---
![[Screenshot 2023-08-28 at 3.28.07 PM 1.png]]
