
RateLimiters are important tools for controlling resource utilization in a system. By limiting the rate at which certain actions can occur, they can help prevent overuse of system resources and maintain a stable performance. This note will discuss how a RateLimiter is designed and how it functions.

## Design of RateLimiter
The primary feature of a RateLimiter is its stable rate, the maximum rate that it allows in normal conditions. This is enforced by "throttling" incoming requests as needed. The simplest way to maintain a rate of QPS (Queries Per Second) is to keep the timestamp of the last granted request, and ensure that `(1/QPS)` seconds have elapsed since then. 

### Example:
- For a rate of QPS=5 (5 tokens per second), if a request isn't granted earlier than 200ms after the last one, then we achieve the intended rate. If a request comes and the last request was granted only 100ms ago, then we wait for another 100ms. 
- At this rate, serving 15 fresh permits (i.e., for an `acquire(15)` request) naturally takes 3 seconds.

## Dealing with Past Underutilization
A RateLimiter only remembers the last request, and not any past underutilization. This can lead to either underutilization or overflow, depending on real-world consequences of not using the expected rate. To deal with this, we add a `storedPermits` variable. This variable is zero when there is no underutilization, and it can grow up to `maxStoredPermits`, for sufficiently large underutilization. 

### Example:
- For a RateLimiter that produces 1 token per second, every second that goes by with the RateLimiter being unused, we increase `storedPermits` by 1. Say we leave the RateLimiter unused for 10 seconds, thus `storedPermits` becomes 10.0 (assuming `maxStoredPermits >= 10.0`). 
- At that point, a request of `acquire(3)` arrives. We serve this request out of `storedPermits`, and reduce that to 7.0. Immediately after, assume that an `acquire(10)` request arriving. We serve the request partly from `storedPermits`, using all the remaining 7.0 permits, and the remaining 3.0, we serve them by fresh permits produced by the rate limiter.

## Dealing with Overflows
The RateLimiter doesn't remember the time of the last request but it remembers the expected time of the next request. This enables us to tell immediately whether a particular timeout is enough to get us to the point of the next scheduling time. When we observe that the expected arrival time of the next request is actually in the past, then the difference (now - past) is the amount of time that the RateLimiter was formally unused, and it is that amount of time which we translate to `storedPermits`.

### Example:
- Consider a RateLimiter with a rate of 1 permit per second, currently completely unused, and an expensive `acquire(100)` request comes. It would be nonsensical to just wait for 100 seconds, and then start the actual task. Instead, we allow starting the task immediately, and postpone by 100 seconds future requests.

## Throttling
The throttling function is responsible for determining how fast or slow the RateLimiter should operate depending on the amount of stored permits and their cost. This function translates storedPermits to throttling time.

### Example:
- If `storedPermits == 10.0`, and we want 3 permits,

.SUCCESS;

5. Store the remaining permits to be served by fresh ones (3.0).
```java
remainingPermits = requestPermits - storedPermits; // remainingPermits = 10 - 7 = 3
```

6. The fresh permits would be served at the stable rate of 1 token/second, so it will take 3 seconds.
```java
waitTimeForFreshPermits = remainingPermits / rate; // waitTimeForFreshPermits = 3 / 1 = 3 seconds
```

7. The stored permits are served by a call to storedPermitsToWaitTime, where we calculate the throttling time for these stored permits.
```java
waitTimeForStoredPermits = storedPermitsToWaitTime(storedPermits, permitsToTake); // this function would calculate the wait time
```

### The Concept of storedPermitsToWaitTime
This function provides a mechanism to determine how stored permits are translated into wait time. This is governed by a mathematical function that maps the stored permits (from 0.0 to maxStoredPermits) onto the interval `1/rate`. The specific function can be designed according to the system's needs and the behavior we desire in case of underutilization.

### Bursty vs Sustained Limiting (Capping `maxStoredPermits`)
If we have a system where we anticipate bursty traffic and we want to allow it to proceed as quickly as possible, we might design a function that goes below the horizontal line `(1/QPS)`. This will mean that the RateLimiter serves stored permits faster than fresh ones, thus taking advantage of previous underutilization. This is useful in situations like network bandwidth limiting where past underutilization translates to "almost empty buffers" which can be filled immediately.

On the contrary, in a system where we want to ensure sustained usage without allowing bursts to consume resources too quickly, we might design a function that goes above the horizontal line `(1/QPS)`. This means stored permits are served slower than fresh ones, making the RateLimiter slower after a period of underutilization. This is important when the real world consequence of not using the expected rate results in the system becoming less prepared for future requests, like when server caches become stale or requests become more likely to trigger expensive operations.

### Remembering the Next Request Rather Than the Last
In traditional RateLimiter designs, the last request timestamp is remembered. The given RateLimiter design differs in that it remembers the (expected) time of the next request. This allows it to start serving a large request immediately and postpones future requests, which is more efficient as it avoids idling. This design also makes it possible to quickly determine if a particular timeout is enough to get us to the point of the next scheduling time since we always maintain that.

### Graphical Representation of throttling and storedPermits
The throttling vs storedPermits can be represented by a graph where the x-axis represents `storedPermits` and the y-axis represents `throttling`.

For example, consider a RateLimiter with a `stableInterval` of 1 token per second, and a `coldFactor` of 3. This means the `coldInterval` is `coldFactor * stableInterval`, or 3 seconds. Let's also say the `warmupPeriod` is 5 seconds.

In such a case, the `thresholdPermits` will be `0.5 * warmupPeriod / stableInterval` = `0.5 * 5 / 1` = `2.5` permits. And the `maxPermits` will be `thresholdPermits + 2 * warmupPeriod / (stableInterval + coldInterval)`

