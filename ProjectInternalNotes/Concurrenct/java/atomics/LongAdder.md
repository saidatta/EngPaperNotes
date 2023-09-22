The `LongAdder` class in Java is a part of the `java.util.concurrent.atomic` package. It provides an alternative to `AtomicLong` and `AtomicInteger` for maintaining a running `long` total across multiple threads.

When you have multiple threads updating a common sum, if the update contention is low (meaning, there's a low probability that two threads will try to update the sum at the same time), the `AtomicLong` and `LongAdder` classes behave similarly.

But when the contention is high (many threads are trying to update the sum simultaneously), `LongAdder` has significantly better performance. It does this by internally maintaining a set of variables for the sum, rather than a single value. This set of variables can grow as needed to reduce contention. The downside of `LongAdder` is that it uses more memory space due to the potentially multiple variables.

One common use-case for `LongAdder` is when you're collecting some kind of statistics in a multi-threaded context. For example, you might have a `ConcurrentHashMap<String, LongAdder>` where each key is some event type and the value is a `LongAdder` that you increment each time the event occurs.

Here's a simple example of using `LongAdder`:

```java
import java.util.concurrent.atomic.LongAdder;

public class Counter {
    private final LongAdder count = new LongAdder();

    public void increment() {
        count.increment();
    }

    public long getCount() {
        return count.sum();
    }
}
```

In this example, multiple threads could call the `increment` method at the same time, but the `LongAdder` will handle this contention gracefully and efficiently. The `getCount` method returns the total count across all of the internal variables.