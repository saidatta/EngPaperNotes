https://juejin.cn/post/6844903813892014087

## 1. Overall Structure

- Uses a Node<K,V>[] array instead of Segment[] from Java 7
- Each bucket can be either:
  - A single node
  - A linked list of nodes
  - A red-black tree (if list becomes too long)

## 2. Key Components

### 2.1 Node<K,V> Class
```java
static class Node<K,V> implements Map.Entry<K,V> {
    final int hash;
    final K key;
    volatile V val;
    volatile Node<K,V> next;
}
```
- `volatile` fields ensure visibility across threads

### 2.2 table Field
```java
transient volatile Node<K,V>[] table;
```
- Main hash table, lazily initialized
- `volatile` ensures visibility of changes
### 2.3 sizeCtl Variable
- Controls initialization and resizing
- Negative values have special meanings:
  - -1: initialization in progress
  - -(1 + number of resizing threads): resizing in progress
![[Screenshot 2024-06-30 at 1.28.40 PM.png]]
## 3. Thread-Safe Operations
### 3.1 Initialization
```java
private final Node<K,V>[] initTable() {
    Node<K,V>[] tab; int sc;
    while ((tab = table) == null || tab.length == 0) {
        if ((sc = sizeCtl) < 0)
            Thread.yield();
        else if (U.compareAndSwapInt(this, SIZECTL, sc, -1)) {
            try {
                if ((tab = table) == null || tab.length == 0) {
                    int n = (sc > 0) ? sc : DEFAULT_CAPACITY;
                    @SuppressWarnings("unchecked")
                    Node<K,V>[] nt = (Node<K,V>[])new Node<?,?>[n];
                    table = tab = nt;
                    sc = n - (n >>> 2);
                }
            } finally {
                sizeCtl = sc;
            }
            break;
        }
    }
    return tab;
}
```
- Uses CAS to ensure only one thread initializes the table
- Double-check locking pattern
### 3.2 Put Operation
```java
final V putVal(K key, V value, boolean onlyIfAbsent) {
    if (key == null || value == null) throw new NullPointerException();
    int hash = spread(key.hashCode());
    int binCount = 0;
    for (Node<K,V>[] tab = table;;) {
        Node<K,V> f; int n, i, fh;
        if (tab == null || (n = tab.length) == 0)
            tab = initTable();
        else if ((f = tabAt(tab, i = (n - 1) & hash)) == null) {
            if (casTabAt(tab, i, null, new Node<K,V>(hash, key, value, null)))
                break;
        }
        else if ((fh = f.hash) == MOVED)
            tab = helpTransfer(tab, f);
        else {
            V oldVal = null;
            synchronized (f) {
                if (tabAt(tab, i) == f) {
                    if (fh >= 0) {
                        // Linked list logic
                    }
                    else if (f instanceof TreeBin) {
                        // Red-black tree logic
                    }
                }
            }
            // Check if conversion to tree is needed
        }
    }
    addCount(1L, binCount);
    return null;
}
```
- Uses CAS for adding to empty bucket
- Synchronizes on first node for adding to non-empty bucket
- Helps with transfer if resizing is in progress
### 3.3 Get Operation
```java
public V get(Object key) {
    Node<K,V>[] tab; Node<K,V> e, p; int n, eh; K ek;
    int h = spread(key.hashCode());
    if ((tab = table) != null && (n = tab.length) > 0 &&
        (e = tabAt(tab, (n - 1) & h)) != null) {
        if ((eh = e.hash) == h) {
            if ((ek = e.key) == key || (ek != null && key.equals(ek)))
                return e.val;
        }
        else if (eh < 0)
            return (p = e.find(h, key)) != null ? p.val : null;
        while ((e = e.next) != null) {
            if (e.hash == h &&
                ((ek = e.key) == key || (ek != null && key.equals(ek))))
                return e.val;
        }
    }
    return null;
}
```
- No locking needed due to `volatile` fields
### 3.4 Resizing
```java
private final void transfer(Node<K,V>[] tab, Node<K,V>[] nextTab) {
    // Complex resizing logic
}
```
- Multiple threads can help with resizing
- Uses `ForwardingNode` to mark buckets being processed
## 4. Key Thread-Safety Mechanisms
1. **Volatile Variables**: Ensure visibility of changes across threads
2. **CAS Operations**: For lock-free updates to shared state
3. **Synchronized Blocks**: Used sparingly for complex operations
4. **helping with resizing**: Allows multiple threads to contribute to resizing
5. **Immutable Key**: Ensures thread-safety for key objects

## 5. Comparison with Java 7 ConcurrentHashMap
- Removed Segment class, simplified structure
- Better scalability with finer-grained locking
- Improved performance for get operations (no locking)
- More efficient resizing with multiple threads helping
## 6. Potential Issues
- ABA problem in CAS operations
- Increased memory usage due to `volatile` fields
- Complex code, harder to maintain and understand
## 7. Best Practices
- Use when high concurrency and thread-safety are required
- Consider `Collections.synchronizedMap()` for simpler cases
- Be aware of slightly higher memory footprint compared to HashMap