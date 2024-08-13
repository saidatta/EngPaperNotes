https://juejin.cn/post/7278286913944272956?searchId=20231014055258E5D9CB3F548616708F5A
https://juejin.cn/post/7177670486583050296?searchId=20231014055258E5D9CB3F548616708F5A
https://juejin.cn/post/7083386824457977869?searchId=20231014055258E5D9CB3F548616708F5A
## Preface
In the previous article, we discussed the implementation and characteristics of `CopyOnWriteArrayList`, noting its limitations in scenarios with high write frequency or large concurrency. This article will delve into `ConcurrentLinkedQueue`, a high-performance queue implementation suitable for concurrent scenarios.

Before reading this article, familiarity with CAS (Compare-And-Swap) and the `volatile` keyword is essential.
## Data Structure
As the name suggests, `ConcurrentLinkedQueue` is a thread-safe, concurrent queue implemented using a linked list.
### Core Fields
- **head**: The head of the queue (first node).
- **tail**: The tail of the queue (last node).
Both `head` and `tail` are `volatile` to ensure visibility across threads without locking.
```java
public class ConcurrentLinkedQueue<E> extends AbstractQueue<E> implements Queue<E>, java.io.Serializable {
    private static class Node<E> {
        volatile E item;
        volatile Node<E> next;
    }
    private transient volatile Node<E> head;
    private transient volatile Node<E> tail;

    public ConcurrentLinkedQueue() {
        head = tail = new Node<E>(null);
    }
}
```
### Initialization
During initialization, both `head` and `tail` point to a node with null data.
```java
public ConcurrentLinkedQueue() {
    head = tail = new Node<E>(null);
}
```
## Design Thoughts
### Delayed Update of Head and Tail Nodes
`ConcurrentLinkedQueue` employs optimistic locking with CAS and failure retry to ensure atomicity of operations. To minimize CAS overhead, it uses delayed updates for the head and tail nodes, meaning these nodes may not always reflect the most current state of the queue.
### Sentinel Nodes
Sentinel nodes, or virtual nodes, are used to simplify the code and reduce concurrency conflicts. These nodes help manage edge cases, such as when the queue has only one node.
## Source Code Implementation
### `offer` Method
The `offer` method adds elements to the queue. It uses a loop to find the actual tail node and attempts to add a new node using CAS.
#### Key Variables
- **t**: Records the tail node.
- **p**: Used for iterating nodes to find the real tail node.
- **q**: Records the successor node of p.
#### Code
```java
public boolean offer(E e) {
    checkNotNull(e);
    final Node<E> newNode = new Node<>(e);
    for (Node<E> t = tail, p = t;;) {
        Node<E> q = p.next;
        if (q == null) {
            if (p.casNext(null, newNode)) {
                if (p != t) casTail(t, newNode);
                return true;
            }
        } else if (p == q) {
            p = (t != (t = tail)) ? t : head;
        } else {
            p = (p != t && t != (t = tail)) ? t : q;
        }
    }
}
```
### `poll` Method
The `poll` method removes elements from the queue. It uses a loop to find the actual head node and attempts to remove it using CAS.
#### Key Variables
- **h**: Records the head node.
- **p**: Used for iterating nodes to find the real head node.
- **q**: Records the successor node of p.
#### Code
```java
public E poll() {
    restartFromHead:
    for (;;) {
        for (Node<E> h = head, p = h, q;;) {
            E item = p.item;
            if (item != null && p.casItem(item, null)) {
                if (p != h) updateHead(h, (q = p.next) != null ? q : p);
                return item;
            } else if ((q = p.next) == null) {
                updateHead(h, p);
                return null;
            } else if (p == q) {
                continue restartFromHead;
            } else {
                p = q;
            }
        }
    }
}
```
### Updating Head and Tail Nodes
The `updateHead` method updates the head node and converts the old head node to a sentinel node.
```java
final void updateHead(Node<E> h, Node<E> p) {
    if (h != p && casHead(h, p)) {
        h.lazySetNext(h);
    }
}
```
## Flowchart Implementation
For better understanding, let's use a simplified example:
```java
public void testConcurrentLinkedQueue() {
    ConcurrentLinkedQueue<String> queue = new ConcurrentLinkedQueue<>();
    queue.offer("h1");
    queue.offer("h2");
    queue.offer("h3");

    String p1 = queue.poll();
    String p2 = queue.poll();
    String p3 = queue.poll();
    String p4 = queue.poll();

    queue.offer("h4");
    System.out.println(queue);
}
```
### Visualization
#### Initial State
```
[head] -> [null] -> [tail]
```
#### After First `offer("h1")`
```
[head] -> [null] -> [h1] -> [tail]
```
#### After Second `offer("h2")`
```
[head] -> [null] -> [h1] -> [h2] -> [tail]
```
#### After Third `offer("h3")`
```
[head] -> [null] -> [h1] -> [h2] -> [h3] -> [tail]
```
#### After First `poll()`
```
[head] -> [null] -> [null] -> [h2] -> [h3] -> [tail]
```
#### After Second `poll()`
```
[head] -> [null] -> [null] -> [null] -> [h3] -> [tail]
```
#### After Third `poll()`
```
[head] -> [null] -> [null] -> [null] -> [null] -> [tail]
```
#### After Fourth `poll()`
Queue is empty.
#### After `offer("h4")`
```
[head] -> [null] -> [h4] -> [tail]
```
## Summary
`ConcurrentLinkedQueue` is a high-performance concurrent queue implemented using a singly linked list. It leverages `volatile` for visibility and CAS for atomic operations. Key features include:

- **Delayed Update**: Reduces CAS overhead by delaying updates to head and tail nodes.
- **Sentinel Nodes**: Simplifies code and reduces concurrency conflicts.
- **Optimistic Locking**: Ensures atomicity without blocking.

This makes `ConcurrentLinkedQueue` highly suitable for scenarios with high concurrency, frequent reads and writes, and operations at the head and tail of the queue.