https://juejin.cn/post/7293121832169455653

---
### **1. Concept of Distributed Locks**
In a single-machine deployment system, traditional lock strategies (like Java synchronization) work well. However, in a **distributed system**, where multiple nodes operate concurrently, a different locking mechanism is required to ensure **mutual exclusion** across different JVMs and machines.

A **distributed lock** ensures that only one process or thread across multiple machines can hold the lock at any given time, providing safe access to shared resources.

### **Distributed Lock Solutions**
- **Database-based locks**: Reliable but slow due to the transactional overhead.
- **Redis-based locks**: High performance but can lead to potential reliability issues.
- **Zookeeper-based locks**: High reliability but with lower performance.

Redis is widely used due to its **high performance**.

---

### **2. Implementing Distributed Lock Using `SETNX`**

Redis provides the `SETNX` command (set if not exists) to implement distributed locks.

#### **Basic Command for Acquiring Lock**
```bash
SET key value NX PX <expiration_time>
```
- **NX**: Ensures the key is only set if it does not already exist.
- **PX <milliseconds>**: Sets an expiration time for the lock in milliseconds.

#### **Java Example of Distributed Lock with Redis**
```java
String lockKey = "lock:resource";
String lockValue = UUID.randomUUID().toString();
Boolean acquired = redisTemplate.opsForValue().setIfAbsent(lockKey, lockValue, 10, TimeUnit.SECONDS);

if (acquired) {
    try {
        // Execute business logic
    } finally {
        // Ensure lock is released
        redisTemplate.delete(lockKey);
    }
} else {
    // Retry acquiring lock
    Thread.sleep(100);
    acquireLock();
}
```

#### **Potential Issue:**
There’s a race condition between acquiring the lock (`SETNX`) and setting its expiration (`EXPIRE`). If an exception occurs between these operations, the lock remains indefinitely.

#### **Solution:**
Use `SET` with `NX` and `PX` in a **single atomic operation**.

```bash
SET key value NX PX 10000
```

---

### **3. Optimizing Distributed Lock Expiration**

#### **Setting Expiration Time in Java**
```java
String lockKey = "lock:resource";
String lockValue = UUID.randomUUID().toString();
Boolean lockAcquired = redisTemplate.opsForValue().setIfAbsent(lockKey, lockValue, 10, TimeUnit.SECONDS);

if (lockAcquired) {
    try {
        // Execute business logic
    } finally {
        // Ensure lock is released by the same client
        if (lockValue.equals(redisTemplate.opsForValue().get(lockKey))) {
            redisTemplate.delete(lockKey);
        }
    }
} else {
    // Retry logic or return failure
}
```

#### **Key Problems in Expiration Handling:**
- If the lock expires before the business logic completes, another client may acquire the lock prematurely.
- Manually releasing the lock after the expiration can result in unintended consequences (e.g., releasing another client's lock).

---

### **4. UUID-based Locking to Prevent Accidental Deletion**

#### **Unique Lock Value with UUID**
Each lock should have a **unique identifier (UUID)** associated with it. This ensures that only the client that acquired the lock can release it.

```java
String lockValue = UUID.randomUUID().toString();
String lockKey = "lock:resource";

Boolean lockAcquired = redisTemplate.opsForValue().setIfAbsent(lockKey, lockValue, 10, TimeUnit.SECONDS);

if (lockAcquired) {
    try {
        // Execute business logic
    } finally {
        // Only release the lock if it's still held by the current client
        if (lockValue.equals(redisTemplate.opsForValue().get(lockKey))) {
            redisTemplate.delete(lockKey);
        }
    }
}
```

#### **Problem:**
Even though UUID comparison helps mitigate some issues, there is still a race condition between **UUID comparison** and **lock deletion**.

---

### **5. Using LUA Scripts to Ensure Atomicity**

#### **Problem:**  
After comparing UUIDs, the lock might expire and be acquired by another process. If the original client still tries to delete the lock, it may unintentionally delete the new owner’s lock.

#### **Solution:**
**LUA scripts** ensure atomicity by combining the comparison and deletion in a single operation.

```lua
-- LUA Script to compare UUID and delete the lock atomically
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
```

#### **Java Implementation Using LUA Script**
```java
String script = "if redis.call('get', KEYS[1]) == ARGV[1] then " +
                "return redis.call('del', KEYS[1]) " +
                "else return 0 end";

String lockKey = "lock:resource";
String lockValue = UUID.randomUUID().toString();

// Acquire lock with expiration
Boolean lockAcquired = redisTemplate.opsForValue().setIfAbsent(lockKey, lockValue, 10, TimeUnit.SECONDS);

if (lockAcquired) {
    try {
        // Execute business logic
    } finally {
        // Use LUA script to release lock safely
        DefaultRedisScript<Long> redisScript = new DefaultRedisScript<>();
        redisScript.setScriptText(script);
        redisScript.setResultType(Long.class);

        redisTemplate.execute(redisScript, Collections.singletonList(lockKey), lockValue);
    }
}
```

#### **Why Use LUA?**
- **Atomicity**: The script ensures both the comparison of the UUID and the deletion happen in one atomic operation.
- **Performance**: Reduces the number of round-trip operations to Redis.

---

### **6. Key Considerations for Distributed Locks**
To ensure distributed locks work correctly, they must satisfy these **four conditions**:

1. **Mutual Exclusion**: Only one client can acquire the lock at any given time.
2. **No Deadlocks**: The lock must eventually release itself, even if the client holding it crashes.
3. **Safe Unlocking**: Only the client that acquired the lock should release it.
4. **Atomic Operations**: Both acquiring and releasing the lock must be atomic.

#### **Mutex Example**
```bash
SET lock:key uuid NX PX 10000
```
- **Mutual Exclusion**: Achieved via `SETNX`.
- **No Deadlock**: Handled with the `PX` expiration.
- **Safe Unlocking**: Verified by comparing UUID before deletion.
- **Atomicity**: Ensured through the use of Lua scripts.

---

### **Conclusion**
Redis distributed locks provide a simple yet powerful solution for synchronizing access to resources across a distributed system. Using `SETNX` with expiration time, UUID-based identification, and Lua scripts ensures robustness and atomicity. 

By following these guidelines and considerations, you can implement highly reliable distributed locks that meet performance and safety requirements in your distributed systems.