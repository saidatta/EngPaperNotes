https://juejin.cn/post/7359896145987551267
#### Overview
In recent work, Redis was used to manage a collection of elements with expiration times. The original implementation had several inefficiencies, prompting a review and redesign. This note explores realistic implementations using Redis, focusing on improving the approach with RSetCache and RMapCache.

#### Realistic Implementation: RScoredSortedSet
```java
public Set<String> get(String s) {
    String redisKey = String.format("aa:bb:%s", s);
    RScoredSortedSet<String> scoredSortedSet = redissonClient.getScoredSortedSet(redisKey);
    Set<String> ss = Sets.newHashSet();
    int sendBackedTimeout = SysConfigUtils.getIntegerValue("aa_bb_timeout");
    long subTime = System.currentTimeMillis() - sendBackedTimeout * 1000 * 60L;
    scoredSortedSet.forEach(key -> {
        double score = scoredSortedSet.getScore(key);
        if (score > subTime) {
            ss.add(key);
        }
    });
    return ss;
}
```

**Summary:**
The code retrieves valid elements within a specified timeout from a Redis sorted set, which can be useful for managing temporary data or event participants.
#### Improved Implementation: RSetCache
To leverage RSetCache for similar functionality, set the expiration time of elements upon insertion.

```java
public Set<String> get(String s) {
    String redisKey = String.format("aa:bb:%s", s);
    RSetCache<String> setCache = redissonClient.getSetCache(redisKey);
    Set<String> ss = Sets.newHashSet();
    int sendBackedTimeout = SysConfigUtils.getIntegerValue("aa_bb_timeout");
    long subTime = System.currentTimeMillis() - sendBackedTimeout * 1000 * 60L;
    setCache.forEach(key -> {
        long ttl = setCache.remainTimeToLive(key);
        if (ttl > 0 && System.currentTimeMillis() + ttl > subTime) {
            ss.add(key);
        }
    });
    return ss;
}
```

**Explanation:**
- Utilizes `RSetCache` to manage element expiration directly.
- Filters elements based on their remaining TTL and current time.

**RMapCache Example:**

```java
import org.redisson.api.RMapCache;
import org.redisson.api.RedissonClient;
import java.util.concurrent.TimeUnit;

public class MapCacheExample {
    public static void main(String[] args) {
        RedissonClient redisson = Redisson.create();
        RMapCache<String, String> mapCache = redisson.getMapCache("anyMapCache");
        mapCache.put("key1", "value1", 60, TimeUnit.SECONDS);
        mapCache.put("key2", "value2", 120, TimeUnit.SECONDS);
        redisson.shutdown();
    }
}
```
**RSetCache Example:**
```java
import org.redisson.api.RSetCache;
import org.redisson.api.RedissonClient;
import java.util.concurrent.TimeUnit;

public class SetCacheExample {
    public static void main(String[] args) {
        RedissonClient redisson = Redisson.create();
        RSetCache<String> setCache = redisson.getSetCache("anySetCache");
        setCache.add("value1", 60, TimeUnit.SECONDS);
        setCache.add("value2", 120, TimeUnit.SECONDS);
        redisson.shutdown();
    }
}
```
#### Extended Use Cases
**1. User Login Token Cache (RMapCache):**
- In this example, whenever a user logs in, we generate a unique token and store it in in association with the user ID `RMapCache`. The token automatically expires after 30 minutes. We also add a listener to handle the token expiration event.
```java
import org.redisson.api.RMapCache;
import org.redisson.api.RedissonClient;
import org.redisson.api.map.event.EntryExpiredListener;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

public class TokenCacheExample {
    public static void main(String[] args) {
        RedissonClient redisson = Redisson.create();
        RMapCache<String, String> tokenCache = redisson.getMapCache("userTokens");
        String userId = "user123";
        String token = UUID.randomUUID().toString();
        tokenCache.put(token, userId, 30, TimeUnit.MINUTES);
        tokenCache.addListener((EntryExpiredListener<String, String>) event -> {
            System.out.println("Token expired: " + event.getKey());
        });
        redisson.shutdown();
    }
}
```

**2. Limited Time Promotion (RSetCache):**
- In this example, we store the IDs of users who participate in a limited-time promotion in `RSetCache`a collection and set the duration of the promotion for each user. When the promotion ends, the user ID is automatically removed from the collection. We also add a listener to handle user ID expiration events.
```java
import org.redisson.api.RSetCache;
import org.redisson.api.RedissonClient;
import java.util.concurrent.TimeUnit;

public class LimitedTimeOfferExample {
    public static void main(String[] args) {
        RedissonClient redisson = Redisson.create();
        RSetCache<String> offerUsers = redisson.getSetCache("limitedTimeOfferUsers");
        String userId = "user123";
        long offerDuration = 2;
        offerUsers.add(userId, offerDuration, TimeUnit.HOURS);
        offerUsers.addListener((EntryExpiredListener<String>) event -> {
            System.out.println("Offer expired for user: " + event.getValue());
        });
        redisson.shutdown();
    }
}
```

### Grouping and Summarizing for System Design Preparation

#### System Design Context

**1. **Time-Limited Data Management:**
   - **Problem:** Efficiently managing temporary data with expiration.
   - **Solution:** Use Redis structures (`RSetCache`, `RMapCache`) for automatic expiration.

**2. **Practical Application Scenarios:**
   - **User Session Management:**
     - Cache user tokens with expiration for secure session handling.
   - **Promotional Campaigns:**
     - Manage limited-time offers by automatically expiring user participation data.

**3. **Design Considerations:**
   - **Scalability:** Redis handles large volumes of data with high performance.
   - **Reliability:** Redis's expiration mechanisms ensure data consistency.
   - **Maintainability:** Using structures like `RMapCache` and `RSetCache` simplifies code for time-based data management.

### Conclusion

Incorporating advanced Redis features like `RSetCache` and `RMapCache` provides efficient solutions for managing time-sensitive data in various applications. These techniques enhance system scalability, reliability, and maintainability, crucial for staff-level software engineers designing robust, scalable systems.

| **System Design Problem**                 | **Details**                                                                                                                                                                                                                                                                                                             | Improved Implementation                                                                                                                                                                           | **Usage Summary**                                                                                                                                                                                                                    | **Improved Implementation Code**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Time-Limited Data Management**          | **Original Implementation:** Retrieves elements within a specified timeout from a Redis sorted set, useful for managing temporary data or event participants.                                                                                                                                                           | Uses RSetCache to manage element expiration directly, improving efficiency.<br>**Example:** Caching user tokens with expiration for secure session handling and managing limited-time promotions. | Using `RSetCache` and `RMapCache` efficiently handles temporary data with expiration times, ensuring only valid data is kept, which is critical for scenarios like user session management and promotional campaigns.                | ```java<br>public Set<String> get(String s) {<br> String redisKey = String.format("aa:bb:%s", s);<br> RSetCache<String> setCache = redissonClient.getSetCache(redisKey);<br> Set<String> ss = Sets.newHashSet();<br> int sendBackedTimeout = SysConfigUtils.getIntegerValue("aa_bb_timeout");<br> long subTime = System.currentTimeMillis() - sendBackedTimeout * 1000 * 60L;<br> setCache.forEach(key -> {<br> long ttl = setCache.remainTimeToLive(key);<br> if (ttl > 0 && System.currentTimeMillis() + ttl > subTime) {<br> ss.add(key);<br> }<br> });<br> return ss;<br>}```                                                                                                                                                              |
| **Practical Application Scenarios**       | **User Login Token Cache:** Store and automatically expire user login tokens after a set period, ensuring secure session management.<br>**Limited Time Promotion:** Manage user participation in limited-time offers by automatically expiring user data after the promotion ends.                                      |                                                                                                                                                                                                   | These applications demonstrate how to use `RMapCache` for session tokens and `RSetCache` for time-limited promotions, showcasing their capability to handle scenarios requiring automatic expiration and cleanup of data.            | **User Login Token Cache Code:** ```java<br>import org.redisson.api.RMapCache;<br>import org.redisson.api.RedissonClient;<br>import java.util.UUID;<br>import java.util.concurrent.TimeUnit;<br>public class TokenCacheExample {<br> public static void main(String[] args) {<br> RedissonClient redisson = Redisson.create();<br> RMapCache<String, String> tokenCache = redisson.getMapCache("userTokens");<br> String userId = "user123";<br> String token = UUID.randomUUID().toString();<br> tokenCache.put(token, userId, 30, TimeUnit.MINUTES);<br> tokenCache.addListener((EntryExpiredListener<String, String>) event -> {<br> System.out.println("Token expired: " + event.getKey());<br> });<br> redisson.shutdown();<br> }<br>}``` |
| **Design Considerations for Scalability** | **Scalability:** Redis structures handle large volumes of data with high performance.<br>**Reliability:** Redis’s expiration mechanisms ensure data consistency.<br>**Maintainability:** Using `RMapCache` and `RSetCache` simplifies code for managing time-sensitive data, reducing the need for manual data cleanup. |                                                                                                                                                                                                   | Redis’s advanced features provide scalable, reliable, and maintainable solutions for managing time-sensitive data, crucial for building robust systems capable of handling high loads and ensuring data consistency and reliability. | **Limited Time Promotion Code:** ```java<br>import org.redisson.api.RSetCache;<br>import org.redisson.api.RedissonClient;<br>import java.util.concurrent.TimeUnit;<br>public class LimitedTimeOfferExample {<br> public static void main(String[] args) {<br> RedissonClient redisson = Redisson.create();<br> RSetCache<String> offerUsers = redisson.getSetCache("limitedTimeOfferUsers");<br> String userId = "user123";<br> long offerDuration = 2; <br> offerUsers.add(userId, offerDuration, TimeUnit.HOURS);<br> offerUsers.addListener((EntryExpiredListener<String>) event -> {<br> System.out.println("Offer expired for user: " + event.getValue());<br> });<br> redisson.shutdown();<br> }<br>}```                                 |

### Summary
Advanced Redis features like `RSetCache` and `RMapCache` are essential for efficiently managing time-sensitive data in various applications. They enhance system scalability, reliability, and maintainability, making them ideal for handling scenarios like secure user session management and limited-time promotions. By leveraging these tools, developers can build robust, scalable systems capable of automatic data expiration and cleanup, reducing manual intervention and improving overall system performance.