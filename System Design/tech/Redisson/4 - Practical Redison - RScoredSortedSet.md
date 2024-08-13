#### Overview
RScoredSortedSet is a Redisson Java object that encapsulates the sorted set functionality of Redis. Redis sorted sets can store a set of non-repeating elements, each associated with a floating-point score, and are sorted based on these scores. This data structure supports fast access to any part of the set.
#### Principle
Redis sorted sets use a skip list, which allows for efficient data access while keeping elements sorted. The average time complexity for search, insertion, and deletion operations is O(log N). Redisson provides methods to interact with these sorted sets, including adding elements, deleting elements, getting element rankings, and iterating over elements. It also supports asynchronous and reactive APIs.

---
### Practical Applications

#### Scenario 1: Ranking System

```java
@Autowired
private RedissonClient redissonClient;

public void addUserScore(String username, double score) {
    RScoredSortedSet<String> leaderboard = redissonClient.getScoredSortedSet("leaderboard");
    leaderboard.addScore(username, score);
}

public Collection<String> getTopUsers(int topN) {
    RScoredSortedSet<String> leaderboard = redissonClient.getScoredSortedSet("leaderboard");
    return leaderboard.entryRangeReversed(0, topN - 1).stream()
            .map(ScoredEntry::getValue)
            .collect(Collectors.toList());
}
```

**Usage Summary:** Add or update user scores and retrieve the top N users for leaderboards.

---
#### Scenario 2: Time Series Data

```java
@Autowired
private RedissonClient redissonClient;

public void recordEvent(String eventId, long timestamp) {
    RScoredSortedSet<String> timeSeries = redissonClient.getScoredSortedSet("events");
    timeSeries.add(timestamp, eventId);
}

public Collection<String> getRecentEvents(int count) {
    RScoredSortedSet<String> timeSeries = redissonClient.getScoredSortedSet("events");
    return timeSeries.entryRangeReversed(0, count - 1).stream()
            .map(ScoredEntry::getValue)
            .collect(Collectors.toList());
}
```

**Usage Summary:** Record events with timestamps and retrieve recent events.

---
#### Scenario 3: Priority Queue

```java
@Autowired
private RedissonClient redissonClient;

public void submitTask(String taskId, double priority) {
    RScoredSortedSet<String> priorityQueue = redissonClient.getScoredSortedSet("tasks");
    priorityQueue.add(priority, taskId);
}

public String fetchNextTask() {
    RScoredSortedSet<String> priorityQueue = redissonClient.getScoredSortedSet("tasks");
    return priorityQueue.pollFirst();
}
```

**Usage Summary:** Add tasks with priority and fetch the next highest priority task.

---
#### Scenario 4: Product Price Sorting

```java
@Autowired
private RedissonClient redissonClient;

public void addOrUpdateProduct(String productId, double price) {
    RScoredSortedSet<String> productPrices = redissonClient.getScoredSortedSet("productPrices");
    productPrices.add(price, productId);
}

public Collection<String> getCheapestProducts(int count) {
    RScoredSortedSet<String> productPrices = redissonClient.getScoredSortedSet("productPrices");
    return productPrices.entryRange(0, count - 1).stream()
                        .map(Map.Entry::getValue)
                        .collect(Collectors.toList());
}
```

**Usage Summary:** Add or update product prices and retrieve the cheapest products.

---

**Advantages:**
- **Performance:** Efficient for large datasets due to the use of skip lists.
- **Sorting and Range Queries:** Supports quick retrieval of elements within a certain score range, ideal for leaderboards and range queries.
- **Uniqueness:** Ensures unique elements, even in concurrent environments.
- **Distributed Environment:** Suitable for distributed systems, providing shared access across multiple clients.
- **Scalability:** Handles large amounts of data while maintaining high performance.

**Disadvantages:**
- **Memory Limitations:** Limited by the memory capacity of the server.
- **Persistence:** Persistence mechanisms may not be as robust as traditional databases.
- **Complexity:** May require manual handling of data associations and transactions for complex relational data.
- **Cost:** Memory-based storage can be more expensive than traditional disk-based databases.
- **Data Consistency:** Maintaining data consistency in a distributed environment may require additional strategies and configuration.
---
### Advanced Examples
#### Real-Time Leaderboard

```java
@Autowired
private RedissonClient redissonClient;

// Add or update user score
public void updateScore(String user, double score) {
    RScoredSortedSet<String> scoredSortedSet = redissonClient.getScoredSortedSet("userScores");
    scoredSortedSet.add(score, user);
}

// Get top 10 users and their scores
public Map<String, Double> getTop10Users() {
    RScoredSortedSet<String> scoredSortedSet = redissonClient.getScoredSortedSet("userScores");
    return scoredSortedSet.entryRangeReversed(0, 9).stream()
            .collect(Collectors.toMap(ScoredEntry::getValue, ScoredEntry::getScore));
}
```
**Usage Summary:** Manage real-time leaderboards for users based on scores.

---
#### Delay Queue

```java
@Autowired
private RedissonClient redissonClient;

// Submit delayed task
public void scheduleTask(String taskId, long delayInSeconds) {
    RScoredSortedSet<String> delayQueue = redissonClient.getScoredSortedSet("delayQueue");
    double score = System.currentTimeMillis() / 1000.0 + delayInSeconds;
    delayQueue.add(score, taskId);
}

// Process due tasks
public void processDueTasks() {
    RScoredSortedSet<String> delayQueue = redissonClient.getScoredSortedSet("delayQueue");
    long now = System.currentTimeMillis() / 1000;
    Collection<String> dueTasks = delayQueue.valueRange(0, true, now, true);
    for (String taskId : dueTasks) {
        // Process task
        processTask(taskId);
        // Remove from queue
        delayQueue.remove(taskId);
    }
}
```
**Usage Summary:** Implement delay queues by sorting elements based on execution time.

---
#### Geolocation Services
```java
@Autowired
private RedissonClient redissonClient;

// Add location with score (e.g., based on distance)
public void addLocation(String locationId, double score) {
    RScoredSortedSet<String> locations = redissonClient.getScoredSortedSet("locations");
    locations.add(score, locationId);
}

// Get nearby locations
public Collection<String> getNearbyLocations(double minScore, double maxScore) {
    RScoredSortedSet<String> locations = redissonClient.getScoredSortedSet("locations");
    return locations.valueRange(minScore, true, maxScore, true);
}
```

**Usage Summary:** Display surrounding points of interest or business rankings based on geographic location.

---
#### Current Limiter
```java
@Autowired
private RedissonClient redissonClient;

// Try to acquire access
public boolean tryAcquire(String api, int maxCalls, long timePeriodSeconds) {
    String key = "rateLimiter:" + api;
    RScoredSortedSet<Long> scoredSortedSet = redissonClient.getScoredSortedSet(key);
    long now = System.currentTimeMillis();
    long clearBefore = now - timePeriodSeconds * 1000;
    
    // Remove old entries
    scoredSortedSet.removeRangeByScore(0, true, clearBefore, true);
    
    // Check if current calls exceed limit
    if (scoredSortedSet.size() < maxCalls) {
        // Record current call
        scoredSortedSet.add(now, now);
        return true;
    }
    return false;
}
```

**Usage Summary:** Implement rate limiting for API requests.

---
### Real-time data analysis and monitoring

In the financial or Internet of Things (IoT) fields, real-time monitoring and analysis of data streams is very important. `RScoredSortedSet`It can be used to store time series data and calculate statistics within sliding windows in real time.

``` java
`@Autowired private RedissonClient redissonClient; // 记录交易数据 

public void recordTransaction(String transactionId, double amount, long timestamp) {     
	RScoredSortedSet<Transaction> transactions = redissonClient.getScoredSortedSet("transactions");     
	transactions.add(new Transaction(transactionId, amount, timestamp), timestamp); 
} // 获取最近一分钟内的平均交易额 

public double getAverageTransactionAmountLastMinute() {     
	long oneMinuteAgo = System.currentTimeMillis() - 60000;     
	RScoredSortedSet<Transaction> transactions = redissonClient.getScoredSortedSet("transactions");     
	Collection<Transaction> recentTransactions = transactions.valueRange(oneMinuteAgo, true, Double.POSITIVE_INFINITY, true);
	return recentTransactions.stream()             
	.mapToDouble(Transaction::getAmount)
	.average() 
	.orElse(0.0); 
}`
```

In this example, we use `RScoredSortedSet`to store transaction data, and use the score (here is the timestamp) to get the transactions in the last minute, and then calculate the average transaction amount.

### Cryptocurrency trading platform order book

Cryptocurrency trading platforms need to manage a large number of buy and sell orders, `RScoredSortedSet`which can be used as an order book, where buy and sell orders are sorted by price.

```java
@Autowired private RedissonClient redissonClient; // 提交买单 

public void submitBuyOrder(String orderId, double price) {     
	RScoredSortedSet<Order> buyOrders = redissonClient.getScoredSortedSet("buyOrders");     
	buyOrders.add(price, new Order(orderId, price)); 
	} // 提交卖单 
	
public void submitSellOrder(String orderId, double price) {
	RScoredSortedSet<Order> sellOrders = redissonClient.getScoredSortedSet("sellOrders");     
	sellOrders.add(price, new Order(orderId, price)); 
	} // 匹配订单 
	
public void matchOrders() {     
	RScoredSortedSet<Order> buyOrders = redissonClient.getScoredSortedSet("buyOrders"); 
	RScoredSortedSet<Order> sellOrders = redissonClient.getScoredSortedSet("sellOrders");  
	Order highestBuyOrder = buyOrders.first();     
	Order lowestSellOrder = sellOrders.first();          
	while (highestBuyOrder != null && lowestSellOrder != null && highestBuyOrder.getPrice() >= lowestSellOrder.getPrice()) {         // 执行交易逻辑         
	executeTrade(highestBuyOrder, lowestSellOrder);         // 移除已匹配的订单         
	buyOrders.remove(highestBuyOrder);        
	sellOrders.remove(lowestSellOrder);         // 更新订单         
	highestBuyOrder = buyOrders.first(); 
	lowestSellOrder = sellOrders.first(); 
	} 
}

```

In this example, we use two `RScoredSortedSet`, one for buy orders and one for sell orders. By comparing the highest bid price and the lowest ask price, we can match the orders and execute the trade.

### Dynamic timeline in social network

In social networks, users' timelines usually need to be sorted according to the time when the updates were published.

```java
`@Autowired private RedissonClient redissonClient; // 发布动态 
public void postStatus(String userId, String statusId, long timestamp) {     
	RScoredSortedSet<String> timeline = redissonClient.getScoredSortedSet("timeline:" + userId);    
	timeline.add(timestamp, statusId); 
} // 获取用户的时间线 

public Collection<String> getUserTimeline(String userId, int page, int pageSize) {     
	RScoredSortedSet<String> timeline = redissonClient.getScoredSortedSet("timeline:" + userId); 
	int startIndex = page * pageSize; 
	int endIndex = (page + 1) * pageSize - 1;     
	return timeline.entryRangeReversed(startIndex, endIndex).stream()           .map(Map.Entry::getValue)
		.collect(Collectors.toList()); 
}
```

In this example, we create one for each user `RScoredSortedSet`to store their posts. When we need to get the user's timeline, we can reverse query the posts based on the timestamp score.

---
### Summary
RScoredSortedSet in Redisson provides a versatile way to manage ordered sets in Redis, supporting various scenarios such as real-time leaderboards, delay queues, geolocation services, and rate limiting. It ensures efficient data access, sorting, and range queries, making it a valuable tool for building scalable and performant systems. By leveraging RScoredSortedSet, developers can implement complex data processing solutions and handle large datasets effectively.