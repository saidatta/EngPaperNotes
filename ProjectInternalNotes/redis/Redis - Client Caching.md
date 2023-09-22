# Obsidian Note: Client-Side Caching in Redis

## Overview
Client-side caching is a feature implemented by Redis to manage and optimize the handling of hot data. It is used to avoid direct access to the database by caching data on the Redis client side. This improves data access performance, reduces delay in obtaining data, and alleviates the load on Redis.
## Benefits of Client-Side Caching
1. **Performance Improvement**: Accessing local memory is faster than accessing Redis through the network. This significantly reduces the delay in obtaining data.

2. **Load Reduction**: By caching frequently accessed data on the client side, the load on the Redis server is decreased, leading to better overall performance.
## Implementation of Client-Side Caching
Redis implements a server-assisted client-side cache through a feature called tracking. This feature has three modes: normal, broadcast, and forwarding.
### Normal Mode
When the tracking mode is turned on, Redis "remembers" each client's requested key. If the value of a tracked key changes, Redis sends an invalidation message to the client. The server-side stores the client's requested key and its corresponding client ID list in a globally unique table, called the Tracking Table. This table is implemented using a radix tree, a multi-fork search tree for sparse long integer data search.

- The server side stores the key accessed by the client and the client ID list information corresponding to the key in a globally unique table (TrackingTable). When the table is full, it will remove the oldest record and trigger a notification to the client that the record has expired end.
- Each Redis client has a unique digital ID. TrackingTable stores each Client ID. When the connection is disconnected, the record corresponding to the ID is cleared.
- The Key information recorded in the TrackingTable table does not consider which database it belongs to. Although the key of db1 is accessed, the client will receive an expiration prompt when the key with the same name in db2 is modified, but this will reduce the complexity of the system and the storage data of the **table** quantity.
### Broadcast Mode
In the broadcast mode, the server does not keep track of the keys that a client has accessed. Instead, it broadcasts the invalidation of all keys to all clients. This mode consumes no memory on the server side but can consume a large amount of network bandwidth if keys are frequently modified. To avoid this, you can set the client to track only the keys with a specified prefix.
### Forwarding Mode
This mode is used for clients that implement the RESP2 protocol, which does not directly support the PUSH invalidation messages. In this case, another client supporting the RESP3 protocol is needed to notify the server to send invalidation messages to the RESP2 client via Pub/Sub.
## Precautions for Client-Side Caching
1. **Choose the Right Keys for Caching**: Not all keys should be cached. Keys that frequently change or are rarely requested should not be cached. The ideal keys to cache are those that are frequently requested and have a reasonable rate of change.
2. **Use of Tracking Mode**: The tracking mode is not enabled by default in the client. It has to be manually turned on using the `CLIENT TRACKING ON|OFF` command.
3. **Use of Broadcast Mode**: Broadcast mode should be used with caution as it can consume a significant amount of network bandwidth if the keys are frequently modified.
## Examples
#### Normal Mode
```redis
CLIENT TRACKING ON
GET user:211
```
The client executes the read-only command and is tracked.
#### Broadcast Mode
```redis
CLIENT TRACKING ON BCAST PREFIX user
```
The client registers to only track the keys with the 'user' prefix.
#### Forwarding Mode
For Client B (ID: 606), using RESP2
```redis
SUBSCRIBE _redis_:invalidate
```
For Client A, using RESP3
```redis
CLIENT TRACKING ON BCAST REDIRECT 606
```
Client B gets the invalidation message through the `_redis_:invalidate` channel.
## Reference Implementation Code
Redis's tracking feature is implemented in the source code file `tracking.c` in the Redis GitHub repository. Here is the [link](https://github.com/antirez/redis/blob/unstable/src/tracking.c) to the source code.