## Background

- Airbnb is migrating to a Service Oriented Architecture (SOA)
- SOA poses challenges for payments apps in maintaining data integrity across distributed services
- Distributed transactions are hard to make consistent without protocols like two-phase commit
## Approaches for Eventual Consistency
1. **Read Repair**: Clients read data and repair any inconsistencies
2. **Write Repair**: Every write call from client attempts to repair broken state (used by Orpheus)
3. **Asynchronous Repair**: Server runs data consistency checks asynchronously

Airbnb uses a write repair approach with the Orpheus idempotency library.
## What is Idempotency?
- An idempotent operation can be applied multiple times without changing the result after the first application
- Crucial for financial operations like payments to avoid double charges/payments
- Enables safe retries from clients to achieve eventual consistency
## Problem Statement
- Need a generic, configurable idempotency solution across Airbnb's payments services
- Can't compromise on data consistency impacting users
- Low latency required, so not a separate service
- Shield developers from handling data integrity complexities 
- Maintain code readability, testability, debuggability
## Solution: Orpheus Idempotency Library

- Based on the Greek mythological hero who could charm all living things
- Uses an idempotency key to uniquely identify each request
- Stores idempotency data in sharded master databases for consistency

### Request Lifecycle

1. **Pre-RPC**: Record request details in database
2. **RPC**: Make external service call over network 
3. **Post-RPC**: Record response details in database

```ad-warning
No network calls allowed in Pre/Post-RPC phases, no database calls in RPC phase to ensure isolation
```

### Java Lambdas

- Uses Java lambda expressions to combine database commits in Pre/Post-RPC phases
- Improves code readability and maintainability

```java
preRpcPhase = () -> {
    insertRequestDetails(...);
    return null;
};

orpheus.executeIdempotent(key, preRpcPhase, 
    this::doRpc, 
    postRpcPhase);
```

### Exception Handling

- Exceptions classified as retryable (network errors) or non-retryable (validation errors)
- Implemented via custom exception classes
- Crucial to avoid infinite retries or double payments

### Client Responsibilities  

- Pass unique idempotency keys, reuse for retries
- Persist keys to retry from in case of failures
- Consume successful responses and unassign keys
- No mutation of requests between retries
- Configure retry policies with backoff/jitter

### Idempotency Keys

- **Request-level**: Random unique key for each request
- **Entity-level**: Deterministic key based on entity to ensure idempotency scoped to that entity

### Request Leases
- Acquires a row-level database lock on the idempotency key
- Lease expiration higher than RPC timeout
- Prevents race conditions and double processing
### Recording Responses
- Responses persisted to indicate final state (success or non-retryable failure)
- Allows quick response for retries but can lead to bloated tables
- Don't make backwards-incompatible response changes
### Master Database Only
- Read/write idempotency data directly to master, not replicas
- Avoids issues from replication lag causing duplicate operations
- Sharded by idempotency key for scaling

## Tradeoffs

- Increased complexity for clients and developers
- Careful exception handling required  
- Schema change and data migration challenges
- Solving edge cases like nested requests

Overall, Orpheus allowed Airbnb to achieve high data consistency for payments while scaling their SOA.

```ad-summary
title: Key Takeaways
collapse: open

- Idempotency crucial for distributed financial operations
- Orpheus separates requests into Pre-RPC, RPC, Post-RPC phases
- Uses Java lambdas to combine database commits atomically  
- Classifies exceptions as retryable or non-retryable
- Clients own retry policies and persisting idempotency keys
- Only uses master databases to avoid replication issues
- Increases complexity but enables SOA scaling with consistency
```