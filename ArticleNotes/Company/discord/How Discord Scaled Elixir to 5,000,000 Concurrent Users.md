## Introduction
From the beginning, Discord has been an early adopter of Elixir, utilizing the Erlang VM for building a highly concurrent, real-time system. This document explores the journey and technical details of scaling Discord's infrastructure to support nearly five million concurrent users, highlighting key lessons learned and custom libraries created.
## Table of Contents
1. [Message Fanout](#message-fanout)
2. [Fast Access Shared Data](#fast-access-shared-data)
3. [Limited Concurrency](#limited-concurrency)
4. [Conclusion](#conclusion)
## Message Fanout
### Initial Approach
Discord's architecture involves users connecting to WebSocket sessions, which communicate with guild processes (GenServers). When an event is published in a guild, it is fanned out to every connected session.
#### Initial Implementation
```elixir
def handle_call({:publish, message}, _from, %{sessions: sessions}=state) do
  Enum.each(sessions, &send(&1.pid, message))
  {:reply, :ok, state}
end
```
#### Visualization
```
+---------------------+
|    Guild Process    |
+----------+----------+
           |
           v
+----------+----------+      +----------+----------+
| Session Process 1   |      | Session Process 2   |
|                     |      |                     |
| Messages: [msg1, ..]|      | Messages: [msg1, ..]|
+---------------------+      +---------------------+
```
### Challenges
- **Scalability Issue:** The above approach worked for small groups but failed under large-scale usage (e.g., 30,000 users in a guild).
- **Performance Bottleneck:** High wall clock time for sending messages due to Erlang de-scheduling.
### Solution: Manifold
- **Manifold**: A library that distributes the work of sending messages to remote nodes, reducing CPU cost and network traffic.
#### Manifold Implementation
```elixir
Manifold.send([self(), self()], :hello)
```
#### Visualization
```
+---------------------+
|    Guild Process    |
+----------+----------+
           |
           v
+----------+----------+      +----------+----------+
| Manifold Partitioner|      | Manifold Partitioner|
|                     |      |                     |
+----------+----------+      +----------+----------+
           |                          |
           v                          v
+----------+----------+      +----------+----------+
| Session Process 1   |      | Session Process 2   |
+---------------------+      +---------------------+
```
### Results
- **Improved CPU Utilization**: Distributed message sending reduced CPU load.
- **Reduced Network Traffic**: Grouping messages by remote node decreased overall network usage.
## Fast Access Shared Data
### Problem
- **Ring Data Structure**: Used for consistent hashing to determine node locations. Performance issues arose with bursts of user reconnections.
- **Original Solution**: Used a single Erlang process with a C port, which became a bottleneck under high load.
### Solution: FastGlobal
- **FastGlobal**: Port of mochiglobal to Elixir, leveraging Erlang's read-only shared heap for constant data access.
#### FastGlobal Implementation
```elixir
defmodule FastGlobalExample do
  use FastGlobal, state: :persistent

  def get_value do
    FastGlobal.get(:my_key)
  end

  def set_value(value) do
    FastGlobal.put(:my_key, value)
  end
end
```
#### Visualization
```
+---------------------+
| FastGlobal Process  |
+----------+----------+
           |
           v
+----------+----------+      +----------+----------+
| Session Process 1   |      | Session Process 2   |
+---------------------+      +---------------------+
           |
           v
+---------------------+
|    Ring Data        |
+---------------------+
```

### Results
- **Reduced Lookup Time**: From 17.5 seconds to 750ms for 500,000 sessions.
- **Minimal Overhead**: Lookup cost reduced to 0.3μs per access.

## Limited Concurrency
### Problem
- **Guild Registry Overload**: 5,000,000 session processes causing stampedes on limited guild registry processes.
- **Resulting Issues**: Cascading service outages due to message queue overflows.

### Solution: Semaphore
- **Semaphore Library**: Implements an atomic counter to limit concurrent requests, preventing overload.

#### Semaphore Implementation
```elixir
semaphore_name = :my_semaphore
semaphore_max = 10
case Semaphore.call(semaphore_name, semaphore_max, fn -> :ok end) do
  :ok ->
    IO.puts "success"
  {:error, :max} ->
    IO.puts "too many callers"
end
```

#### Visualization
```
+---------------------+
|   Semaphore         |
+----------+----------+
           |
           v
+----------+----------+      +----------+----------+
| Session Process 1   |      | Session Process 2   |
+---------------------+      +---------------------+
```

### Results
- **Improved Stability**: Prevented cascading failures by managing concurrent access.
- **High Throughput**: Maintained performance under high load conditions.

## Conclusion
Choosing Elixir and Erlang for Discord's real-time infrastructure proved to be a beneficial decision, allowing for significant scalability and performance improvements. The development of custom libraries such as Manifold, FastGlobal, and Semaphore addressed critical challenges and ensured the system could handle millions of concurrent users efficiently.

### Additional Resources
- **Manifold**: [GitHub Repository](https://github.com/discordapp/manifold)
- **FastGlobal**: [GitHub Repository](https://github.com/discordapp/fastglobal)
- **Semaphore**: [GitHub Repository](https://github.com/discordapp/semaphore)

This journey demonstrates the importance of understanding the underlying architecture and leveraging the right tools and techniques to scale a highly concurrent system like Discord.