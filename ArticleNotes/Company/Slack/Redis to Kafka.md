
Here is a visual representation of the journey Slack took from using Redis to Apache Kafka:

![Slack's Journey from Redis to Kafka](https://showme.redstarplugin.com/d/ZCLOTzpx)

[You can edit this diagram online if you want to make any changes.](https://showme.redstarplugin.com/s/BZ6IBRNP)

Let's delve into the details:

1. **Redis as a Queue (2016)**: Slack initially used Redis as a queue for jobs that were too slow for a web request, such as unfurling links, notifications, search index updates, security checks, etc. However, a database slowdown led to a job execution slowdown, which eventually caused Redis to hit its memory limit. This resulted in data loss and Redis getting stuck.

2. **Redis Limitations**: The issues with Redis were mainly due to the data structure they were using for dequeuing, the difficulty in scaling workers, the bipartite topology, and the memory deadlock. They identified the need for durable storage, a better job scheduler, and decoupling of job workers from Redis.

3. **Introduction of Apache Kafka (2016)**: To overcome the limitations of Redis, Slack introduced Apache Kafka. Initially, Kafka was added in front of Redis as a durable store. Over time, Kafka was used to collect data from clients and servers and push it to S3 using Pinterest's Secor library. Kafka was also used to manage distributed tracing events.

4. **Scaling Kafka (2021)**: By early 2021, Slack had a 90-broker Kafka cluster, which was reaching its network limit. To address this, they split the cluster and used modern nitro-enabled d3en instances, achieving a 250% improvement.

5. **Kafka as Central Nervous System (2022)**: By 2022, Kafka had become a central component in Slack's infrastructure, powering logging pipelines, trace data, billing, enterprise analytics, and security analytics. At this point, Slack was handling 6.5 Gbps, millions of messages per second, 700TB of data across 10 Kafka clusters and hundreds of nodes.

This transition from Redis to Kafka allowed Slack to handle a significantly larger scale of data and operations, demonstrating the power and scalability of Kafka for large-scale data processing.

---
**May 2023**
- Slack uses Apache Kafka at scale:  
- 6.5Gbps  
- 700TB of data  
- 100s of nodes  
  
Here's their story 👇   
   
It started in 2016 when they were using Redis as a queue.  
Web Apps -> Redis -> Workers  
  
Any jobs that were too slow for a web request went there - unfurling links, notifications, search index updates, security checks, etc.  
  
In 2016 they had a big incident with it 😱  
  
A database slowdown spiraled into a job execution slowdown spiraled into Redis hitting its memory limit.  
  
They couldn't enqueue new jobs at this point (data loss).  
  
But worse off:  
  
🤦‍♂️ Dequeueing a job from Redis requires a tiny amount of memory.  
  
Redis got totally stuck.  
  
Mind you, their peak scale at this time was up to 33,000 jobs a second!  
  
Redis wasn't cutting it:  
❌ - the data structure they were using dequeued at O(N). The longer the queue, the harder it got to dequeue  
❌ - workers were hard to scale, as they placed extra load polling Redis   
❌ - bipartite topology - every web app had to connect to every Redis  
❌ - memory deadlock: enqueue faster than you dequeue for long enough = OOM & complex manual intervention to recover  
  
So, they identified a few things to fix:  
✅ - introduce durable storage to avoid memory exhaustion and job (data) loss  
✅ - better job scheduler to support rate limiting and prioritization  
✅ - decouple the job workers from Redis, to scale more easily  
  
The solution?  
You know it.  
  
Apache Kafka ✨  
- 16 brokers  
- i3.2xlarge (61 GiB, 8 vCPU, 1.9TB NVMe SSD)  
- version 0.10\.1\.2 👵  
  
They replaced Redis incrementally - first, they added Kafka in front as a durable store.  
  
Next up, in 2017, they share that Kafka is also used to collect data from their clients & servers to push it all to S3.  
  
They use Pinterest's Secor library as the service (really, a sink connector) that persists Kafka messages into S3  
  
Kafka is also used to shepherd their distributed tracing events into the appropriate stores to allow for visualization.  
  
As you can imagine, this is at SCALE 🔥:  
- 310M traces a day (3587/s)  
- 8.5B spans a day (98.3k/s)  
  
Around early 2021, they had a 90-broker cluster, which was capping out its network at 40k packets a second.  
Consumer lag incidents were a daily occurrence, violating their SLO.  
  
How did they fix this?  
  
Easy. Split the cluster!  
  
With the modern nitro-enabled d3en instances their new cluster achieved similar performance on 20 brokers - a 250% improvement!  
  
Year by year, Kafka became an increasingly-central nervous system at their company, moving mission-critical data.  
  
In 2022, it powered:  
⭐️ - logging pipelines  
⭐️ - trace data  
⭐️ - billing  
⭐️ - enterprise analytics  
⭐️ - security analytics  
  
They shared more numbers - they were even larger 🤯  
👇   
- 6.5 Gbps  
- 1,000,000s of messages a second  
- 700TB of data (0.7PB)  
- 10 Kafka clusters  
- 100s of nodes