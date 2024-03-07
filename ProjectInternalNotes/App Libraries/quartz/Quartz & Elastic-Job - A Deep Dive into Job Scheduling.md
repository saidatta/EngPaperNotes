## Quartz Overview

### Introduction

Quartz is a robust open-source task scheduling framework presented by OpenSymphony. It assists developers in executing scheduled tasks, like extracting database information at specific times.

### Implementation

1. **Job Interface**: Begin by creating an interface for the business logic named `Job`. Classes that inherit this interface can implement diverse tasks.
    ```java
    public class HelloJob implements Job {
        // Business Logic Implementation
    }
    ```
2. **Trigger**: After defining the `Job`, a trigger is essential to execute it timely. 
    ```java
    Trigger trigger = TriggerBuilder.newTrigger("Every 1 minute");
    ```

3. **Scheduler**: This combines the `Job` and the `Trigger` to ensure the latter invokes the former at the required time.
    ```java
    Scheduler scheduler = scheduler.scheduleJob(jobDetail,trigger);
    scheduler.start();
    ```

Key Components of Quartz:
- **Scheduler**: Manager for scheduling jobs
- **Task**: JobDetail
- **Trigger**: Includes both SimpleTrigger and CronTrigger

## Elastic-Job

### Basic Principles

1. **Sharding**: Elastic-Job leverages sharding to enhance task concurrency by splitting a task into various shards. This division enables multiple execution units to process different data segments concurrently.

2. **Job Scheduling and Execution**: Elastic-Job adopts a decentralized task scheduling framework. A master node is chosen among multiple nodes. This master node is in charge of distributing the shards, with other nodes waiting. Once divided, the shards are stored in zookeeper, from where each node retrieves its share for execution.

3. **Types of Jobs**:
   - **Simple type job**: General task processing.
   - **Dataflow type job**: Specifically for data streams.
   - **Script type job**: Supports multiple scripts such as shell, python, and perl.
### Execution Principle

The Elastic-Job startup process can be illustrated using the `SimpleJob` type:
```java
public class MyElasticJob implements SimpleJob {
    public void execute(ShardingContext context) {
        // Business Logic Implementation
    }
}
```
For distributed management, including master selection, metadata storage, distributed monitoring, etc., Elastic-Job depends on zookeeper.
- Initialize the zookeeper information.
- Create a `Job` class to perform the tasks.
- Define the basic job details.
- Launch the JobScheduler for job initialization.

The `JobScheduler` class plays a critical role:
```java
public class JobScheduler {
    // Various properties like liteJobConfig, regCenter, schedulerFacade, etc.
    // Constructor to initialize properties
    // Method to initialize jobs
}
```
Upon initializing the `JobScheduler`, various steps are taken:

1. Metadata, like total shard count, is stored in the `JobRegistry`.
2. The `JobScheduleController` takes charge of the job operations like starting, pausing, and resuming.
3. The scheduler facade registers the job startup details.
4. Job scheduling is done through the `scheduleJob` method.

## Conclusion

Both Quartz and Elastic-Job offer powerful capabilities to manage and execute scheduled tasks in various environments. While Quartz provides a more centralized way of defining and triggering jobs, Elastic-Job introduces concepts like sharding and distributed task management to ensure efficient processing across multiple machines.

---


## Execution Process

### **1. Job Startup Process**:

When you start up a job, its procedure can be broken down as follows:

---

### **2. Elastic-Job Execution Process**:
- Quartz's execution is based on the business logic defined in **JobDetail**.
- To know the job execution process, inspect the content in **jobDetail**.

```java
private JobDetail createJobDetail(final String jobClass) {
    JobDetail result = JobBuilder.newJob(LiteJob.class)
                                 .withIdentity(liteJobConfig.getJobName())
                                 .build(); 
    // other codes omitted for brevity 
}
```

#### 2.1 Task Execution Content:
- Task's execution content is present in **LiteJob** class.
```java
public final class LiteJob implements Job {

    @Setter
    private ElasticJob elasticJob; 

    @Setter
    private JobFacade jobFacade; 

    @Override
    public void execute(final JobExecutionContext context) throws JobExecutionException {
        JobExecutorFactory.getJobExecutor(elasticJob, jobFacade).execute(); 
    } 
}
```
#### 2.2 Job Executor:
- **LiteJob** fetches the job executor through **JobExecutorFactory** and then triggers its execution.
```java
public final class JobExecutorFactory {
    public static AbstractElasticJobExecutor getJobExecutor(final ElasticJob elasticJob, final JobFacade jobFacade) { 
        if (null == elasticJob) { 
            return new ScriptJobExecutor(jobFacade); 
        } 
        if (elasticJob instanceof SimpleJob) { 
            return new SimpleJobExecutor((SimpleJob) elasticJob, jobFacade); 
        } 
        if (elasticJob instanceof DataflowJob) { 
            return new DataflowJobExecutor((DataflowJob) elasticJob, jobFacade); 
        } 
        throw new JobConfigurationException("Cannot support job type '%s'", elasticJob.getClass().getCanonicalName()); 
    } 
}
```
#### 2.3 `execute()` Function:
Here's a step-by-step breakdown of the `execute()` function:
1. **Check Job Execution Environment**:
   - Validates the environment where the job is set to run.
2. **Sharding Context**:
   - Fetches the current job server's sharding context.
   - The master node divides sharding items based on the appropriate sharding strategy.
   - Once divided, results are stored in Zookeeper. Other nodes then fetch these partition results.
3. **Job Status Tracking Events**:
   - Events to track the status of the job execution are published.
4. **Skipping Missed Jobs**:
   - Running jobs that were missed are skipped.
5. **Pre-job Execution Methods**:
   - Any actions/methods that need to be taken before the actual job execution begins.
6. **Actual Job Execution**:
   - If a job is triggered normally, the `execute` method in **MyElasticJob** is called to execute user-defined business logic.
---
### **3. Optimization Practice of Elastic-Job**:
#### 3.1 Idling Problem:
- Jobs in Elastic-Job can have implementation classes (`SimpleJob`, `DataFlowJob`) or may not (`ScriptJob`, `HttpJob`).
- In production, there can be multiple machines executing jobs but very few shards for each user-registered job.
- This can lead to wastage of computational resources due to idling machines.
#### 3.2 Solution:
- Let users specify the execution servers during platform registration tasks.
- **Number of Execution Servers**: \(M = \text{number of shards} + 1\)
   - The additional machine acts as a backup.
---
### **4. OPPO's Job Scheduling Solution**:
- Elastic-Job uses Zookeeper to implement elastic distribution functions.
- **Limitations**:
   - Heavy reliance on Zookeeper can lead to performance bottlenecks.
   - Some machines can remain idle if the number of shards is less than the number of instances executing tasks.
- **OPPO's Centralized Scheduling Solution**:
   - No need for regular triggers via Quartz.
   - Local tasks are activated through server messages.
   - Central server triggers message execution.
   - This method overcomes the limitations of Zookeeper and can handle massive tasks.
---
### **Summary**:
Elastic-Job integrates Quartz for job scheduling and employs Zookeeper for distributed management. By adding the concepts of elasticity and data sharding, it maximizes distributed server resources, achieving distributed task scheduling. However, the sharding concept might lead to server idling, which can be avoided in production settings.

---
The process of scheduling, triggering, and executing a distributed job, such as the `EmailPromotionJob` from our example, involves several components and steps. Let's use Elastic-Job, along with Quartz for scheduling and ZooKeeper for coordination, to illustrate this lifecycle:
### 1. Job Configuration and Registration:
Before a job can be scheduled and executed, it needs to be configured and registered.
- **Configuration:** Define the job details, including its name, class, cron expression (for scheduling), sharding parameters, etc.
- **Registration:** Register the job configuration in a central repository, which can be a relational database or a distributed configuration center like ZooKeeper.
### 2. Job Scheduling with Quartz:
Quartz is a powerful job scheduling library that uses a database (JobStore) to manage job persistence.
- The Quartz scheduler is initialized on each node (job instance).
- The scheduler polls the JobStore (database) for upcoming triggers.
- When a trigger's firing time is reached, Quartz notifies the appropriate node to execute the associated job.
### 3. Job Distribution with Elastic-Job:
Elastic-Job leverages ZooKeeper for distributed coordination.
- **Leader Election:** When multiple job nodes start, they negotiate a leader using ZooKeeper. The leader is responsible for sharding assignments.
- **Sharding:** The leader node assigns sharding items (or shards) to available job nodes based on the sharding strategy.
- **Job Execution:** When a job node is notified by Quartz to execute a job, it checks its assigned sharding items and executes the job logic only for those items.
### 4. Job Execution:
Using our `EmailPromotionJob` as an example:
- The node fetches its assigned sharding item(s) from the context.
- It then fetches the users for the respective shard from the database.
- The promotional emails are sent to the fetched users.
- Any results or logs from the job execution can be stored in a database or a logging system.
### 5. Job Completion and Result Storage:
After the job is executed:
- Execution metadata, like execution time, duration, status (success/failure), etc., can be stored in a database.
- If the job produces a tangible result (like a report), that result can be stored in a designated system or database.
- Notification mechanisms (e.g., email or messaging systems) can be used to inform stakeholders about job completion or failures.
### 6. Resilience and Failover:
Using ZooKeeper, Elastic-Job can handle node failures:
- If a node fails to execute its shard, the leader can reassign that shard to another node.
- If the leader node fails, a new leader is elected.
### 7. Monitoring and Maintenance:
Continuously monitor job nodes for health, performance, and failures. Tools and dashboards can be built around Quartz databases and ZooKeeper data to gain insights into job executions, timings, and results.

In summary, the lifecycle of our example job in a distributed setup involves configuring and registering the job, scheduling it with Quartz, distributing the workload using Elastic-Job and ZooKeeper, executing the job logic, storing results, and handling failures and monitoring. The entire system can be made robust, fault-tolerant, and highly available with the right tools and practices.


In the distributed approach, sharding is essential to effectively distribute the job load among different nodes and ensure efficient utilization of resources. Here's how sharding works with Elastic-Job, especially when the actual job logic resides on different nodes:

### 1. **Job Sharding**:

- Each instance of the job (`PromotionalEmail`, in this case) that resides on different nodes is treated as a shard.
- The job's total sharding count is usually pre-defined. For example, if you specify a sharding count of 3, it means that the job will be divided into three shards.

### 2. **Zookeeper's Role**:

- **Election of Leader:** Elastic-Job uses ZooKeeper to elect a leader amongst its instances. Only the leader will make decisions about how to shard the tasks, ensuring that the same job isn't executed multiple times simultaneously by different nodes.
  
- **Storing Sharding Information:** Once the leader has determined the sharding, it will store this information in ZooKeeper. Other nodes (followers) will check ZooKeeper for their assigned shard(s).
  
- **Sharding Strategy:** The decision on which node gets which shard depends on the sharding strategy used. There are different strategies like average distribution, Odevity (odd and even sharding items assigned to different servers), and custom strategies.
  
- **Dynamic Re-sharding:** If a node goes down or if new nodes are added, ZooKeeper helps in re-sharding. The leader will re-evaluate the sharding based on current available nodes and update the shard assignment in ZooKeeper. Existing nodes will keep polling ZooKeeper for changes in their shard assignments.

### 3. **Job Execution**:

- When it's time to execute the job, each node will check its assigned shard(s) from ZooKeeper and execute only those specific shards.
  
- For the `PromotionalEmail` example, imagine the job is to send promotional emails to 300 users. If we have three shards (or nodes), then:
  - Shard 1 might handle emails for users 1-100.
  - Shard 2 for users 101-200.
  - Shard 3 for users 201-300.
  
- Each node, based on its shard assignment, will pick up its respective segment of the job and execute it.

### 4. **Handling Job Logic on Different Nodes**:

- If the `PromotionalEmail` logic resides on different nodes from the scheduler, when the scheduler node determines it's time to run a shard of the job, it sends a signal to the respective node responsible for that shard.
  
- This can be achieved through various means like RESTful calls, RPCs, or even message queues. The node then processes its shard of the job and can report back the status/results to the scheduler or directly to ZooKeeper.

### 5. **Reporting & Monitoring**:

- Nodes can report their job execution status, success, failures, etc., back to ZooKeeper.
  
- Monitoring tools or even the Elastic-Job itself can then check these statuses in ZooKeeper for monitoring, alerting, and logging purposes.

In essence, the distributed approach using ZooKeeper ensures that each shard of the job is executed once and only once by maintaining the shard-to-node mapping and dynamically updating it as nodes come and go.


----
Certainly! Sharding a task means splitting a task into smaller subtasks that can be processed in parallel across multiple nodes. In the context of Elastic-Job and many distributed job scheduling frameworks, the concept of sharding is used to evenly distribute the workload among all available nodes to ensure efficient and parallel processing.

Here's a basic example to illustrate the idea of task sharding:

### Task:
Let's say we have a task of sending promotional emails to 1 million users.

### Without Sharding:
Without sharding, one node might try to send emails to all 1 million users. This approach is not efficient and may strain the resources of that single node.

### With Sharding:
With sharding, we can split this task into smaller subtasks, where each subtask is responsible for sending emails to a portion of the 1 million users.

1. **Divide the users into shards:** 
   
   For simplicity, let's divide our 1 million users into 10 shards, with each shard containing 100,000 users.

2. **Distribute the shards across nodes:** 

   If we have 10 nodes available in our cluster, each node will pick up one shard and will be responsible for sending emails to its respective 100,000 users.

3. **Execute the shards:** 

   Each node will now process its shard independently and in parallel with the other nodes. This parallel processing ensures that the job is completed more quickly and efficiently than if it were processed by a single node.

### Code Example:

In the context of Elastic-Job, each shard is called a "sharding item." Here's a conceptual example:

```java
public class EmailPromotionJob implements SimpleJob {
    
    @Override
    public void execute(ShardingContext shardingContext) {
        // Get the sharding item (0 to 9 in our case)
        int shardingItem = shardingContext.getShardingItem();
        
        // Calculate the start and end index for each shard
        int startIndex = shardingItem * 100_000;
        int endIndex = (shardingItem + 1) * 100_000;
        
        // Fetch users for this shard
        List<User> users = fetchUsers(startIndex, endIndex);
        
        // Send promotional emails to the fetched users
        sendPromotionalEmails(users);
    }
    
    private List<User> fetchUsers(int startIndex, int endIndex) {
        // Fetch users from the database based on the startIndex and endIndex
        // ...
        return users;
    }
    
    private void sendPromotionalEmails(List<User> users) {
        // Logic to send emails
        // ...
    }
}
```

In this example, the `EmailPromotionJob` class defines a task to send promotional emails. When the job is executed, it fetches the users for its sharding item (shard) and sends them emails.

The real power of sharding comes into play when dealing with large datasets and tasks. By distributing the load among several nodes, you can achieve high availability, fault tolerance, and increased throughput.