
# Abstract

Borg is Google's cluster management system that supports high utilization, high availability, and simplified user experience. It runs hundreds of thousands of jobs from various applications across large clusters. Borg offers a declarative job specification language, name service integration, real-time job monitoring, and tools for analyzing and simulating system behavior.

# 1. Introduction

Borg provides three main benefits: (1) it hides resource management and failure handling details, allowing users to focus on application development, (2) it operates with high reliability and availability and supports applications with the same, and (3) it effectively manages workloads across tens of thousands of machines.

# 2. The User Perspective

Borg's users include Google developers and system administrators (SREs) who manage Google applications and services. Users submit jobs, composed of one or more tasks, to Borg, which are then executed within a Borg cell.

## 2.1 The Workload

Borg handles a heterogeneous workload consisting of:

1.  Long-running services that should "never" go down and handle short-lived, latency-sensitive requests. These services support end-user-facing products like Gmail, Google Docs, and web search, as well as internal infrastructure services like BigTable.
2.  Batch jobs that take a few seconds to a few days to complete, which are less sensitive to short-term performance fluctuations.

The workload mix varies across cells and over time. Borg is designed to handle all these cases effectively.

## 2.2 Clusters and Cells

Machines in a cell belong to a single cluster, defined by the high-performance datacenter-scale network fabric connecting them. A cluster resides within a single datacenter building, and a collection of buildings makes up a site.

A cluster usually hosts one large cell and may have smaller-scale test or special-purpose cells. The median cell size is about 10,000 machines, excluding test cells. Machines in a cell are heterogeneous in terms of size, processor type, performance, and capabilities. Borg isolates users from these differences by managing resource allocation, task execution, monitoring, and failure recovery.

## 2.3 Jobs and Tasks

A Borg job has properties like name, owner, and the number of tasks. Jobs can have constraints to force tasks to run on machines with specific attributes. Constraints can be hard or soft, acting like preferences rather than requirements. Jobs run in just one cell.

Tasks are sets of Linux processes running in a container on a machine. Tasks have properties like resource requirements and task index within the job. Resource dimensions are specified independently at fine granularity.

Users interact with jobs through remote procedure calls (RPCs) and Borg job configurations are written in BCL, a declarative configuration language.

Tasks can be updated with new specifications by pushing new job configurations to Borg. Tasks can also request to be notified before being preempted, allowing time for clean-up and saving state.

## 2.4 Allocs

Allocs (short for allocations) are reserved sets of resources on a machine where one or more tasks can run. Allocs can be used to reserve resources for future tasks, retain resources between stopping and starting tasks, and gather tasks from different jobs onto the same machine.

Allocs function similarly to resources of a machine, and multiple tasks running inside an alloc share its resources. If an alloc must be relocated to another machine, its tasks are rescheduled with it. Alloc sets are groups of allocs reserving resources on multiple machines, and jobs can be submitted to run in these alloc sets.

## 2.5 Priority, Quota, and Admission Control

Borg uses priority and quota to handle an excess of work. Every job has a priority, a small positive integer, with higher-priority tasks obtaining resources at the expense of lower-priority ones. Priority bands include monitoring, production, batch, and best effort.

Priority expresses the relative importance of jobs, while quota is used for admission control. Quota is expressed as a vector of resource quantities at a given priority for a period of time. Jobs with insufficient quota are rejected upon submission.

Higher-priority quota costs more, and quota allocation is handled outside of Borg. Users can only submit jobs if they have sufficient quota at the required priority. Borg also has a capability system that grants special privileges to some users.

## 2.6 Naming and Monitoring

Borg creates a stable "Borg name service" (BNS) name for each task, which includes the cell name, job name, and task number. Borg writes the task's hostname and port into a consistent, highly-available file in Chubby with this name.

Borg writes job size and task health information into Chubby for load balancing. Tasks include built-in HTTP servers that publish health and performance metrics. Borg monitors these and restarts tasks that do not respond promptly or return an HTTP error code.

Sigma, a web-based UI, allows users to examine the state of their jobs and tasks. Borg records all job submissions, task events, and resource usage information in Infrastore. This data is used for various purposes, including usage-based charging, debugging, and capacity planning.

## 3. Borg Architecture

Borg consists of a set of machines, a logically centralized controller called the Borgmaster, and an agent process called the Borglet that runs on each machine in a cell. All components are written in C++.

### 3.1 Borgmaster

Each cell's Borgmaster has two processes: the main Borgmaster process and a separate scheduler. The main Borgmaster process handles client RPCs, manages state machines for various objects, communicates with Borglets, and offers a web UI as a backup to Sigma.

The Borgmaster is logically a single process but is replicated five times. Each replica maintains an in-memory copy of most of the cell's state, which is also recorded in a highly-available, distributed, Paxos-based store. A single elected master per cell serves as the Paxos leader and state mutator. Master election and failover typically take about 10 seconds but can take up to a minute in a big cell.

The Borgmaster's state at a point in time is called a checkpoint and is stored in the Paxos store. Checkpoints have various uses, including restoring state, fixing issues, building a persistent log of events, and offline simulations.

A Borgmaster simulator called Fauxmaster can read checkpoint files and contains a complete copy of the production Borgmaster code with stubbed-out interfaces to the Borglets. Fauxmaster is used to debug failures, perform capacity planning, and conduct sanity checks before making changes to a cell's configuration.

## 3.2 Scheduling

When a job is submitted, the Borgmaster records it in the Paxos store and adds the job's tasks to the pending queue. The scheduler scans the queue asynchronously, assigning tasks to machines with sufficient available resources. It proceeds from high to low priority, using a round-robin scheme for fairness. The scheduling algorithm consists of feasibility checking and scoring.

Feasibility checking finds machines that meet the task's constraints and have enough available resources. Scoring determines the "goodness" of each feasible machine, taking into account user-specified preferences and built-in criteria.

Borg originally used a variant of E-PVM for scoring but has since shifted to a hybrid model that reduces stranded resources, providing better packing efficiency. If a machine selected by scoring doesn't have enough available resources, Borg preempts lower-priority tasks to make room for the new task.

Task startup latency is an area of ongoing improvement, with package installation taking about 80% of the total time. To reduce startup time, the scheduler prefers assigning tasks to machines that already have the necessary packages installed. Borg also uses tree- and torrent-like protocols to distribute packages in parallel.

### 3.3 Borglet

The Borglet is a local Borg agent present on every machine in a cell. It starts and stops tasks, restarts them if they fail, manages local resources, rolls over debug logs, and reports the machine's state to the Borgmaster and other monitoring systems.

The Borgmaster polls each Borglet every few seconds to retrieve the machine's current state and send it any outstanding requests. Each Borgmaster replica runs a stateless link shard to handle communication with some Borglets for performance scalability and resiliency.

If a Borglet does not respond to several poll messages, its machine is marked as down and any tasks it was running are rescheduled. If communication is restored, the Borgmaster tells the Borglet to kill rescheduled tasks to avoid duplicates. A Borglet continues normal operation even if it loses contact with the Borgmaster, ensuring tasks and services remain operational.

## 3.4 Scalability

Borg's centralized architecture has managed to overcome scalability limits so far. A single Borgmaster can manage thousands of machines and handle arrival rates above 10,000 tasks per minute. Busy Borgmasters use 10-14 CPU cores and up to 50 GiB RAM. Several techniques have been used to achieve this scale.

Early Borgmaster versions used a simple, synchronous loop. To handle larger cells, the scheduler was split into a separate process, operating in parallel with other Borgmaster functions. The scheduler replica operates on a cached copy of the cell state and repeatedly retrieves state changes from the elected master, updating its local copy, assigning tasks, and informing the elected master of those assignments.

To improve response times, separate threads were added to communicate with Borglets and respond to read-only RPCs. These functions were then sharded across the five Borgmaster replicas. This keeps the UI response time and Borglet polling interval low.

Several factors make the Borg scheduler more scalable:

1.  **Score caching**: Evaluating feasibility and scoring a machine is expensive, so Borg caches scores until the properties of the machine or task change, reducing cache invalidations.
2.  **Equivalence classes**: Tasks in a Borg job usually have identical requirements and constraints. Borg only performs feasibility and scoring for one task per equivalence class, which consists of tasks with identical requirements.
3.  **Relaxed randomization**: The scheduler examines machines in a random order until it finds enough feasible machines to score, reducing the amount of scoring and cache invalidations needed, and speeding up task assignments.

Scheduling an entire cell's workload from scratch typically takes a few hundred seconds, while an online scheduling pass over the pending queue completes in less than half a second.

## 4. Availability

Failures are common in large-scale systems. To handle such events, applications running on Borg use techniques like replication, storing persistent state in a distributed file system, and taking occasional checkpoints. Borg mitigates the impact of these events by:

-   Automatically rescheduling evicted tasks
-   Reducing correlated failures by spreading tasks across failure domains
-   Limiting the allowed rate of task disruptions and simultaneous task downtime
-   Using declarative desired-state representations and idempotent mutating operations
-   Rate-limiting task relocation from unreachable machines
-   Avoiding repeating task::machine pairings that cause crashes
-   Recovering critical intermediate data by repeatedly re-running a logsaver task

A key design feature of Borg is that already-running tasks continue even if the Borgmaster or a task's Borglet goes down. Borgmaster achieves 99.99% availability in practice through replication, admission control, and deploying instances using simple, low-level tools.

## 5. Utilization

Borg aims to efficiently use Google's fleet of machines to save costs. Figure 5 shows that segregating prod and non-prod work into different cells would need more machines. Both graphs illustrate the extra machines required if the prod and non-prod workloads were sent to separate cells, expressed as a percentage of the minimum number of machines needed to run the workload in a single cell.

## 5.1 Evaluation methodology

To evaluate policy choices for Borg, cell compaction was chosen as a metric, which measures how small a cell the given workload could fit into. This metric avoids pitfalls of synthetic workload generation and modeling. Fauxmaster was used to obtain high-fidelity simulation results using data from real production cells and workloads.

15 Borg cells were selected for evaluation, maintaining machine and workload heterogeneity. Experiments were repeated 11 times for each cell with different random-number seeds. Error bars were used to display the min and max number of machines needed, with the 90%ile value as the result.

Figures 6 and 7 showcase the additional machines needed for segregating users and subdividing cells into smaller ones, respectively. Cell compaction provides a fair, consistent way to compare scheduling policies and translates directly into cost/benefit results.

## 5.2 Cell sharing

Nearly all machines in Borg run both prod and non-prod tasks simultaneously. Segregating prod and non-prod work would need 20-30% more machines in the median cell, as Borg reclaims unused resources from prod jobs to run non-prod work, reducing the need for additional machines.

Pooling resources by sharing Borg cells among thousands of users reduces costs significantly. However, there are concerns regarding CPU interference due to packing unrelated users and job types onto the same machines. Experiments were conducted to assess potential performance interference, and the results were not clear-cut:

1.  CPI (cycles per instruction) had a positive correlation with overall CPU usage and number of tasks on the machine, but the correlations only explain 5% of the variance in CPI measurements.
2.  Mean CPI in shared cells was 1.58, while it was 1.53 in dedicated cells, indicating a 3% worse CPU performance in shared cells.
3.  The Borglet had a CPI of 1.20 in dedicated cells and 1.43 in shared ones, suggesting a 1.19x faster performance in dedicated cells.

Figures 8 and 9 showcase how bucketing resource requirements would need more machines and that no bucket sizes fit most tasks well.

Despite the potential performance interference, sharing is still beneficial as the CPU slowdown is outweighed by the decrease in machines required across different partitioning schemes, and the sharing advantages apply to all resources, including memory and disk.

##   5.3 Large cells

Google builds large cells to allow large computations and decrease resource fragmentation. Tests were conducted by partitioning the workload for a cell across multiple smaller cells. Figure 7 confirms that using smaller cells would require significantly more machines.

## 5.4 Fine-grained resource requests

Borg users request CPU in units of milli-cores and memory and disk space in bytes. Figure 8 shows that users take advantage of this granularity, with few obvious "sweet spots" in the amount of memory or CPU cores requested and few correlations between these resources.

Offering fixed-size containers or virtual machines, common among IaaS providers, would not be a good match for Google's needs. Bucketing CPU core and memory resource limits by rounding them up to the next nearest power of two would require 30-50% more resources in the median case (Figure 9).

## 5.5 Resource reclamation

Resource reclamation is the process of estimating a task's resource usage and reclaiming unused resources for work that can tolerate lower-quality resources, like batch jobs. Borg scheduler uses limits for prod tasks to ensure they don't rely on reclaimed resources, while non-prod tasks can be scheduled into reclaimed resources.

If reservations (predictions) are wrong, a machine may run out of resources at runtime. In such cases, non-prod tasks are killed or throttled, never prod ones. Figure 10 shows that many more machines would be required without resource reclamation, with about 20% of the workload running in reclaimed resources in a median cell.

Figure 11 shows the ratio of reservations and usage to limits. Memory limit exceeding tasks are preempted first, while CPU can be throttled, allowing short-term spikes to push usage above reservation.

Figure 12 shows the results of adjusting the parameters of the resource estimation algorithm in a live production cell. Reservations are closer to usage in more aggressive settings, with a slight increase in out-of-memory (OOM) events. After review, the net gains outweighed the downsides, and medium resource reclamation parameters were deployed to other cells.

## 6. Isolation

Sharing machines between applications increases utilization but requires good security and performance isolation mechanisms to prevent tasks from interfering with one another.

### 6.1 Security isolation

Linux chroot jail is the primary security isolation mechanism between multiple tasks on the same machine. The borgssh command is used for remote debugging, providing tighter access control. VMs and security sandboxing techniques are used to run external software by Google’s AppEngine (GAE) and Google Compute Engine (GCE).

### 6.2 Performance isolation

All Borg tasks run inside a Linux cgroup-based resource container, offering improved control. However, occasional low-level resource interference still occurs. Borg tasks have an appclass, distinguishing between latency-sensitive (LS) tasks and batch tasks. High-priority LS tasks receive the best treatment, potentially starving batch tasks temporarily.

Borg tasks are divided into compressible resources (CPU cycles, disk I/O bandwidth) and non-compressible resources (memory, disk space). The Borglet is responsible for handling resource management and task termination when necessary. To improve performance isolation, LS tasks can reserve entire physical CPU cores, while batch tasks can run on any core but with tiny scheduler shares relative to LS tasks. Work is ongoing to further optimize performance isolation, including thread placement, CPU management, and improving Borglet control fidelity.

Most tasks are allowed to consume resources up to their limit and exploit unused (slack) resources. This complements the results for reclaimed resources, with batch tasks exploiting unused and reclaimed memory opportunistically.

## 7. Related work

Resource scheduling has been extensively studied in various contexts such as HPC supercomputing Grids, networks of workstations, and large-scale server clusters. This section focuses on the most relevant work in the context of large-scale server clusters.

-   Cluster traces from Yahoo!, Google, and Facebook have been analyzed to understand the challenges of scale and heterogeneity in modern datacenters and workloads.
-   Apache Mesos separates resource management and placement functions between a central resource manager and multiple frameworks. Borg mostly centralizes these functions using a request-based mechanism.
-   YARN is a Hadoop-centric cluster manager, which negotiates resources with a central resource manager. YARN's resource manager recently became fault-tolerant, and it has been extended to support multiple resource types, priorities, preemptions, and advanced admission control.
-   Facebook's Tupperware is a Borg-like system for scheduling cgroup containers on a cluster, providing a form of resource reclamation.
-   Microsoft's Autopilot system provides software provisioning and deployment, system monitoring, and repair actions for faulty software and hardware.
-   Quincy uses a network flow model for fairness and data locality-aware scheduling, whereas Borg uses quota and priorities to share resources among users.
-   Cosmos focuses on batch processing, ensuring fair access to resources donated to the cluster.
-   Microsoft's Apollo system uses per-job schedulers for short-lived batch jobs to achieve high throughput, while Borg uses a central scheduler for placement decisions.
-   Alibaba's Fuxi supports data-analysis workloads with a central FuxiMaster (replicated for failure-tolerance) to gather resource-availability information and match it with application requests.
-   Omega supports multiple parallel, specialized verticals, each equivalent to a Borgmaster minus its persistent store and link shards.
-   Google's open-source Kubernetes system places applications in Docker containers onto multiple host nodes, running on both bare metal and cloud hosting providers.
-   High-performance computing community solutions like Maui, Moab, and Platform LSF have different requirements in terms of scale, workloads, and fault tolerance compared to Google's cells.
-   Virtualization providers and datacenter solution providers offer cluster management solutions that typically scale to around 1,000 machines.
-   Automation and operator scaleout are important parts of managing large-scale clusters, with Borg's design philosophy allowing it to support tens of thousands of machines per operator (SRE).
- 
## 8. Lessons and future work

This section covers qualitative lessons learned from operating Borg in production for more than a decade, and how these observations have been leveraged in designing Kubernetes.

### 8.1 Lessons learned: the bad

-   **Jobs are restrictive as the only grouping mechanism for tasks**: Borg lacks a first-class way to manage an entire multi-job service as a single entity or to refer to related instances of a service. Kubernetes uses labels instead, which are arbitrary key/value pairs that users can attach to any object in the system, allowing more flexibility than the single fixed grouping of a job.
-   **One IP address per machine complicates things**: In Borg, all tasks on a machine use the single IP address of their host, sharing the host's port space, leading to various difficulties. Kubernetes provides every pod and service with its own IP address, making it more user-friendly and eliminating complications.
-   **Optimizing for power users at the expense of casual ones**: Borg provides a large set of features aimed at power users, which can make it harder for casual users and constrain its evolution. The solution is to build automation tools and services that run on top of Borg, with settings determined from experimentation.

## 8.2 Lessons learned: the good

A number of Borg's design features have been beneficial and have stood the test of time.

-   **Allocs are useful**: The Borg alloc abstraction has led to widely-used patterns such as logsaver and data-loader tasks. Kubernetes uses pods, similar to allocs, as resource envelopes for containers that are always scheduled onto the same machine and can share resources.
-   **Cluster management is more than task management**: Applications running on Borg benefit from cluster services like naming and load balancing. Kubernetes supports these features with the service abstraction, automatically load-balancing connections among pods and keeping track of pod locations.
-   **Introspection is vital**: Borg surfaces debugging information to users, enabling self-help as the first step in debugging. Kubernetes aims to replicate many of Borg's introspection techniques with tools like cAdvisor, Elasticsearch/Kibana, Fluentd, and a unified event recording mechanism.
-   **The master is the kernel of a distributed system**: Borgmaster evolved from a monolithic system to a kernel at the heart of an ecosystem of services managing user jobs. Kubernetes architecture goes further with an API server at its core and small, composable micro-services as clients, such as the replication controller and the node controller.

## 8.3 Conclusion

Over the past decade, virtually all of Google's cluster workloads have switched to using Borg. The system continues to evolve, and lessons learned from Borg have been applied to Kubernetes.

## Acknowledgments

The authors of the paper performed evaluations and wrote the paper, but the success of Borg is credited to the dozens of engineers who designed, implemented, and maintained its components and ecosystem. The initial Borgmaster and Borglet were primarily designed and implemented by a group of individuals, with numerous subsequent contributors. The Borg SRE team has also been crucial to its success. Borg configuration language (BCL) and borgcfg tool were originally developed by Marcel van Lohuizen and Robert Griesemer. The authors also thank their reviewers and their shepherd for their feedback on the paper.

### TAGS
-   #GoogleBorg  #ClusterManagement 
-   #Allocs  #ResourceAllocation
-   #Priority
-   #Quota
-   #AdmissionControl
-   #Naming
-   #Monitoring
-   #BNS
-   #Sigma
-   #Infrastore
-  #Borgmaster
-   #Borglet
-   #Paxos
-   #Checkpoints
-   #Fauxmaster 
-   #Scheduling
-   #BorgScheduling
-   #FeasibilityChecking
-   #Scoring
-   #TaskStartupLatency
-   #Borglet
-   #LinkShard
-   #Scalability
-   #BorgScalability
-   #BorgScheduler
-   #ScoreCaching
-   #EquivalenceClasses
-   #RelaxedRandomization
-  #CellCompaction
-   #Fauxmaster
-   #BorgCells
-   #MachineHeterogeneity
-   #WorkloadHeterogeneity
