https://www.uber.com/blog/odin-stateful-platform/?uclick_id=28918488-4758-4dfb-9ce0-4c7f90a8d89b
#### **Introduction**
Uber’s Odin platform is a robust, technology-agnostic system designed to manage the vast scale of Uber’s data infrastructure. Odin was developed to handle the complex operational demands of managing stateful workloads across global, regional, and zonal deployments. The platform’s design emphasizes automation, fault tolerance, scalability, and self-healing capabilities, enabling Uber to efficiently manage thousands of databases and other stateful services with minimal human intervention.

### **The Scale of Odin**

Since its inception in 2014, Odin has grown to manage a staggering amount of infrastructure:

- **Hosts**: The platform manages over 100,000 hosts.
- **Workloads**: Odin supports around 300,000 workloads, which are similar to Kubernetes pods, each comprising multiple containers.
- **Containers**: The platform oversees approximately 3.8 million individual containers.
- **Storage Capacity**: Uber's fleet has multiple exbibytes of storage and millions of compute cores.
- **Technologies Supported**: Odin supports 23 different technologies, including MySQL, Cassandra, HDFS, Presto, Kafka, and resource scheduling frameworks like Yarn and Buildkite.

Odin’s technology-agnostic nature allows it to manage these diverse systems efficiently, regardless of the underlying technology or cloud provider, making it an essential part of Uber’s infrastructure.

### **Self-Healing and Intent-Based Platform**
Odin is designed to be fully declarative and intent-based, ensuring that the platform is self-healing, scalable, and failure-tolerant. The key concept here is the **goal state**, which represents the desired state of the system. Odin continuously monitors the actual state of the system and uses **remediation loops** to converge the actual state to the goal state.

#### **Remediation Loops**
Remediation loops are modular and loosely coupled, allowing them to be developed as independent microservices. Each loop follows these steps:

1. **Inspect the Goal State**: Determine the desired configuration or state of the system.
2. **Collect the Actual State**: Gather real-time data about the system’s current state.
3. **Identify Discrepancies**: Compare the goal state with the actual state to find differences.
4. **Adjust the System State**: Use Cadence workflows to make necessary adjustments to bring the actual state closer to the goal state.

This mechanism is similar to Kubernetes controllers but operates on a much larger and more complex scale, with Odin managing the global system state through its data integration platform, **Grail**.

### **Grail: The Data Integration Platform**

Grail provides a unified data model that spans all technologies within Odin. It offers an up-to-date view of both the goal and actual states across Uber’s global infrastructure. Grail acts like an enhanced resource model in the Kubernetes APIServer, enabling comprehensive queries and insights into the platform's status. This unified view is crucial for efficiently managing the platform's vast scale and complexity.

### **Managing Generic Storage Clusters**

Odin employs a technology-agnostic cluster model that supports generic operations across various storage clusters, such as:

- Adding or removing workloads
- Upgrading container images
- Managing cluster-related tasks

This model allows Uber’s teams to extend the platform with plugins that tailor cluster management to the specific needs of each technology.

### **Host-Level Architecture**

Odin’s control plane is divided into global and host-local components. At the host level, two agents operate:

1. **Odin-Agent**: A technology-agnostic agent shared among all workloads on a host. It manages host-level resources such as CPU, memory, disk volumes, and bootstrapping workloads.
2. **Worker**: A technology-specific agent running in a sidecar container alongside the workload. It ensures the running containers align with the goal state and reports health data to the global control plane.

This separation of concerns between global and host-local components enhances robustness and resilience, allowing for more fine-grained control over operations and reducing the risk of widespread failures.

### **Optimizing for Robustness and Resilience**

Odin is designed to be highly resilient, with features like:

- **Separation of Control Plane**: The global and host-local separation allows for safer, more localized changes.
- **Make-Before-Break Operations**: When replacing workloads, Odin ensures the new workload is fully operational before the old one is terminated. This approach minimizes downtime and prevents clusters from becoming under-provisioned during transitions.

### **Transitioning Identities and Workload Replacement**

In Odin, transitioning workloads from one host to another—referred to as **replacing** workloads—is a common operation. This process ensures the workload identity is preserved across rescheduling, which is crucial for stateful services. Unlike Kubernetes’ StatefulSets, Odin must handle the additional complexity of locally attached drives, requiring the system to replicate data before shutting down the original workload.

### **Coordination Challenges and Growing Pains**

As Odin scaled, the need for automated coordination became apparent. Early versions of Odin relied heavily on human operators to initiate changes, but as the platform grew, this approach became impractical. Complex operations like workload migration, host bin-packing, and fleet-wide upgrades required a more sophisticated system to manage concurrency and ensure that operations did not conflict, potentially causing downtime or data loss.

### **Centralized Coordination and Concurrency Control**

To address these challenges, Uber developed a centralized system for coordinating operations across the platform. This system ensures that:

- **Global Concurrency**: Operations are globally coordinated to prevent conflicts.
- **Rate Limits**: Platform-wide rate limits are enforced to prevent system overloads.
- **Technology-Specific Customization**: The system supports customization based on the specific needs of each technology, ensuring safe and efficient operations.

### **Conclusion**

Odin represents a significant advancement in Uber’s ability to manage its stateful infrastructure at scale. By abstracting away the complexities of individual technologies and providing a unified, self-healing platform, Odin allows Uber to operate its vast infrastructure efficiently and reliably. The platform's intent-based design, coupled with robust coordination mechanisms, ensures that Uber’s databases and other stateful services can scale to meet the demands of a global, always-on business.

### **Future Directions**

The next post in this series will delve deeper into how Uber has further optimized Odin for large-scale operations, focusing on the coordination mechanisms and automation improvements that have enabled Uber to manage its infrastructure at an unprecedented scale. Stay tuned!