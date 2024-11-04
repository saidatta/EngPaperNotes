
### Obsidian Notes for "Build an Orchestrator in Go"

#### Chapter 1: What is an Orchestrator?

##### Introduction

This chapter introduces the concept of orchestration in software deployment, focusing on understanding and building a Kubernetes-like orchestrator from scratch. The emphasis is on learning through doing, aiming to implement a basic yet functional orchestrator to grasp the complexities of more sophisticated systems like Kubernetes, Apache Mesos, and HashiCorp’s Nomad.

##### Key Learning Objectives

- Understand the evolution of application deployments.
- Learn the components and interactions within an orchestration system.
- Explore state management and design tradeoffs in orchestrators.

##### 1.1 Why Implement an Orchestrator from Scratch?

- **Purpose**: Not to replace Kubernetes but to deepen understanding of orchestration mechanics.
- **Learning Method**: Building a small-scale orchestrator (less than 3,000 lines of code) to grasp large-scale systems operation.
- **Scope**: Focus on core functionalities; not production-ready but functional for educational purposes.

##### 1.2 The Evolution of Application Deployments

- **Pre-2000s**: Applications deployed directly on physical servers, each with unique hardware and maintenance needs.
- **Post-2000s**: Introduction of virtualization, allowing multiple virtual machines on a single physical server, reducing hardware dependency but still OS-dependent.
- **Mid-2010s**: Emergence of container technology (e.g., Docker) and orchestrators (e.g., Kubernetes), decoupling applications from OS and improving deployment flexibility.

##### 1.3 Container vs. Virtual Machine

- **Virtual Machine (VM)**: Emulates entire hardware system, running multiple OS instances.
- **Container**: Isolates applications at the process level using Linux kernel features (namespaces and cgroups), sharing the host OS, lighter and faster than VMs.

##### 1.4 What is an Orchestrator?

- **Definition**: A system that automates deployment, scaling, and management of containers.
- **Functionality**: Similar to a CPU scheduler but targets containers and sometimes other workloads (e.g., Nomad supports multiple types).

##### 1.5 Components of an Orchestration System

- **Task**: Smallest unit of work, runs in a container.
- **Job**: A collection of tasks performing a larger function.
- **Scheduler**: Allocates tasks to the best-suited machine.
- **Manager**: Central control entity, interacts with users and schedules tasks on machines.
- **Worker**: Executes tasks assigned by the manager, maintains task states and health metrics.
- **Cluster**: Collection of all components, scaling and HA considerations.
- **CLI**: User interface for interacting with the orchestrator, managing tasks and querying system status.

##### Figures and Diagrams

1. **Figure 1.1**: Physical deployment scenario in 2002.
2. **Figure 1.2**: Applications running on VMs.
3. **Figure 1.3**: Applications running in containers.
4. **Figure 1.4**: Basic components of an orchestration system.
	1. ![[Screenshot 2024-04-21 at 10.06.01 AM.png]]
5. **Figure 1.5**: Google’s Borg architecture.![[Screenshot 2024-04-21 at 10.06.19 AM.png]]
6. **Figure 1.6**: Apache Mesos architecture.
	1. ![[Screenshot 2024-04-21 at 10.06.41 AM.png]]
7. **Figure 1.7**: Kubernetes architecture.
	1. ![[Screenshot 2024-04-21 at 10.06.54 AM.png]]
8. **Figure 1.8**: Nomad’s architecture.
	1. ![[Screenshot 2024-04-21 at 10.07.05 AM.png]]
9. **Figure 1.9**: Mental model for Cube, the DIY orchestrator project.
	1. ![[Screenshot 2024-04-21 at 10.07.17 AM.png]]
##### Conclusion
Understanding orchestrators through hands-on development offers insight into the complexities and architectural decisions behind modern deployment systems. The goal is to equip software engineers with foundational knowledge to understand or contribute to large-scale orchestration systems.