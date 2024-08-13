This chapter delves into the initial coding phase of the orchestrator project, focusing on creating the basic structure for each primary component: the task, worker, manager, and scheduler. This approach is akin to framing a house, where the skeleton provides a basic outline that will be fleshed out with detailed implementations later.
#### Key Concepts Covered
- Creating skeleton code for the orchestrator components.
- Identifying task states and their transitions.
- Designing interfaces for schedulers to support various scheduling strategies.
- Writing a test program to ensure basic compilability and functionality.
##### 2.1 The Task Skeleton
- **Purpose**: Represents the smallest unit of work within the orchestrator.
- **States**: Defines various stages of a task's lifecycle, including Pending, Scheduled, Running, Completed, and Failed.
- **Structure**:
  - Unique identifiers (UUIDs) for task identification.
  - Docker-specific attributes for container management, such as Image, Memory, Disk, ExposedPorts, and PortBindings.
  - Task control features like RestartPolicy.
  - Time tracking with StartTime and FinishTime for monitoring task duration.
##### 2.2 The Worker Skeleton
- **Role**: Executes and manages tasks.
- **Capabilities**:
  - Running tasks as Docker containers.
  - Accepting tasks from a manager.
  - Providing system statistics for task scheduling.
  - Maintaining task states and details.
- **Implementation**:
  - Uses a `Queue` for task processing and a `Db` map for task storage to ensure FIFO order and task retrieval, respectively.
##### 2.3 The Manager Skeleton
- **Function**: Central control unit that manages task distribution and system state.
- **Features**:
  - Manages a queue of tasks waiting to be assigned to workers.
  - Utilizes in-memory databases for tasks and task events.
  - Maintains a record of all workers and the tasks assigned to them.
- **Responsibilities**:
  - Scheduling tasks on appropriate workers.
  - Monitoring and updating task and system states.
##### 2.4 The Scheduler Skeleton
- **Definition**: An interface that outlines the necessary functions for any scheduler implementation.
- **Functions**:
  - `SelectCandidateNodes`: Identifies potential workers for a given task.
  - `Score`: Evaluates and ranks workers based on suitability for the task.
  - `Pick`: Chooses the best worker based on scoring.
##### 2.5 Additional Components: Node
- **Concept**: Represents the physical aspect of workers, including details about the hardware on which tasks run.
- **Attributes**:
  - Basic identifiers and networking details like Name and IP.
  - Hardware resources such as Memory and Disk, along with their allocated amounts.
  - A counter for tasks running on the node to help with load balancing and resource management.

##### 2.6 Testing the Skeleton Code

- **Objective**: Verify that the skeleton code is functional and can be compiled.
- **Method**:
  - Creating instances of each component.
  - Invoking methods to simulate operations (e.g., starting/stopping tasks, collecting statistics).
  - Outputting results to confirm that each component is behaving as expected.

##### Summary

This chapter sets the foundation for the orchestrator's development by translating the mental model into tangible, testable code. The skeleton code defines the basic interactions and responsibilities of each component, preparing the ground for more detailed implementation in subsequent chapters. The use of interfaces, particularly for the scheduler, highlights a flexible design approach that accommodates different strategies and requirements. This phase is crucial for ensuring that the underlying structure of the orchestrator is robust and adaptable.