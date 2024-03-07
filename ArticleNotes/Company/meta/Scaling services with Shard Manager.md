https://engineering.fb.com/2020/08/24/production-engineering/scaling-services-with-shard-manager/

## Introduction to Shard Manager
- **Purpose**: Shard Manager is a generic platform designed to streamline the development and operation of reliable sharded applications at Facebook.
- **Motivation**: Before Shard Manager, multiple teams created overlapping custom sharding solutions, leading to inefficiencies.
- **Adoption Scale**: Manages tens of millions of shards across hundreds of thousands of servers and hundreds of applications.
![[Screenshot 2024-01-26 at 10.21.22 PM.png]]
## Concept of Sharding
- **Sharding for Scaling**: Sharding is a strategy to scale out services to support high throughput.
- **Basic Strategy**: Data is spread across servers using a deterministic scheme.
- **Hashing Approach**: Simple hashing (e.g., `hash(data_key) % num_servers`) has limitations, such as data reshuffling with server addition.
- **Consistent Hashing**: Minimizes reshuffling but is limited in load balancing and constraint-based allocation.
- **Explicit Partitioning**: Divides data into shards for better flexibility and supports various constraints like data locality.
![[Screenshot 2024-01-26 at 10.21.39 PM.png]]
## Shard Management Challenges
- **Failover**: Ability to redirect traffic away from failed servers and rebuild replicas.
- **Load Balancing**: Dynamic adjustment of shard distribution for uniform resource utilization.
- **Shard Scaling**: Adjusting replication factors based on shard load.
- **Custom Solutions**: Teams at Facebook developed custom solutions, often with limited capabilities.

## Sharding with Shard Manager
- **Platform Usage**: Used by hundreds of applications including Facebook app, Messenger, WhatsApp, Instagram.
- **Application Integration**: Requires implementing `add_shard` and `drop_shard` primitives.
- **Intent-Based Specification**: Allows applications to declare reliability and efficiency requirements.
- **Load-Balancing Capability**: Uses a generic constrained optimization solver for versatile load balancing.
- **Integration in Infrastructure Ecosystem**: Supports development and operation, more sophisticated than platforms like Apache Helix.

## Types of Applications on Shard Manager
1. **Primary Only**: Single replica per shard, suitable for state externalization.
2. **Secondaries Only**: Multiple replicas with equal role, typically for read-only applications.
3. **Primary-Secondaries**: Multiple replicas with primary and secondary roles, used in storage systems requiring strict consistency.

## Building Applications with Shard Manager
- **Standardized Steps**: Linking Shard Manager library, implementing shard state transition interface, providing intent-based specification, using a common routing library.
- **Shard State Transition Interface**: Includes `add_shard` and `drop_shard` functions.
- **Sophisticated Protocols**: Supports seamless ownership handoff and minimizes downtime.
- **Role Transition and Membership Update**: For primary-secondaries applications, with functions like `change_role` and `update_membership`.

## Shard Manager Functionalities
- **Fault Tolerance**: Configures replication factors and fault domains.
- **Load Balancing**: Considers heterogeneous hardware and dynamic loads.
- **Shard Scaling**: Adjusts replication factors based on real-time load changes.
- **Operational Safety**: Coordinates with container management system Twine for event handling.

## Client Request Routing
- **Routing Library**: Used for directing requests based on application name and shard ID.

## Shard Manager Design and Implementation
- **Infrastructure Layering**: Includes host management, container management, shard management, and application layers.
- **Central Control Plane**: Monitors application states and orchestrates shard movements.
- **Opaque Shards**: Shards are treated as opaque entities, allowing flexibility in application use cases.
- **Shard Granularity**: Balances load balancing quality and management overhead.
- **Constrained Optimization Solver**: Enables versatile shard allocation strategies.
![[Screenshot 2024-01-26 at 10.21.07 PM.png]]
## Architecture of Shard Manager
- **Components**:
  - **Shard Manager Scheduler**: Orchestration of shard transitions.
  - **Application Integration**: Link to Shard Manager library, implementation of interfaces.
  - **Service Discovery System**: For shard allocation information propagation.
  - **Client Libraries**: For endpoint discovery and request routing.

## Future Challenges and Directions
- **Expansion Goals**: Support for millions of shards per application, higher modularity and pluggability, simplified onboarding for smaller applications.
- **Long-Term Vision**: Continuously enhancing the sharding solution and contributing to the technical community.

## Technical Considerations for Software Engineers
- **Implementing Shard Manager**: Understanding the implementation of `add_shard` and `drop_shard` functions, role transitions, and membership updates in sharded applications.
- **Shard Granularity and Load Balancing**: Balancing granularity for optimal load balancing and infrastructure efficiency.
- **Fault Tolerance Strategies**: Configuring replication and fault domains for distributed systems.
- **Operational Safety and Efficiency**: Integrating with container management systems for handling operational events.
- **Constrained Optimization**: Leveraging generic solvers for flexible shard allocation.
- **Client-Side Integration**: Utilizing routing libraries for efficient request handling.
- **Infrastructure Layering**: Understanding the role of shard management in the broader infrastructure context.

## Code

 Examples and Equations
- **Hashing for Sharding**: Example of simple hash-based sharding: `hash(data_key) % num_servers`.
- **Consistent Hashing**: Illustration of consistent hashing for data distribution.
- **Shard State Transition Functions**:
  ```python
  def add_shard(shard_id):
      # Load shard data and logic
      pass

  def drop_shard(shard_id):
      # Unload shard and cleanup
      pass
  ```
- **Client Routing Code Snippet**:
  ```python
  rpc_client = create_rpc_client(app_name, shard_id)
  rpc_client.foo(...)
  ```