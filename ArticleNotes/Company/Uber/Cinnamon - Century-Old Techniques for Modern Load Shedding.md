https://www.uber.com/blog/cinnamon-using-century-old-tech-to-build-a-mean-load-shedder/
**Uber's Engineering Challenge: Scaling with Grace**
- Uber operates thousands of microservices, catering to ~130 million monthly users across various domains like rides, food delivery, and freelance opportunities.
- This diverse ecosystem generates millions of requests per second, demanding robust management to prevent service overload due to spikes or inefficiencies, such as aggressive batch jobs or database slowdowns.

**Prioritizing Graceful Degradation**
- Uber emphasizes graceful degradation to handle overload, prioritizing user-facing requests over backend processes like batch jobs.
- The goal is to maintain user experience quality despite backend load challenges.
- Implementing an effective solution requires it to be automatic and configuration-free due to the scale and dynamic nature of Uber's services.

**Cinnamon: Uber's Advanced Load Shedder**
- Developed from lessons learned with *QALM* (*uber*), Cinnamon introduces a dynamic, configuration-free approach to managing service capacity and load shedding.
- Utilizes a modified TCP-Vegas algorithm for dynamic capacity adjustment and a PID controller, based on 17th-century control theory, for real-time load shedding decisions.
	- https://www.youtube.com/watch?v=6_w1vcoNjPM - TCP - vegas
		- increase congestion window = expectedRTT - actual RTT < 0
- Offers significant improvements in handling overloads, with minimal latency impact and overhead, while requiring no manual configuration.

**Foundation: QALM and CoDel**
- QALM leveraged the Controlled Delay (CoDel) algorithm to manage overload by dropping lower-priority requests based on queue times.
- Required manual configuration for priority and concurrency limits, which hindered adoption and flexibility.
- ![[Screenshot 2024-03-26 at 10.21.00 AM.png]]
**Cinnamon's Architecture and Differentiation**
- Serves as RPC middleware within Uber’s Go-based services, automatically handling request priorities across the service mesh.
- Innovates in priority propagation, zero-configuration deployment, and performance efficiency.
- Categorizes requests into tiers and cohorts, facilitating nuanced load shedding across 768 priority levels with an efficient priority queue mechanism.
- Cinnamon uses the priority attached to the request and if not present it sets a default one, depending on the calling service. The priority is actually composed of two different components, a _tier_ level and a _cohort._ The tier level designates how important the request is. At Uber we use 6 tiers, from 0 to 5, where tier 0 is the highest priority and tier 5 is the lowest.
- The cohorts are used to further segregate the requests within a tier, such that we can load-shed the same subset of requests across services. It is very much inspired by [WeChat’s approach](https://arxiv.org/abs/1806.04075?uclick_id=28918488-4758-4dfb-9ce0-4c7f90a8d89b), where we divide the requests into 128 cohorts, based on the user involved in the request. That way, if a service needs to load-shed say 5% of its tier 1 requests, it will be the same set of users (i.e., 5%) that are load-shedded. To do so, we use a simple sharding scheme with hashing and some timing information to group and automatically shift users around, to ensure that the same users are not always shedded first. To align with tier levels, the highest priority cohort is 0 and the lowest is cohort 127.
- With 6 tiers and 128 cohorts for each, we end up with 768 different priorities that we can then load-shed on. With “only” 768 different priorities, we can also build a very efficient priority queue with separate buckets for each priority.
![[Screenshot 2024-03-26 at 10.22.14 AM.png]]
**Operational Mechanics of Cinnamon**
- Assigns priority to incoming requests, with a fallback default for untagged ones.
- Incorporates a rejector for overload determination and a scheduler for managing concurrency within defined limits.
- Employs a PID controller and an auto-tuner for dynamic threshold adjustments and optimal throughput without compromising latency.

**Experimental Validation**
- Cinnamon demonstrated superior capacity to maintain service integrity and prioritize critical flows under heavy load, outperforming QALM.
- Showed better goodput under overload conditions, with a nuanced approach to shedding lower-priority requests, ensuring vital services remain unaffected.
- Maintained lower latencies for high-priority requests, evidencing its refined control mechanism over service request handling.

**Conclusion and Implications for Distributed Systems**
- Cinnamon represents a significant advancement in load shedding, combining historical insights with modern algorithmic enhancements for dynamic, efficient, and autonomous service management.
- Its design philosophy and technical solutions offer valuable lessons for designing resilient, scalable distributed systems that can automatically adapt to fluctuating loads without manual intervention.
- Reflects a broader trend in distributed systems towards self-regulating, intelligent infrastructure components capable of minimizing human configuration and maximizing operational efficiency.

**Academic and Practical Insights**
- For academia, Cinnamon exemplifies the application of classic control theory in modern distributed systems, offering a case study in cross-disciplinary innovation.
- Practitioners can draw lessons on the importance of automatic scalability solutions, the utility of historical algorithms in modern contexts, and the benefits of prioritization in complex service ecosystems.
- Highlights the potential of middleware solutions in abstracting and solving systemic challenges in large-scale, service-oriented architectures.
-----
### Cinnamon in Action: A Real-World Use Case

#### Scenario: Handling New Year's Eve Surge
It's New Year's Eve, and our ride-sharing service is experiencing an unprecedented surge in demand. Users are flooding the system with ride requests, leading to potential overload on the service infrastructure, particularly the ride-matching service. This service is crucial for matching riders with nearby drivers and is built as a microservice architecture using Java for business logic and Rust for performance-critical components.
#### Challenge: Prioritizing User Requests
During the surge, the system must prioritize incoming ride requests (high priority) over other non-critical operations, such as ride history synchronization (low priority) and data analytics tasks (lowest priority). The goal is to ensure that ride requests are processed as smoothly as possible, even under extreme load.
#### Implementing Cinnamon: A Step-by-Step Approach
1. **Priority Tagging at the Edge**: As requests enter the system, they're tagged with priorities at the edge layer. High-priority requests (ride requests) are tagged with tier 0, mid-priority requests (ride history synchronization) with tier 3, and low-priority requests (data analytics tasks) with tier 5. This tagging leverages the OpenTelemetry protocol for tracing and metadata propagation in Java services.
2. **Rust-Based Rejector Mechanism**: Inside the ride-matching service, a Rust-based rejector component evaluates incoming requests. Rust is chosen for this component due to its performance and safety features. The rejector uses a PID controller to dynamically adjust which requests to process based on the current load, with minimal latency.
3. **Java Priority Queue Management**: Requests passing the rejector are placed into a priority queue, managed by Java. This queue is designed to ensure that higher-priority requests are processed first. The Java component utilizes concurrent data structures available in the `java.util.concurrent` package to manage this queue efficiently, ensuring thread-safe operations without compromising performance.
4. **Concurrency Control with Rust**: To manage the concurrency of request processing, a Rust component enforces the inflight limit, ensuring that the system does not exceed its capacity to handle requests concurrently. This limit is dynamically adjusted by a feedback loop that monitors processing latency and throughput, similar to the TCP-Vegas algorithm's approach to congestion control.
5. **Load Shedding in Action**: When the system detects that it's nearing its capacity (e.g., the ride-matching service is processing requests at its limit), it starts shedding lower-priority requests based on the dynamic thresholds set by the PID controller. For example, data analytics tasks might be deferred or dropped to ensure that ride requests continue to be processed quickly.
#### Outcome: Enhanced Service Reliability and User Experience
By implementing Cinnamon, the ride-sharing app successfully manages the New Year's Eve surge. High-priority ride requests are processed without significant delays, ensuring users can book rides efficiently. Lower-priority tasks are gracefully degraded, preserving system stability and maintaining a high level of service reliability.
#### Technical Takeaways
- **Cross-Language Integration**: The use case demonstrates how Cinnamon's principles can be applied across different programming languages and components within a microservice architecture, leveraging the strengths of each language (Java's robust ecosystem and concurrency support, Rust's performance and memory safety).
- **Dynamic Load Management**: The PID controller and dynamic capacity adjustment algorithms enable the system to respond in real-time to changing loads, showcasing the application of classical control theory in a modern distributed systems context.
- **Zero Configuration**: Cinnamon's zero-configuration approach simplifies deployment and operation in dynamic environments, reducing the operational burden and potential for human error in high-stress situations like service surges.

This example illustrates the practical application and benefits of Cinnamon's load shedding strategy in maintaining service quality under extreme conditions, offering valuable insights for software engineers and system architects designing resilient distributed systems.