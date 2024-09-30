https://www.linkedin.com/pulse/you-collecting-way-too-many-logs-mudit-verma-r3ztf/?trackingId=P7AfbjNU2VO6WhEtHcVAdQ%3D%3D
#### **Introduction**
- **Context:** With the rise of cloud-native applications and microservices architectures, observability is essential for monitoring and troubleshooting systems. However, the exponential growth of telemetry data creates challenges in balancing comprehensive observability with resource consumption and costs.
- **Problem:** Traditional log collection strategies often lead to excessive data collection, increased costs, and data noise, which reduce efficiency and make it harder to identify important signals. On the other hand, minimal logging can result in missing crucial data during faults.
- **Solution:** IBM Research developed **Self-Adjusting Log Observability (SALO)**, an intelligent framework that dynamically adjusts log collection based on real-time needs, system health, and data granularity.

---

#### **Understanding the Log Observability Challenge**
- **Excessive Data Collection:**
  - Collecting logs at a consistent rate across all components, regardless of system state, leads to redundancy and irrelevant data during normal operations.
  - **Example:** Continuously logging detailed information in a healthy system might generate vast amounts of unnecessary data.
  
- **Increased Costs:**
  - Storing and processing excessive log data escalates infrastructure costs, making observability expensive.
  
- **Data Noise:**
  - A large volume of logs introduces noise, obscuring important signals (e.g., anomalies) and making it difficult to diagnose issues.
  
- **Impact on Performance-Critical Applications:**
  - Performance-critical applications may reduce or disable logging to avoid overhead, risking missing crucial data during system faults.

---

#### **Introducing Self-Adjusting Log Observability (SALO)**
SALO is designed to intelligently manage log collection based on three key questions:
1. **When to Collect Logs:**
   - Logs should only be collected when necessary, based on the health of the system components.
   - **Implementation:** A sidecar component called **GateKeeper** is attached to each application component. It includes a health detector and filtering agent to adjust log granularity dynamically.
   - **Example:** During normal operations, minimal logs are collected. When a component's health deteriorates, GateKeeper allows higher log levels.
  
2. **From Where to Collect Logs:**
   - Identify which components need increased logging, especially in a distributed system where faults can cascade.
   - **Concept:** The concept of **Blast Radius** is used to determine the impact of a faulty component on neighboring components. Logs from both the faulty component and its neighbors are collected at higher granularity.
   - **Example:** If Component A fails, SALO increases logging for both Component A and neighboring components B and C to capture cascading effects.

3. **At What Granularity to Collect Logs:**
   - Log levels are adjusted based on the interaction intensity between components.
   - **Example:** If Component A frequently interacts with Component B, and A fails, the log level for B is increased proportionally.
   - 
![[Screenshot 2024-08-19 at 1.32.46 PM.png]]
---
![[Screenshot 2024-08-19 at 1.31.57 PM.png]]
#### **SALO Architecture Overview**
- **GateKeeper:**
  - **Health Detection:** Monitors the health of the component using a lightweight model trained with the Drain algorithm.
  - **Filtering Agent:** Adjusts log levels based on health status.

- **Central Controller:**
  - **Blast Radius Calculation:** Determines the potential impact of faults and adjusts log levels for affected components.
  - **Log Level Management:** Coordinates log granularity adjustments across the system.

---

#### **Results**
- **Experiments:**
  - SALO was tested with microservice-based applications like QoTD and Train Ticket (TT) under various anomaly scenarios.
  - **Scenario Example:** In one experiment, SALO reduced log volume by up to 95% while improving downstream task accuracy (e.g., fault classification) and reducing turnaround time.
  
- **Outcome:**
  - The experiments revealed that reducing log volume while focusing on high-quality data improved the effectiveness of downstream AIOps tasks.
  - **Key Insight:** Reducing unnecessary logs can actually enhance the accuracy of fault detection by minimizing noise in the data.

- **Visualization:**
  ```
  +-------------------------+
  | Log Volume Reduction (%)|
  +-------------------------+
  | Scenario 1:    95%      |
  | Scenario 2:    90%      |
  | Scenario 3:    85%      |
  +-------------------------+
  ```

---

#### **Code Example: Simplified SALO GateKeeper Implementation**

```python
class GateKeeper:
    def __init__(self, component_name):
        self.component_name = component_name
        self.health_status = "healthy"
        self.log_level = "minimal"
    
    def update_health_status(self, status):
        self.health_status = status
        self.adjust_log_level()
    
    def adjust_log_level(self):
        if self.health_status == "unhealthy":
            self.log_level = "detailed"
        else:
            self.log_level = "minimal"
    
    def log(self, message):
        if self.log_level == "minimal":
            pass  # Log minimal details
        elif self.log_level == "detailed":
            print(f"{self.component_name}: {message}")

# Example usage
gatekeeper = GateKeeper("Component A")
gatekeeper.update_health_status("unhealthy")
gatekeeper.log("Anomaly detected")
```

---

#### **Conclusion**
- **SALO** presents a new paradigm for log observability by collecting logs based on necessity, location, and granularity rather than uniformly collecting all logs.
- **Benefits:**
  - Reduced log volume (up to 95%).
  - Enhanced accuracy in downstream AIOps tasks.
  - Lower costs for storage and processing.
  - Improved efficiency in observability pipelines.
  
- **Future Considerations:**
  - Expanding SALO to integrate with more advanced AI models for even more granular log collection.
  - Further automation to reduce human intervention in observability processes.

---

This deep dive into SALO demonstrates how intelligent log management can lead to more efficient and cost-effective observability in cloud-native environments, making it a valuable solution for large-scale distributed systems.