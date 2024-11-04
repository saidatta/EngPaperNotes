## Overview

Operational analytics refers to the **collection, processing, and analysis of data near the source of data generation**, typically at or near the user-facing application, rather than relying solely on analytical plane systems. This approach involves performing analytical workloads close to the applications or microservices that generate the data, leveraging the **streaming plane**'s ability to run asynchronous processes at scale.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Why Execute Analytical Workloads on the Operational Plane](#why-execute-analytical-workloads-on-the-operational-plane)
   - [Benefits of Operational Analytics](#benefits-of-operational-analytics)
3. [Limitations and Challenges](#limitations-and-challenges)
4. [Trusting Streaming Data](#trusting-streaming-data)
   - [Frank McSherry's Three Characteristics](#frank-mcsherrys-three-characteristics)
   - [Mathematical Representation of Value Degradation Over Time](#mathematical-representation-of-value-degradation-over-time)
5. [Alignment with Data Mesh](#alignment-with-data-mesh)
   - [Definition of Data Mesh](#definition-of-data-mesh)
   - [Pillars of Data Mesh](#pillars-of-data-mesh)
   - [Operational Analytics and Data Mesh Alignment](#operational-analytics-and-data-mesh-alignment)
6. [Challenges of Implementing a Data Mesh](#challenges-of-implementing-a-data-mesh)
7. [Simplifying Data Mesh with Streaming Databases](#simplifying-data-mesh-with-streaming-databases)
8. [Summary](#summary)
9. [References](#references)
10. [Tags](#tags)

---

## Introduction

Operational analytics focuses on performing analytical workloads near the source of data generation, typically at or near user-facing applications or microservices. This approach leverages the **streaming plane**'s capabilities to run asynchronous processes at scale, enabling real-time insights and decision-making.

---

## Why Execute Analytical Workloads on the Operational Plane

It's reasonable to question why one would execute analytical workloads on the operational plane, especially considering the traditional separation between transactional (OLTP) and analytical (OLAP) workloads. However, moving analytical workloads closer to the operational plane offers several benefits:

### Benefits of Operational Analytics

1. **Real-Time Decision-Making**:
   - **Definition**: The ability to derive insights from data as it is generated, allowing for immediate action.
   - **Benefit**: Crucial for making quick and informed decisions that impact ongoing operations.
   - **Example**: A retail website adjusting product recommendations in real-time based on a user's current browsing behavior.

2. **Enhanced Efficiency**:
   - **Definition**: Streamlining processes by embedding analytical capabilities directly into operational systems.
   - **Benefit**: Reduces the need for manual intervention, enhances efficiency, and decreases processing time.
   - **Example**: Automated fraud detection in financial transactions occurring instantly as transactions are processed.

3. **Improved Customer Experience or Personalization**:
   - **Definition**: Personalizing interactions with customers based on real-time data.
   - **Benefit**: Enhances customer satisfaction and loyalty by providing relevant experiences.
   - **Example**: Real-time chatbots offering personalized support based on user interactions.

4. **Proactive Issue Resolution and Predictive Analytics**:
   - **Definition**: Using predictive modeling to identify potential issues before they escalate.
   - **Benefit**: Allows for timely intervention and issue resolution, preventing downtime or customer dissatisfaction.
   - **Example**: Predictive maintenance in manufacturing to prevent equipment failures.

5. **Cost Savings and Resource Optimization**:
   - **Definition**: Optimizing resource allocation through real-time analytics.
   - **Benefit**: Leads to cost savings and improved resource utilization.
   - **Example**: Dynamic pricing strategies adjusting prices based on current demand and inventory levels.

6. **Agility and Responsiveness**:
   - **Definition**: The ability to quickly adapt to changing conditions.
   - **Benefit**: Organizations become more data-driven and responsive to rapidly changing operational environments.
   - **Example**: Supply chain adjustments in response to sudden changes in demand or supply disruptions.

---

## Limitations and Challenges

While operational analytics offers significant benefits, it also presents challenges:

- **Limited Access to Historical Data**:
  - **Explanation**: Operational systems typically lack the storage capacity to hold extensive historical data compared to analytical systems.
  - **Impact**: Analytical workloads on the operational plane have limited access to historical context, potentially affecting the depth of analysis.

- **Data Gravity**:
  - **Definition**: The concept that large volumes of data attract applications and services, making data movement difficult.
  - **Impact**: Moving historical data from the analytical plane to the operational plane is challenging due to data gravity.

- **Scalability Limitations**:
  - **Explanation**: Operational systems may not scale as efficiently for analytical workloads, which can be resource-intensive.
  - **Impact**: Limits the complexity and scale of analytical processing that can be performed.

- **Consistency Requirements**:
  - **Explanation**: Ensuring data consistency in real-time streaming environments is critical, especially for applications where inaccuracies can have significant consequences.
  - **Impact**: Inconsistent data can lead to loss of trust and errors in decision-making.

- **Complexity in Sourcing Historical Data**:
  - **Explanation**: Building solutions to source and integrate historical data into operational analytics can be complex and resource-intensive.

---

## Trusting Streaming Data

**Frank McSherry**, a computer scientist and Chief Scientist at Materialize.io, emphasizes the importance of trust in streaming data. Trust unfolds into three key characteristics[^1]:

### Frank McSherry's Three Characteristics

1. **Responsiveness**:
   - **Definition**: The ability to access analytical data synchronously and interactively.
   - **Metrics**:
     - **Query Latency**: Time taken to execute a query.
     - **Queries Per Second (QPS)**: Number of queries the system can handle per second.
     - **Concurrency**: Number of end-users the system can support simultaneously.

2. **Freshness**:
   - **Definition**: How up-to-date the analytical results are.
   - **Explanation**: The value of data diminishes over time; thus, fresher data is more valuable.

3. **Consistency**:
   - **Definition**: Ensuring that data remains accurate and reliable across the system.
   - **Importance**: Critical for applications where inaccuracies can lead to significant negative outcomes.

### Mathematical Representation of Value Degradation Over Time

The value of data decreases as time progresses. We can model the value \( V(t) \) of data as a function of time \( t \).

Assuming an **exponential decay model**:

\[
V(t) = V_0 e^{-\lambda t}
\]

Where:

- \( V_0 \): Initial value of the data at time \( t = 0 \).
- \( \lambda \): Decay constant, representing the rate at which value decreases.
- \( t \): Time elapsed since data generation.

**Interpretation**:

- When \( t = 0 \), \( V(t) = V_0 \).
- As \( t \) increases, \( V(t) \) decreases exponentially.

**Graphical Representation**:

![Value Degradation Over Time](value_degradation_over_time.png)

*Figure 9-5: The definition of real time is based on how quickly or slowly value degrades as time progresses.*

- The "Real-Time" box contains data that hasn't lost significant value due to time.
- The scale of the x-axis (time) can vary depending on the use case (milliseconds, seconds, minutes, hours).

**Implications**:

- Systems must process and deliver data before its value diminishes significantly.
- The acceptable latency depends on how quickly the value degrades for the specific application.

---

## Alignment with Data Mesh

### Definition of Data Mesh

**Data Mesh** is a conceptual framework for data architecture introduced by **Zhamak Dehghani** in 2019. It advocates for a **decentralized approach** to data management, where data is treated as a product, and ownership is distributed among different domains or business units.

### Pillars of Data Mesh

Data Mesh comprises four key principles:

1. **Domain-Oriented Decentralized Data Ownership**:
   - **Explanation**: Data ownership is distributed among domains aligned with business units.
   - **Benefit**: Fosters autonomy and accountability within each domain.

2. **Data as a Product**:
   - **Explanation**: Data is managed with the same rigor as a product, including quality, documentation, and accessibility.
   - **Benefit**: Encourages a mindset focused on delivering high-quality, usable data.

3. **Self-Serve Data Infrastructure**:
   - **Explanation**: Infrastructure designed to empower domain teams to access and manage data independently.
   - **Benefit**: Reduces bottlenecks and increases agility.

4. **Federated Computational Governance**:
   - **Explanation**: Governance policies are applied across domains, balancing standardization with local autonomy.
   - **Benefit**: Ensures consistency and compliance while allowing flexibility.

### Operational Analytics and Data Mesh Alignment

Bringing analytical workloads closer to the operational plane aligns with Data Mesh concepts in the following ways:

1. **Domain-Oriented Decentralized Data Ownership**:
   - Operational teams own and analyze the data generated within their specific domain.
   - **Example**: A marketing team directly analyzes campaign performance data without relying on a centralized data team.

2. **Data as a Product**:
   - Operational teams manage the end-to-end lifecycle of their data, including analysis and application.
   - **Example**: A product team treats user interaction data as a product, ensuring it's high-quality and accessible for feature development.

3. **Self-Serve Data Infrastructure**:
   - Operational teams have direct access to analytical tools and platforms.
   - **Example**: Providing teams with tools like Tableau or Looker to perform their own data analyses.

4. **Federated Computational Governance**:
   - Analytical workloads adhere to common standards while allowing domain-specific practices.
   - **Example**: Data privacy policies are enforced globally, but each team can define their own data access controls within those policies.

---

## Challenges of Implementing a Data Mesh

Implementing a Data Mesh presents several challenges:

1. **Cultural Shift**:
   - **Explanation**: Transitioning from centralized control to a decentralized model requires significant change management.
   - **Impact**: Resistance from teams accustomed to centralized data management.

2. **Technical Complexity**:
   - **Explanation**: Building a self-serve data infrastructure that aligns with Data Mesh principles can be technically challenging.
   - **Impact**: Requires integration with existing systems and may involve substantial architectural changes.

3. **Organizational Silos**:
   - **Explanation**: Existing silos can hinder cross-functional collaboration necessary for Data Mesh.
   - **Impact**: May impede the sharing of data and best practices across domains.

4. **Skill Set Gaps**:
   - **Explanation**: Teams may lack the necessary skills to manage data as a product.
   - **Impact**: Requires training and development to build necessary competencies.

5. **Data Quality and Governance**:
   - **Explanation**: Balancing local autonomy with overarching standards for data quality and governance is challenging.
   - **Impact**: Risk of inconsistent data practices and potential compliance issues.

6. **Tooling and Infrastructure**:
   - **Explanation**: Advanced tooling is needed to support self-service capabilities.
   - **Impact**: May necessitate investment in new technologies and platforms.

**Quote**:

> "While promising, Data Mesh adoption may face hurdles related to expertise, necessitating advanced tooling and infrastructure for self-service capabilities."
>
> — Roland Meertens et al., *InfoQ AI, ML, and Data Engineering Trends Report: September 2023*

---

## Simplifying Data Mesh with Streaming Databases

**Streaming Databases** can help simplify the adoption and implementation of Data Mesh by:

- **Providing Familiar Interfaces**:
  - Use SQL and other familiar tools, lowering the learning curve.
  - **Example**: Materialize allows SQL queries over streaming data.

- **Enabling Data Accessibility**:
  - Make data products accessible to multiple domains globally.
  - **Example**: Real-time data streams can be subscribed to by different teams as needed.

- **Accelerating Iterative Development**:
  - Support rapid development and refinement of data solutions.
  - **Example**: Quick deployment of data transformations without heavy infrastructure overhead.

- **Facilitating Data Productization**:
  - Treat data streams as products that can be consumed and reused.
  - **Example**: A sales team's data stream of customer interactions can be productized for use by the marketing team.

---

## Summary

- **Operational Analytics** involves performing analytical workloads near the source of data generation, leveraging the streaming plane.
- **Benefits** include real-time decision-making, enhanced efficiency, improved customer experience, and cost savings.
- **Challenges** involve limitations in historical data access, data gravity, scalability, and consistency requirements.
- **Trust in Streaming Data** is built on responsiveness, freshness, and consistency.
- **Data Mesh** promotes decentralized data ownership, treating data as a product, self-serve infrastructure, and federated governance.
- **Alignment with Operational Analytics** is evident as both approaches promote decentralization and autonomy.
- **Challenges in Data Mesh** implementation are significant but can be mitigated with the use of streaming databases, which simplify adoption.

---

## References

1. **Zhamak Dehghani**, "How to Move Beyond a Monolithic Data Lake to a Distributed Data Mesh" — *Martin Fowler's blog*, May 2019.
2. **Frank McSherry**, discussions on trusting streaming data and Materialize.io.
3. **Roland Meertens et al.**, "InfoQ AI, ML, and Data Engineering Trends Report: September 2023".

---

## Tags

#OperationalAnalytics #DataMesh #StreamingDatabases #RealTimeAnalytics #DataEngineering #StaffPlusNotes #DataGravity #Decentralization #DataGovernance #DataProducts

---

[^1]: The three characteristics are from Frank McSherry's discussions on trusting streaming data.

---

Feel free to reach out if you have any questions or need further clarification on any of these topics.