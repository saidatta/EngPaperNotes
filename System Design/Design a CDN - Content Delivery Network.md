https://www.youtube.com/watch?v=1GdOkYB43UI
#### Overview
- Discussion on Content Delivery Networks (CDNs), a crucial component in modern web architecture.
- Inspired by implementations in companies like Akamai and Cloudflare.
- Focuses on high-level architecture, traffic routing models, failover, and scaling mechanisms.
#### Key Concepts
- **CDN**: A network of geographically distributed proxy servers.
- **Proxy Server**: Acts as an intermediary between a client and the origin server, positioned close to end-users for reduced latency and bandwidth saving.
- **Core Functions**: CDN optimizes traffic routing, offers DDOS protection, and manages network behaviors.
#### Problems Addressed by CDN
1. **High Latency**: Especially relevant when the service is deployed far from the user (e.g., a US-based service used in Asia).
2. **Data-Intensive Applications**: Issues arise due to multiple ISPs being in the path, which can lead to congestion and packet loss.
#### High-Level Component Design and Workflow
![[Screenshot 2024-01-25 at 4.07.03 PM.png]]
1. **Routing System**: Directs clients to optimal CDN locations based on various factors like user location, content location, and data center load.
2. **Scrubber Servers**: Differentiate between good and malicious traffic, particularly during DDOS attacks. They are advanced enough to implement real-time firewall rules.
3. **Proxy and Edge Proxy Servers**: Cache content and serve it, usually from RAM for quick access.
4. **Content Distribution System**: Manages the distribution of content across proxy servers, using a tree-like model.
5. **Origin Servers**: Host the original content distributed via CDN.
6. **Data Control System**: Monitors resource usage and provides data for optimal routing.
#### Main Workflow
1. **Origin Server** delegates content for specific DNS domains to CDN.
2. **Content Distribution**: Uses push and pull models to distribute content to edge proxy servers.
3. **Client Requests**: Obtains the nearest proxy server IP or uses anycast IP for routing.
4. **Edge Proxy Servers**: Serve client requests and provide health information to the data control system.
#### Distribution Model
- **Tree-Like Replication**: Minimizes data transfers, replicating content once per region via CDN's internal network.
![[Screenshot 2024-01-25 at 4.11.38 PM.png]]
#### Routing Models
1. **DNS with Load Balancing**: Historically popular; uses DNS redirection to guide clients to specific CDN locations.
	1. ![[Screenshot 2024-01-25 at 4.12.03 PM.png]]
2. **Anycast**: Shares a single IP address across multiple edge server locations, leveraging BGP for routing.
	1. ![[Screenshot 2024-01-25 at 4.12.23 PM.png]]

#### DNS and Anycast Comparison
- **DNS with Load Balancers**: Uses unicast IPs, effective but has limitations during outages.
- **Anycast**: Increases network resilience, automatically reroutes traffic during data center outages, and provides enhanced DDOS protection.

![[Screenshot 2024-01-25 at 4.12.37 PM.png]]
#### Real-World Example: Cloudflare
- Built on a global proxy network, claiming proximity to a large portion of the internet-connected population.
- Utilizes anycast IPs to handle high traffic volumes and large-scale DDOS attacks.
#### Best Practices for CDN Design
- **Choosing Routing Model**: Based on traffic patterns, geographic distribution of users, and resilience requirements.
- **Scalability and Maintenance**: Regularly update routing protocols and content distribution strategies.
- **Security Measures**: Implement robust security protocols at both proxy and edge proxy levels.

#### Interview Preparation
- Understand the differences between DNS-based and anycast routing models.
- Discuss scenarios involving CDN use for latency reduction and handling DDOS attacks.
- Explore the implications of CDN in high-traffic environments.

#### Further Reading
- In-depth CDN design: [crashingtechinterview.com](https://crashingtechinterview.com)
- Evolution of CDN technologies and their impact on internet infrastructure.

#### Appendix
- **Glossary**:
  - **Proxy Server**: Intermediary server between client and origin.
  - **Anycast IP**: A single IP used across multiple locations.
  - **BGP (Border Gateway Protocol)**: Protocol for routing internet traffic.
- **Diagrams and Slides**: Visual representations of CDN architectures and workflows.

### Creating Obsidian Links and Tags
- Link to related topics like [[Latency Reduction in Web Services]], [[DDOS Protection Strategies]], [[Internet Traffic Routing]].
- Use tags like #CDN, #TrafficOptimization, #Cloudflare, #Akamai for easy retrieval and cross-referencing.