#### Overview
- Focus on Edge Architecture in system design.
- Inspired by Netflix and Dropbox implementations.
- Sits at the intersection of software engineering, infrastructure, and network.
#### Key Concepts
- **Edge Architecture**: Crucial for balancing load and routing in large-scale web applications.
- **Single Region Design**: A basic design with a single point of access and potential availability issues.
- **Multi-Region with GeoDNS**: Enhances availability and load balancing across different geographic locations.
- **Hybrid Anycast GeoDNS Approach**: Combines benefits of unicast and anycast IPs for efficient routing and failover.
#### Architecture and Components
1. **Single Region Design**:
   - Simplest form, with everything deployed in one region.
   - Uses a public DNS record (e.g., `api.company.com`).
   - Involves a load balancer and an edge proxy (API Gateway).
   - **API Gateway**: Handles routing, authentication, authorization, rate limiting, and A/B testing.
2. **Multi-Region with GeoDNS**:
   - Edge stack deployed across multiple regions.
   - Public IP from each site is added to DNS.
   - GeoDNS routes requests based on the closest location.
   - Improves reliability but has limitations due to DNS TTL and stale data.
![[Screenshot 2024-01-25 at 3.49.32 PM.png]]
1. **Hybrid Anycast GeoDNS Approach**:
   - Uses both unicast and anycast IPs.
   - Allows immediate fallback and efficient routing via BGP (Border Gateway Protocol).
   - Suitable for organizations with a private backbone network or cloud service.
![[Screenshot 2024-01-25 at 3.48.36 PM.png]]
#### Challenges and Solutions
- **DNS Caching Issue**: With multi-region design, even with a low TTL, DNS caching can delay traffic rerouting.
- **Site Outage Handling**: Anycast allows quick rerouting of traffic in case of site outages without waiting for DNS TTL to expire.
- **Scalability with Anycast**: Handling a large number of IPs can be challenging.

#### Best Practices
- **Choosing the Right Design**: Based on availability requirements, traffic distribution, and organizational infrastructure.
- **Scalability and Maintenance**: Regularly review and update routing protocols and IP management strategies.
- **Security Considerations**: Implement robust security measures at the API Gateway and load balancer levels.

#### Case Studies: Netflix and Dropbox
- Implementations show practical applications of these architectures.
- Real-world challenges and solutions can be gleaned from these examples.

#### Interview Preparation
- Understand the differences between single region, multi-region, and hybrid designs.
- Be ready to discuss real-world scenarios of DNS caching and traffic routing.
- Know the implications of using API Gateways in large-scale systems.

#### Further Reading
- Detailed explanations of Edge Design: [crashingtechinterview.com](https://crashingtechinterview.com)
- BGP and its role in internet routing.

#### Appendix
- **Glossary**:
  - **GeoDNS**: Geographic DNS, routes requests based on geographic location.
  - **API Gateway**: A server that acts as an API front-end, receiving API requests and handling them.
  - **BGP (Border Gateway Protocol)**: A protocol to route internet traffic efficiently.
- **Diagrams and Slides**: For a visual representation of the architectures discussed, refer to provided resources.

### Creating Obsidian Links and Tags
- Link to related topics like [[Load Balancing]], [[DNS Caching]], [[BGP in System Design]].
- Use tags like #EdgeArchitecture, #APIDesign, #Netflix, #Dropbox for easy retrieval and cross-referencing.