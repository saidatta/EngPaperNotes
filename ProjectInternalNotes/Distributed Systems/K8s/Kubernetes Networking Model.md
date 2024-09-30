https://github.com/guevara/read-it-later/issues/6041
---
### Introduction
Kubernetes is designed to manage distributed systems, and networking is a central piece of this puzzle. The **Kubernetes Networking Model** enables communication between Pods, Services, and external systems. Understanding this model is crucial for ensuring that applications deployed on Kubernetes are correctly configured, monitored, and scalable. This guide provides an in-depth analysis of Kubernetes networking, focusing on the following key areas:
- **Container-to-Container Networking**
- **Pod-to-Pod Networking**
- **Pod-to-Service Networking**
- **Internet-to-Service Networking**
Each of these topics is explored in terms of how traffic flows within Kubernetes, the technologies used, and the challenges in implementing the network model.
---
### 1. Kubernetes Basics

#### 1.1 Kubernetes API Server
- The **kube-apiserver** is the central point of control for Kubernetes. It handles all API requests for managing the cluster.
- All Kubernetes components, including Pods, Services, and Nodes, interact with the API server.
- Behind the API server, **etcd** stores the desired state of the cluster.

#### 1.2 Controllers
- **Controllers** maintain the desired state by continuously monitoring the cluster’s actual state and reconciling any differences.
- Example: The **kube-scheduler** places new Pods on Nodes, while **kubelet** configures the networking for these Pods.

#### 1.3 Pods
- **Pods** are the smallest unit of deployment and encapsulate one or more containers. They share the same network namespace and IP address.
  
#### 1.4 Nodes
- **Nodes** can be physical machines or virtual machines that run Pods. Each Node is part of the larger Kubernetes cluster.

---

### 2. The Kubernetes Networking Model

The Kubernetes networking model imposes several requirements that must be satisfied by the underlying network infrastructure:
1. **Pod-to-Pod communication** must not require **Network Address Translation (NAT)**.
2. **Node-to-Pod communication** must not require NAT.
3. The **Pod IP** should be the same internally and externally (i.e., the same IP seen by the Pod itself is visible to other Pods).

These constraints create several networking challenges, including managing **Container-to-Container**, **Pod-to-Pod**, **Pod-to-Service**, and **Internet-to-Service** networking.

---

### 3. Container-to-Container Networking
Each Pod in Kubernetes operates within its own **network namespace**, a Linux construct that provides a separate networking stack (including IP routing tables, firewall rules, and network devices). Within a Pod, containers share the same network namespace and IP address, which allows them to communicate with each other via `localhost`.
#### Example:
If a Pod has two containers (Container A and Container B), both can communicate with each other over `localhost:<port>` because they share the same network namespace.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
  - name: container-a
    image: nginx
  - name: container-b
    image: redis
```

In this example, `container-a` and `container-b` can communicate over the loopback interface (`localhost`), as they share the same network stack.

---
### 4. Pod-to-Pod Networking

Each Pod has its own **unique IP address**, and Pods can communicate with other Pods directly using these IPs, whether they are on the same or different Nodes.

#### 4.1 Networking on the Same Node

Kubernetes uses **virtual Ethernet pairs (veth)** to connect Pod namespaces on the same Node. A **veth pair** acts as a virtual cable connecting two network namespaces (one end in the Pod's network namespace, and the other in the root namespace).

##### Example: veth pair setup
- **Pod A** is connected to **veth0**, which connects to the **root network namespace** via a **bridge**.
- **Pod B** is connected to **veth1**, which is also attached to the same bridge in the root namespace.

```plaintext
Pod A (eth0) --> veth0 --> root namespace (bridge) --> veth1 --> Pod B (eth0)
```

Packets from Pod A are sent via **eth0**, which are routed to **veth0**, then forwarded through the bridge, and finally delivered to **veth1** (and vice versa).

---

#### 4.2 Networking Across Nodes

To route packets between Pods on different Nodes, Kubernetes assigns each Node a **CIDR block** (Classless Inter-Domain Routing) that defines the IP addresses available to Pods on that Node. When traffic needs to be routed across Nodes, Kubernetes relies on **underlying network configurations** and routing protocols to forward packets correctly.

##### Example: Cross-Node Communication
1. **Pod A** on Node 1 sends a packet to **Pod B** on Node 2.
2. The packet is routed from Pod A’s `eth0` interface to Node 1’s **root namespace** via **veth**.
3. The **bridge** in Node 1 forwards the packet to the network interface.
4. The packet is routed across the network to Node 2 based on the **CIDR block** assigned to Node 2.
5. Node 2’s root namespace receives the packet and forwards it to **Pod B** via the **veth pair**.

```plaintext
Node 1 (Pod A) --> Node 1 (root namespace) --> Network --> Node 2 (root namespace) --> Node 2 (Pod B)
```

---

### 5. Pod-to-Service Networking

In Kubernetes, **Services** abstract a group of Pods and provide a stable IP (ClusterIP) for clients to communicate with, even if the Pods behind the Service are replaced or rescheduled.

#### 5.1 How Services Work:
- A **ClusterIP** is assigned to each Service.
- Traffic to the ClusterIP is load-balanced across the Pods backing the Service.
- **kube-proxy** sets up **iptables** or **IPVS** rules to handle this load balancing.

##### Example: Traffic Flow for a Pod-to-Service Packet
1. A packet from **Pod A** destined for a Service IP reaches the Node's root namespace.
2. **iptables** rules rewrite the destination IP from the Service IP to a specific Pod’s IP backing the Service.
3. The packet is then routed to the Pod via **Pod-to-Pod networking**.

```plaintext
Pod A --> Service IP (ClusterIP) --> iptables --> Pod B (actual Pod IP)
```

---

#### 5.2 IPVS for Service Load Balancing

**IPVS (IP Virtual Server)** offers an alternative to **iptables** for load balancing Services. It’s more scalable and faster because it uses hash tables to store routing rules instead of linear lists, allowing faster lookups.

Key advantages of IPVS:
- **Faster performance**: Supports millions of concurrent connections.
- **Scalability**: Suitable for large-scale deployments.

---

### 6. Internet-to-Service Networking

Kubernetes provides **LoadBalancers** and **Ingress Controllers** to expose internal services to external clients.

#### 6.1 Egress: Node to Internet

When Pods communicate with external systems, Kubernetes uses **NAT (Network Address Translation)** to rewrite the Pod’s IP to the Node’s IP before routing it to the internet. This is necessary because public networks typically do not recognize Pod IPs.

**Steps for Egress Traffic**:
1. A packet from the Pod travels through the Node’s root namespace.
2. **iptables** rewrites the Pod’s IP to the Node’s IP.
3. The packet is routed to the **Internet Gateway** and reaches its destination.

---

#### 6.2 Ingress: Internet to Node

Kubernetes provides two mechanisms to expose services to external clients:
1. **Layer 4 Ingress (LoadBalancers)**: Works at the transport layer (TCP/UDP).
2. **Layer 7 Ingress (Ingress Controllers)**: Operates at the application layer (HTTP/HTTPS), allowing traffic to be routed based on URL paths.

##### Example: Layer 7 Ingress (HTTP Load Balancer)
- An **Ingress Controller** sets up rules to route external HTTP requests to specific Services based on the request’s URL path.
- This allows for more granular routing, like directing `/api` requests to a specific Service.

```plaintext
External Client --> Ingress (URL-based routing) --> Service
```

---

### 7. Wrapping Up

Kubernetes networking is a complex but powerful abstraction that makes deploying and scaling distributed applications easier. The key elements like **veth pairs**, **bridges**, **iptables**, and **IPVS** form the foundation for Pod-to-Pod, Pod-to-Service, and Internet-to-Service networking. Understanding these concepts is crucial for designing, debugging, and scaling applications running on Kubernetes.

---

### Key Glossary Terms

- **NAT (Network Address Translation)**: Rewrites the IP address of packets as they traverse a network.
- **veth (Virtual Ethernet Device)**: A pair of virtual network interfaces used to connect namespaces.
- **iptables**: A user-space utility for configuring the packet filtering rules in the Linux kernel.
- **IPVS (IP Virtual Server)**: An alternative to iptables for handling large-scale load balancing using hash tables.
- **Ingress**: A Kubernetes object that routes HTTP/HTTPS traffic to Services based on rules, such as URL paths.

---

This guide should serve as a detailed reference for Kubernetes networking, covering everything from basic concepts to advanced routing mechanisms used within a cluster and for connecting services to the outside world. Understanding these fundamentals is essential for ensuring that Kubernetes deployments are scalable, efficient, and fault-tolerant.