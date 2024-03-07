
https://juejin.cn/post/7222995333303287845?searchId=2023071508383838FA16B4EC10E6962405
---
#### I. Initial Steps and Identification
- **Check High CPU Threads**: 
  - Command: `top -Hp <pid>`.
  - Identify and note down the thread IDs with high CPU usage.

- **View Stack Information**:
  - Convert TID to hexadecimal: `printf '%x\n' <tid>`.
  - Retrieve stack information: `jstack <pid> | grep <tid> -A 50`.

#### II. Analyzing Kafka Consumer Behavior
- **Stack Trace Analysis**: 
  - Indications that Kafka consumer message processing is causing high CPU usage.
  - High CPU usage typically linked to processing a large volume of messages.

- **Message Lag Investigation**:
  - Check for unusual message lag, particularly persistent negative values, which can indicate issues.

#### III. Investigating Further
- **Reducing Consumer Count**: 
  - Initially consider lowering the number of consumers as a potential solution.

- **Community and Research Insights**:
  - Explore solutions from Kafka’s GitHub and community forums.
  - Consider upgrading the client version as a common recommendation.

- **Flame Graph Analysis**:
  - Generate and analyze a flame graph to pinpoint methods consuming CPU.
  - Review `poll` method usage and other network request processing.

#### IV. Deep Dive into Kafka Internals and HW Equation
- **HW (High Watermark) and Negative Lag**:
  - **HW Equation**: `lag = HW - consumerOffset`.
    - `HW`: High watermark, typically equals the smallest Log End Offset (LEO) among all ISRs.
    - `consumerOffset`: The offset acknowledged by the consumer.
  - **Negative Lag**: Persistent negative lag suggests `consumerOffset` is greater than `HW`, possibly due to HW not being updated.

- **Broker and Consumer Group Analysis**:
  - Investigate the health and status of consumer groups and brokers.
  - Find brokers where replicas are not part of ISR, indicating potential issues.

- **Polling Frequency and Broker Response**:
  - Unusual increase in `poll` frequency can lead to high CPU usage.
  - Network packet captures and broker responses (e.g., "Not Leader For Partition") can give further clues.

#### V. Resolution Strategy and Summary
- **Resolving in Test Environment**:
  - In test environments, consider deleting and recreating the topic as a direct solution.

- **Key Takeaways and Recommendations**:
  - Utilize network packet capturing for insightful analysis.
  - In production environments, handle broker issues cautiously, possibly starting from the Kafka controller.
  - Documentation and regular monitoring are crucial for early detection and efficient troubleshooting.

#### VI. Additional Considerations
- **Potential Causes and Mitigations**:
  - Explore other factors like network issues, client configuration, or broker performance.
  - Continuous observation and tweaking of Kafka configurations based on system behavior.

---
To perform a packet capture of network traffic from a Kafka consumer process on Linux, you can use tools like `tcpdump`, `Wireshark`, or `tshark`. Here's a general approach to capturing network packets for a Kafka consumer:

### 1. Identify Network Interface and Kafka Consumer IP/Port
- **Network Interface**: Find out the network interface through which the Kafka consumer communicates. You can use the `ip a` command to list all network interfaces.
- **Consumer IP/Port**: Identify the IP address and port number used by the Kafka consumer. This can be found in the Kafka consumer configuration.

### 2. Using tcpdump
- **Basic Capture**: Run `tcpdump` on the identified interface. For example: 
  ```
  sudo tcpdump -i <interface-name> -w kafka_capture.pcap
  ```
  Replace `<interface-name>` with your actual network interface name.
  
- **Filter for Kafka Consumer**: Apply a filter to capture traffic specific to the Kafka consumer's IP and port. For example:
  ```
  sudo tcpdump -i <interface-name> host <kafka-consumer-ip> and port <kafka-port> -w kafka_capture.pcap
  ```

### 3. Using Wireshark or tshark
- **Wireshark**: This graphical tool can be used if you have a GUI environment. Run Wireshark, select the appropriate network interface, and apply a filter for the Kafka consumer's IP and port.
- **tshark**: It's the command-line version of Wireshark. The command for capturing would be similar to `tcpdump`:
  ```
  tshark -i <interface-name> -f "host <kafka-consumer-ip> and port <kafka-port>" -w kafka_capture.pcap
  ```

### 4. Analyzing the Captured Data
- The `.pcap` file generated can be analyzed using Wireshark or similar tools to inspect the packets. Look for patterns or anomalies in the communication between your Kafka consumer and the Kafka cluster.

### 5. Security and Permissions
- Ensure you have the necessary permissions to capture network traffic.
- Be aware of security and privacy concerns when capturing network traffic, especially in a production environment.

### 6. Advanced Filtering
- If needed, you can apply more advanced filters to capture specific types of traffic or to exclude irrelevant data.

### 7. Continuous Capture
- For a continuous capture, especially in a production environment, consider tools like `tcpdump` with rotation (`-G` option for file rotation based on time) or a dedicated network monitoring solution.

Remember to replace placeholder values like `<interface-name>`, `<kafka-consumer-ip>`, and `<kafka-port>` with actual values relevant to your Kafka consumer setup.