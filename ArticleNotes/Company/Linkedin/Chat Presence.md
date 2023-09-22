# **Obsidian Notes: Chat Presence at LinkedIn**

## **Introduction**
Green online indicators, often referred to as “presence indicators,” are common in instant messaging applications and social networks. They show the online status of users. On LinkedIn, a platform with approximately 500 million members, displaying these indicators in real time presents unique challenges.

## **Tools and Technologies Used**
- Play Framework
- Akka Actor Model

## **Problems & Solutions**

### **Problem 1: Connection Determination**
- **Issue**: How to know when a member is connected to LinkedIn?
- **Solution**:
  - Leveraged LinkedIn’s **Real-time delivery platform** designed for Instant Messaging.
  - This platform is a **publish/subscribe system** that streams data from server to mobile or web clients over a persistent connection as events occur.
  
    ```java
    if (userConnects()) {
      establishPersistentConnection(RealTimePlatform);
    }
    ```

  - **Real Life Scenario**: When Alice connects to LinkedIn, a persistent connection indicates she's online. If she wishes to view Bob's online status, she subscribes to a topic for Bob's presence status. If Bob opens his app, the Presence Platform detects and publishes an online event. Alice sees Bob's indicator turn green.

### **Problem 2: Jittery Connections**
- **Issue**: Members with mobile devices often face connection interruptions due to unstable networks.
- **Solution**:
  - Use **periodic heartbeats** from the Real-time Platform.
  
    ```python
    def sendHeartbeat(memberID):
        while connectionExists(memberID):
            emitHeartbeat(memberID)
            sleep(d_seconds)
    ```

  - If a connection is active, the Real-time Platform emits a heartbeat with the member's ID at fixed intervals `(d seconds)`. The Presence Platform considers a member online if a heartbeat is received every d seconds.
  - Temporary disconnections less than d seconds won't change the online status.

### **Platform Architecture**

#### **1. Handling Heartbeats**
- When LinkedIn is opened, a persistent connection is established with the Real-time Platform. This platform starts emitting heartbeats for that user.
- Presence Platform processes these heartbeats.

    ```scala
    for (heartbeat in heartbeats) {
      if (noEntry(heartbeat.memberID) || entryExpired(heartbeat.memberID)) {
          publishOnlineEvent(heartbeat.memberID);
          addToStore(heartbeat.memberID, d + ε);
      } else {
          updateLastHeartbeatTimestamp(heartbeat.memberID);
      }
    }
    ```

- Each heartbeat checks a distributed K/V store for Presence for an unexpired entry.
  - If no entry exists, the member is considered online, an online event is published, and an entry is added to the store.
  - If an entry exists, update the timestamp.

#### **2. Offline Detection**
- **Challenge**: Direct offline detection is tough as users don't necessarily send a "goodbye" signal.
- **Strategy**: Background jobs scan the K/V store for entries not updated beyond `(d + ε)` duration, signaling that a user might be offline.

