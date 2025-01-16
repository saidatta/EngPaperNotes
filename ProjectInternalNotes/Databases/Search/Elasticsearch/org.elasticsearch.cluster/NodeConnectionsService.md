
Here is a class diagram representing the `NodeConnectionsService` and `ConnectionTarget` classes:

![Class Diagram](https://showme.redstarplugin.com/d/UnaMjTOF)

[You can edit this diagram online if you want to make any changes.](https://showme.redstarplugin.com/s/Ktb5cnyq)

### NodeConnectionsService
Responsible for maintaining connections from the current node to all other nodes listed in the cluster state. It also disconnects from nodes once they are removed from the cluster state. It periodically checks that all connections are still open and restores them if needed.

Key methods include:
- `connectToNodes(DiscoveryNodes, Runnable)`: Connects to all the given nodes, but does not disconnect from any extra nodes. Calls the completion handler on completion of all connection attempts to _new_ nodes.
- `disconnectFromNodesExcept(DiscoveryNodes)`: Disconnects from any nodes to which the service is currently connected which do not appear in the given nodes.
- `ensureConnections(Runnable)`: Makes a single attempt to reconnect to any nodes which are disconnected but should be connected.
- `reconnectToNodes(DiscoveryNodes, Runnable)`: For disruption tests, re-establishes any disrupted connections.

### ConnectionTarget

This class represents a target for a connection. It holds the `DiscoveryNode` to which it connects, an `AtomicInteger` for the consecutive failure count, an `AtomicReference<Releasable>` for the connection reference, and a `List<Releasable>` for pending references.

Key methods include:

- `connect(Releasable)`: Returns a `Runnable` that registers a reference and initiates a connection.
- `doConnect()`: Initiates a connection to the target node.
- `disconnect()`: Disconnects from the target node.

The `NodeConnectionsService` maintains a `ConnectionTarget` for each node in the cluster state.