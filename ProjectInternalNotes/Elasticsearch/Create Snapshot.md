
This function is responsible for creating a new snapshot in Elasticsearch.

Here's a high-level summary of what each part of the function does, followed by detailed descriptions and examples:

1. It first ensures that the repository exists and the snapshot name is not already in use.
2. It separates the requested indices into system indices and non-system indices.
3. It checks if the global state or feature states are requested. If so, it processes these requests.
4. Then, it checks if any feature states are requested to be included in the snapshot and if the feature is associated with any system indices. If yes, these indices are added to the list of indices to be included in the snapshot.
5. Next, it creates a mapping from the index name to `IndexId` objects, which represent the metadata for the indices.
6. It computes the minimum version compatible with all nodes in the cluster, the repository data, and the snapshot to be created.
7. Then, it retrieves the current status of each shard in the indices.
8. It checks if the snapshot request is not a partial snapshot. If not, it checks if all indices have a primary shard. If an index doesn't have a primary shard, it throws an exception.
9. Finally, it creates a `SnapshotsInProgress` object, which represents the status of all ongoing snapshot operations.

Now let's dive into the details of the function:

The function `createSnapshot` receives four parameters: a `CreateSnapshotTask` object which contains the data necessary to create the snapshot, a `TaskContext` object for managing the task's lifecycle, the current cluster state, and an object representing the snapshots that are currently in progress.

**Validation**

It starts by extracting the repository and snapshot data from the `CreateSnapshotTask` object. It then checks if the repository exists and if the snapshot name is available, i.e., it's not already in use in the repository or currently running.

**Indices and Feature States**

Next, the function categorizes the requested indices into system indices and non-system indices. System indices are internal Elasticsearch indices used by plugins and core Elasticsearch modules to store configuration and metadata, whereas non-system indices are usually the indices you create to store your data.

The function then checks if the global state or any specific feature states are requested to be included in the snapshot. Feature states represent the state of an Elasticsearch feature, such as indices, ingest pipelines, ILM policies, etc., that belong to a particular system feature.

If feature states are requested, the function processes these requests. For instance, if the "none" feature state is requested along with other feature states, it throws an exception because "none" implies that no feature state should be included.

**Adding Feature-related Indices**

The function then checks if any of the requested feature states are associated with any system indices. If yes, these indices are added to the list of indices to be included in the snapshot.

**Preparing for Snapshot**

Next, the function creates a mapping from the index name to `IndexId` objects, which represent the metadata for the indices.

Then, it computes the minimum version that is compatible with all nodes in the cluster, the repository data, and the snapshot to be created. This is necessary because different versions of Elasticsearch may have different features, and a snapshot created on a newer version might not be compatible with older versions.

The function retrieves the current status of each shard in the indices, which will later be included in the snapshot. The status of a shard can be one of the following: `INIT`, `STARTED`, `FINALIZING`, `DONE`, `FAILED`, or `MISSING`.

If the snapshot request is not a partial snapshot, the function checks if all indices have a primary shard. If an index doesn't have a primary

 shard, it throws an exception. This is because a snapshot operation requires a primary shard to take a snapshot of the index.

**Starting the Snapshot**

Finally, it creates a `SnapshotsInProgress` object, which represents the status of all ongoing snapshot operations, adds the snapshot to this object, and sets the task to succeed. If the state of the snapshot is completed, it ends the snapshot.

Note: This function is part of Elasticsearch's internal code and is subject to change in future versions. Make sure to refer to the official documentation and the specific version's source code for the most accurate information.
