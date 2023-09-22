
### Overview
F4, Facebook's Warm Binary Large OBjects (BLOB) Storage System. A BLOB refers to data items in a binary format like images, videos, and documents uploaded by users.
![[Screenshot 2023-05-30 at 4.02.13 PM.png]]
In Facebook's context, BLOBs are typically created once, read numerous times, rarely modified, and occasionally deleted. To better manage these data and to meet the increasing volume of data over time, Facebook developed F4 as a warm BLOB storage system complementing Haystack, its hot storage system.

Here's a brief overview of how F4 works in conjunction with Haystack:
1.  **Understanding Hot and Cold Data Distribution**: Facebook noticed that access frequency for BLOBs correlates with their creation time, similar to a long tail distribution. The most frequently accessed BLOBs are considered 'hot' while less frequently accessed BLOBs are considered 'warm'.
2.  **Hot and Warm Access Strategy**: Facebook decided to use Haystack as a hot storage system to handle frequently accessed traffic and F4 to handle less frequently accessed BLOB traffic. Since F4 only stores data that doesn't change much and is accessed relatively less, its design can be greatly simplified.
3.  **Routing Layer**: To effectively use Haystack and F4, a dedicated routing layer was created. This layer makes decisions and routes the data requests to either Haystack or F4 as needed.
4.  **Updates to Haystack**: Over time, Haystack has undergone updates to make it more efficient. The 'Flag' bit was removed, and a journal file was added to record deleted BLOB entries, alongside the existing data file and index file.
5.  **Design Purpose of F4**: The main purpose of F4 is to minimize the effective replication factor while ensuring fault tolerance. This is to handle the growing demand for accessing warm data. Furthermore, F4 is designed to be modular and scalable. As the data volume grows, more machines can be added to F4 seamlessly to handle the increased load.

_**For F4, the main design purpose is to reduce the effective redundancy factor**_ ( _effective-replication-factor_ ) as much as possible under the premise of ensuring fault tolerance , so as to cope with the growing demand for _**warm data access.**_ In addition, it is more modular and has better scalability, that is, it can be smoothly expanded by adding machines to cope with the continuous growth of data.

---

### Data Level
As of 2014, Facebook hosted more than 400 billion images. The access frequency for these images shows a heat map distribution, where the creation time significantly affects changes. For example, data created less than a day ago is accessed about 100 times more frequently than data created a year ago. 

The threshold distinguishing between 'hot' and 'warm' data is approximately one month for most BLOBs. User profile pictures remain as 'hot' data, while ordinary pictures use a three-month threshold. 

### Overall Architecture
The principle behind Facebook's storage system is simplicity, cohesiveness, and high suitability. The system handles creation, deletion, and reading of data:

- Creation (C1-C2) is managed by Haystack
- Deletion (D1-D2) is mainly managed by Haystack, with a minor portion handled by F4
- Reading (R1-R4) is shared between Haystack and F4

### Volume and File Organization
BLOBs are aggregated into logical volumes to minimize meta-information and reduce IO numbers. Each volume is designed to have a capacity of approximately 100GB. A volume is in an 'unlocked state' until full, then it becomes a 'locked state', allowing only reading and deletion.

Each volume contains three files:

1. Data file: Records the BLOB itself and its meta-information.
2. Index file: A snapshot of the search structure in memory.
3. Journal file: Newly added, it records all deleted BLOBs.

In the unlocked state, all three files are readable and writable. In the locked state, only the journal file can be read and written; the other two files become read-only.

### Controller Module
The controller module coordinates the entire system:

- Provides new storage machines
- Maintains a pool of unlocked volumes
- Ensures all logical volumes have enough physical volumes for backup
- Creates physical volumes as needed
- Performs periodic maintenance tasks such as data compaction and garbage collection

---

### Code Example

While there's no directly associated code example for the concepts above (as they're architectural components of Facebook's backend), consider the following pseudocode illustrating a read operation from the perspective of the controller module:

```python
def read_data(blob_id):
    # Determine whether the data is 'hot' or 'warm'
    data_type = determine_data_type(blob_id)
    
    if data_type == 'hot':
        # Delegate to Haystack for hot data
        data = haystack.read_data(blob_id)
    else:
        # Delegate to F4 for warm data
        data = f4.read_data(blob_id)
    
    return data
```

In this pseudocode, the `read_data` function checks whether the requested data is 'hot' or 'warm', then delegates the read operation to either Haystack or F4 depending on the data type.

Absolutely, let's proceed with more detailed notes on the additional components of Facebook's BLOB storage architecture:

---

### Route Tier
The routing layer serves as the external interface of the BLOB storage system, effectively shielding the underlying system's implementation and making it possible to add subsystems like F4 seamlessly. This tier stores all state data (like mapping of logical volumes to physical volumes) in a separate database, enabling smooth scalability.

- Read requests: The routing module parses the logical volume id from the BLOB id and retrieves corresponding physical volume information from the database. It typically fetches data from the nearest host, resorting to the next host in case of a failure or timeout event.
- Creation requests: The routing module selects a logical volume with free space, sending the BLOB for writing to all corresponding physical volumes. If any issue arises, the write operation is aborted, the data is discarded, and the process is repeated with a different logical volume.
- Deletion requests: The routing module sends the request to all corresponding physical volumes for asynchronous deletion. It continually retries until successful deletion of the corresponding BLOB from all volumes.

### Transformer Tier
The transformation layer handles operations on retrieved BLOB data such as image scaling and cropping. In previous versions, these computation-heavy tasks were conducted on storage nodes, causing bottlenecks. The transformation layer now enables storage nodes to focus on their primary role — storage services — and permits separate scaling of storage and transformation layers. 

### Caching Stack
The caching stack was initially designed to handle requests for hot BLOB data and reduce the load on the backend storage system. This also holds true for warm storage. The caching layer includes content delivery networks (CDNs) and caches provided by content distributors.

---

## Facebook BLOB Storage: Haystack Hot Storage

Haystack, designed to maximize Input/Output Operations Per Second (IOPS), handles all creation requests, most deletion requests, and high-frequency read requests, thus simplifying the design of warm storage.

- Read requests: Haystack retrieves the metadata of the requested BLOB from memory, checks whether it has been deleted, and fetches the corresponding BLOB data using only one I/O operation via the physical file position, offset, and size.
- Creation requests: Haystack appends the BLOB data to the data file, updates the meta-information in memory, and writes changes into the index file and the journal file.
- Deletion requests: Haystack updates the index and journal files. The corresponding data still exists in the data file until a periodical compaction operation deletes the data and reclaims the corresponding space.

---

### Code Example

Again, the concepts presented do not easily lend themselves to code examples. However, for illustrative purposes, consider this pseudocode for a deletion request:

```python
def delete_data(blob_id):
    # Get the logical volume for the BLOB
    logical_volume = get_logical_volume(blob_id)
    
    # Get the physical volumes for the logical volume
    physical_volumes = get_physical_volumes(logical_volume)
    
    # Send the deletion request to all physical volumes
    for volume in physical_volumes:
        volume.delete_data(blob_id)
```

In this pseudocode, `delete_data` sends a deletion request to all corresponding physical volumes for a given BLOB.

---
### Fault Tolerance
Haystack ensures fault tolerance at various levels: hard drives, hosts, racks, and data centers. It replicates each copy across racks in a data center and places another copy in a different data center. This approach, in conjunction with RAID-6 for extra hard disk fault tolerance, results in an effective redundancy factor of 3.6x. This redundancy contributes to the system's high level of fault tolerance, even though it implies increased storage usage.

### Expiry-Driven Content
Certain types of BLOBs have an expiration time, after which they need to be deleted. An example of this would be user-uploaded videos, which are converted from their original format to the storage format, with the original video deleted afterward. Haystack handles these frequent deletion requests, enabling space reclamation through regular compactions, thus eliminating the need to move such data to f4.

### f4 Design
The f4 subsystem is designed to maintain fault tolerance with efficiency. It ensures resilience against hard disk errors, host failures, rack issues, and data center disasters while minimizing the effective redundancy factor. It uses Reed-Solomon (RS) coding for redundancy coding within a data center and XOR coding for redundancy across data centers.
![[Screenshot 2023-05-30 at 4.48.26 PM.png]]
### Single f4 Cell
Each f4 cell handles only read and delete operations for locked volumes. Both data files and index files are read-only, and there's no journal file as in Haystack. Instead, f4 uses a different approach to handle "deletion": each BLOB is encrypted and stored, with the encryption key kept in an external database. When a deletion request is received, only the corresponding key is deleted. This method provides privacy guarantees for users and minimizes the delay in deletion operations.

Index files, being relatively small, are stored in triplicate for reliability, circumventing the complexity of encoding and decoding. The data files use n=10, k=4 Liso codes for encoding. Each data file is divided into n consecutive blocks, each with a fixed size b, and for every n such blocks, k parity blocks of the same size are generated. 

---
To illustrate the concept of deletion in the f4 cell, consider this pseudocode:

```python
def delete_blob(blob_id):
    # Retrieve the encryption key for the BLOB from the external database
    encryption_key = get_encryption_key(blob_id)

    # Delete the encryption key from the database
    delete_encryption_key(encryption_key)

    # Now, the BLOB data cannot be decrypted and is effectively deleted
```

This pseudocode shows that the `delete_blob` function does not directly delete the BLOB data but deletes its encryption key instead. This means that while the BLOB data technically still exists, it can no longer be decrypted and is therefore effectively deleted.

---
### Name Node
The Name Node holds the mapping between data blocks, parity blocks, and the storage nodes that physically store these blocks. It utilizes a master-slave backup strategy for fault tolerance.

### Storage Nodes
Storage Nodes are the primary components of the Cell, handling regular read and delete requests. They expose two APIs to the outside world: 

1. **Index API**: Responsible for providing Volume lookup and location information.
2. **File API**: Provides actual data access.

Each Storage Node stores the index file (containing the mapping from BLOB to volume, offset, and length) on the hard disk and loads it into memory in a custom storage structure. 

Each BLOB is encrypted, and its key is placed in additional storage, typically a database. By deleting its secret key, the actual deletion of the BLOB can be achieved, which helps avoid data compaction and eliminates the need for a journal file to track deletion information.

The reading process in the Storage Node involves verifying the existence of the file through the Index API, transferring the request to the Storage Node where the BLOB's data block is located, and then reading directly from the block where the BLOB resides. In the case of a failure, the intact n blocks of all n+k sibling blocks in the damaged block will be read through the Data API and sent to the backoff node for reconstruction.

---
### Backoff Nodes
Backoff Nodes offer a fallback plan when the standard reading process fails. When a failure occurs in a cell, certain blocks become inaccessible and need online recovery from their sibling blocks and parity blocks. The Backoff Nodes, being IO-sparse and computation-intensive, manage these computation-intensive online recovery operations.

During a normal read failure, the Backoff Node exposes the File API to handle the fallback retry. It only restores the data corresponding to the BLOB, not the information of the entire data block where it is located. The recovery of the entire data block is carried out offline by the Rebuilder Nodes.

---
### Rebuilder Nodes
Rebuilder Nodes come into play when the number of physical machines reaches a level where hard disk and node failures are inevitable. These nodes, being storage-sparse and compute-intensive, are responsible for silently rebuilding in the background. They detect data block errors through probes and report them to the Coordinator Nodes. The Rebuilder Nodes rebuild the damaged node using the undamaged n blocks from the sibling blocks and parity blocks in the same stripe. To prevent adverse impact on normal user requests, they throttle their throughput. The Coordinator Nodes are responsible for coordinating and scheduling reconstruction work to minimize the risk of data loss.

---
Consider this pseudocode to illustrate how a Backoff Node works:

```python
def handle_read_failure(read_request):
    # parse the read request into a tuple of data file, offset, and length
    data_file, offset, length = parse_read_request(read_request)

    # get the damaged data block's sibling blocks and parity blocks
    sibling_and_parity_blocks = get_sibling_and_parity_blocks(data_file, offset)

    # read the information of the corresponding length from n blocks
    data = read_from_blocks(sibling_and_parity_blocks, length)

    # perform error correction
    corrected_data = perform_error_correction(data)

    return corrected_data
```

This pseudocode shows that the `handle_read_failure` function of a Backoff Node reads from the sibling blocks and parity blocks of the damaged data block, performs error correction, and returns the corrected data.

---
## Definitions

- **Data Unit (Cell)**: A data deployment and rollback unit composed of 14 racks and 15 machines on each rack.
- **Data Volume**: Divided into logical volume and physical volume, contains multiple data stripes.
- **Data Stripe**: A collection of original n data blocks and generated k parity blocks.
- **Data Block**: Typically around 1G, scattered across different fault-tolerant units.
- **Storage Nodes**: Physical machines responsible for storing the final data.
- **Backoff Node**: Handles error cases and retrieves n sibling blocks for recovery.
- **Coordinator Nodes**: Memory-sparse and computation-intensive nodes that perform data-unit-wide tasks.
- **Encryption Key**: Key used to encrypt the BLOB.
- **Replica**: A redundancy strategy. The most common strategy is to save a few more copies of the same data for backup.
- **Warm Storage**: Refers to storage built for data that is not accessed frequently, in contrast to hot storage.

## Overview
- Facebook Blob Storage System includes different nodes such as Storage Nodes, Name Node, Backoff Nodes, Rebuilder Nodes, and Coordinator Nodes.
- **Storage Nodes** handle regular read and delete requests. They also manage the index file and the file API.
- **Name Node** holds the mapping between data blocks, parity blocks, and storage nodes.
- **Backoff Nodes** manage computation-intensive online recovery operations in case of a read failure.
- **Rebuilder Nodes** are responsible for silently rebuilding in the background in case of hard disk and node failures.
- **Coordinator Nodes** are used to perform data-unit-wide tasks and manage global data distribution.

## XOR Encoding for Geographic Backup

- Geographic backup XOR coding (XOR coding) provides fault tolerance at the data center level by placing the XOR results of two different volumes in a third data center.
- If a datacenter is entirely unavailable, the read request is routed to a geo-backoff node, which retrieves the corresponding BLOB data from the two buddy nodes and XOR nodes in the other data center and reconstructs the damaged BLOB.
- XOR encoding is chosen for its simplicity and effectiveness.
- Load factor calculation: (1.4 * 3) / 2 = 2.1, which gives the effective redundancy factor.

## Pseudocode: Handling Read Failure in a Backoff Node
```python
def handle_read_failure(read_request):
    # parse the read request into a tuple of data file, offset, and length
    data_file, offset, length = parse_read_request(read_request)

    # get the damaged data block's sibling blocks and parity blocks
    sibling_and_parity_blocks = get_sibling_and_parity_blocks(data_file, offset)

    # read the information of the corresponding length from n blocks
    data = read_from_blocks(sibling_and_parity_blocks, length)

    # perform error correction
    corrected_data = perform_error_correction(data)

    return corrected_data
```

---

## Summary

- Facebook Blob Storage uses a distributed system to store data across multiple nodes for fault tolerance.
- The system utilizes XOR coding for geographic backup, ensuring data resilience at the data center level.
- The Backoff Node recovers data when a read request fails by leveraging sibling and parity blocks.
- Coordinator Nodes manage global data distribution and handle tasks such as data reconstruction and balancing of data block placements.
