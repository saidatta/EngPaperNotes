https://www.youtube.com/watch?v=oTk4NFiEf8M
#### Overview
- Exploration of Google File System (GFS), a distributed file system.
- GFS is a specific case of block storage systems used in cloud computing.
- The video compares GFS with HDFS (Hadoop Distributed File System) and Colossus, its successor at Google.

#### Key Concepts
- **Block Storage**: Fundamental concept behind GFS, used in creating virtual machines on platforms like AWS, Google Cloud.
- **GFS Architecture**: Comprises a single master server, multiple chunk servers, and many clients.
- **Data Storage**: Chunk servers store data on local disks as Linux files, with each chunk replicated for reliability.
![[Screenshot 2024-01-25 at 5.24.47 PM.png]]
#### GFS Master Architecture
- **Master Server**: Coordinates the GFS cluster, manages file system metadata, chunk lease management, garbage collection, and chunk migration.
- **Chunk Servers**: Store actual data, communicate with the master for operations like data replication.
- **Clients**: Libraries that make read/write requests to GFS, directly interacting with chunk servers for data access.
![[Screenshot 2024-01-25 at 5.27.20 PM.png]]
#### Technical Details
1. **Metadata Storage**: All metadata is stored in the master's memory for quick access.
2. **Operation Log**: Metadata changes are journaled to ensure fault tolerance.
3. **Chunk Size**: GFS uses a large chunk size (64 MB) to optimize for high throughput, reducing client-master communication.
4. **Data Integrity**: Utilizes checksums for each 64 KB block within a chunk to detect data corruption.

#### Handling Small Files
- Small files can create hotspots on chunk servers. GFS addresses this by increasing the replication level of such files.

#### Data Integrity and Fault Tolerance
- Checksums verify data integrity.
- Master's state is periodically serialized and replicated for recovery.
- In case of corruption, chunk servers return errors and clients read from alternative replicas.

#### Scaling Challenges and Colossus
- **Scaling Issue with GFS**: The single master became a bottleneck as the number of clients increased.
- **Introduction of Colossus**: Google migrated to Colossus, which uses a distributed metadata model for scalability.
- **Colossus Architecture**: Features multiple "curators" instead of a single master, storing metadata in Google's Bigtable.

#### Example: Implementing a Basic GFS Client
```python
class GFSClient:
    def __init__(self, master_server):
        self.master_server = master_server

    def read_file(self, filename):
        # Get chunk locations from the master
        chunks = self.master_server.get_chunks(filename)
        file_data = []

        for chunk in chunks:
            # Read each chunk from the chunk server
            chunk_server = self.choose_chunk_server(chunk)
            data = chunk_server.read_chunk(chunk)
            file_data.append(data)

        return ''.join(file_data)

    def choose_chunk_server(self, chunk):
        # Implement logic to choose an optimal chunk server
        # This could be based on proximity, load, etc.
        pass

# Example usage
master_server = MasterServer(...)
client = GFSClient(master_server)
file_data = client.read_file("example.txt")
```
- This pseudo-code demonstrates a simple client interface for GFS, handling file read operations by interacting with the master server and chunk servers.

#### Best Practices for GFS Design
- **Chunk Size Optimization**: Based on the type of data and access patterns.
- **Metadata Management**: Efficient handling to avoid bottlenecks.
- **Fault Tolerance**: Implement robust mechanisms for data integrity and master server recovery.

#### Interview Preparation
- Understand GFS architecture, its role in distributed systems, and compare with HDFS and Colossus.
- Be prepared to discuss scaling challenges and solutions in distributed file systems.
- Explore potential improvements or alternative approaches to GFS design.

#### Further Reading
- In-depth GFS design and variations: [crashingtechinterview.com](https://crashingtechinterview.com)
- Comparative studies of GFS, HDFS, and Colossus.

#### Appendix
- **Glossary**:
  - **Chunk Server**: Stores data chunks in GFS.
  - **Master Server**: Manages metadata in GFS.
  - **Curators (Colossus)**: Distributed metadata managers in Colossus.
- **Diagrams and Slides**: For visual understanding of GFS and Colossus architecture.

### Creating Obsidian Links and Tags
- Link to related topics like [[Distributed Systems Architecture]], [[Cloud Computing Fundamentals]], [[Data Integrity in Distributed Systems]].
- Use tags like #GFS, #DistributedFileSystem, #GoogleCloud, #Colossus for easy retrieval and cross-referencing.