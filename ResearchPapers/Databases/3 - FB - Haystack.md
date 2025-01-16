### Overview
A data storage and retrieval system called Haystack, used by Facebook to handle a vast amount of photos and data. 
![[Screenshot 2023-05-30 at 4.58.18 PM.png]]
Haystack is essentially a system that optimizes the storage and retrieval of large quantities of small files, like images, by focusing on two design aspects:

1.  **Combining small files into large files**: A computer's file system stores metadata for every file. When you have millions or billions of small files, the overhead of managing these files' metadata can be significant. Haystack mitigates this problem by combining many small files into a single large file, which results in a much smaller quantity of metadata. This approach reduces the storage needs for metadata and increases the overall system efficiency.   
2.  **Simplifying file metadata**: Most file systems are built on the POSIX standard, which contains a lot of metadata information that may not be necessary for certain use cases. Facebook's Haystack removes all unnecessary metadata based on their specific use case, further reducing the overhead of metadata management.

The aim of these design principles is to reduce the metadata to a size that can be comfortably stored in memory (RAM). Keeping the index information in memory allows for much faster data access and retrieval times, because accessing data in RAM is orders of magnitude faster than accessing it from a hard drive or SSD.

In traditional systems, each data access could require multiple Input/Output operations (IO) due to the need to read the metadata first and then the data. However, with Haystack's design, each data access is likely to be completed with a single IO operation, making it more efficient.

### Facebook's Business Scale

Facebook handles a large volume of photos: 260 million new photos per day, amounting to about 20 petabytes of data.
The peak upload rate is about 1 million pictures per second.

### Key Observation
Traditional file systems involve a significant amount of metadata lookup. Facebook needed to reduce the metadata per image, so their system, Haystack, can perform all metadata lookups in memory.

### Haystack's Role
Haystack is an object storage system used to store shared pictures. It's designed for datasets that are written once, read frequently, never changed, and rarely deleted. Traditional storage systems weren't efficient for such high concurrency, which led Facebook to develop their own solution.

### Problem with Traditional Systems (POSIX)
POSIX systems, which use directory organization, store excessive metadata for Facebook's needs, such as user permissions. Reading a file requires loading its metadata into memory first. This is inefficient when there are billions of files, especially for NAS (network attached storage).

Reading or writing an image involves three disk accesses:
1. Translate filenames to inodes
2. Read the inode from disk
3. Read the file itself based on inode information

### Haystack's Features
- **High concurrency and low latency**: Using a CDN was too costly, so Facebook reduced per-file metadata and stored all metadata in memory to aim for a single disk access.
- **Fault Tolerance**: Haystack uses offsite backup for reliability.
- **Cost-effective**: It's cheaper than using a CDN.
- **Simplicity**: The system was kept as simple as possible to ensure usability, despite limited time for refinement and testing.

### The Old Process
The traditional method was to access a web server to obtain a global URL for the photo, containing location information. The CDN served as a cache. If the photo was not in the cache, it was fetched from photo storage and loaded to the CDN.

However, when thousands of photos were stored in a folder, the directory-to-block mapping increased, and it couldn't be loaded into memory all at once, increasing the number of visits.

Neither internal nor external caching (like memcached) was helpful for long-tail traffic.
Each photo occupied at least one inode, which introduced several hundred bits of additional metadata.

----
When discussing "long-tail" traffic, the context is usually about the distribution of requests across a set of items or resources. In this case, the resources might be photos, videos, articles, or other pieces of content.

The "head" of the distribution represents a small number of items that are frequently accessed or requested, while the "long tail" represents a large number of items that are infrequently accessed.

So, the statement "Neither internal nor external caching (like memcached) was helpful for long-tail traffic." means that caching systems, whether they are internal (i.e., built into the system) or external (like Memcached, a general-purpose distributed memory caching system), are not effective for this type of traffic.

The reason for this is straightforward: Caching systems typically store the most frequently accessed items to provide quick access. However, in the case of long-tail traffic, the infrequently accessed items (the majority of the items) are not stored in the cache because they're not accessed frequently enough. This means when a request does come in for one of these items, it's not available in the cache and needs to be fetched from the primary storage, leading to a cache miss.

This is a significant challenge when dealing with large-scale data systems because the sheer number of infrequently accessed items can be enormous, making it impractical to keep them all in the cache. Consequently, optimizing access to long-tail content often involves strategies other than caching.

----
### The New Design
Conventionally, CDNs are used to address bottlenecks in serving static web page content. However, for long-tail files, Facebook needed a different approach. While access to non-popular images is necessary, Facebook aimed to reduce the frequency of such accesses.

Since storing one picture per file resulted in too much metadata, Facebook decided to store multiple pictures in a single file.
Facebook distinguished two kinds of metadata:
1. **Application metadata**: Used to construct URLs that browsers can use to retrieve images.
2. **File system metadata**: Allows a host to locate the image on its disk.
---
### Haystack Overview
Haystack consists of three main components: Haystack Store, Haystack Directory, and Haystack Cache. These are respectively referred to as storage, directory, and cache in this context.
- **Haystack Store**: This is the component responsible for the persistence of image data. It manages the metadata of image files. In terms of its design, the storage capacity of a host machine is divided into physical volumes. Multiple physical volumes across different host machines are grouped into a logical volume. These physical volumes act as replicas of the logical volume, ensuring data backup, fault tolerance, and distribution.
- **Haystack Directory**: This component maintains the mapping of logical volumes to physical volumes. It also manages application-related information, including mappings from pictures to logical volumes and identifying logical volumes with available free space.
- **Haystack Cache**: This component functions as an internal Content Delivery Network (CDN). When a request for a popular image comes in and the higher-level CDN is unavailable or misses the request, Haystack Cache facilitates direct access to the Store.

A typical image request URL to access the CDN looks like this:
```
http://⟨CDN⟩/⟨Cache⟩/⟨Machine id⟩/⟨Logical volume, Photo⟩
```
If a requested image is found (hit) at any level in the CDN, Cache, or Machine, it is returned. If not, the request is forwarded to the next layer.

### Uploading a Photo
**Uploading a photo**: When uploading an image, the request first goes to the web server. The server then selects a writable logical volume from the Directory. Finally, the web server assigns an ID to the image and uploads it to several physical volumes corresponding to the selected logical volume.
![[Screenshot 2023-05-30 at 5.03.51 PM.png]]
- There are some questions to consider in the upload process. 
	- How is a logical volume selected? Is it related to the region of the image request? 
	- What happens if, after selecting the logical volume from the Directory, it is found to be full or the write operation fails due to network problems? 
### Haystack Directory
The Directory is responsible for the following functions:
1. Maintaining the mapping of logical volumes to physical volumes.
2. Load balancing of write operations across logical volumes and read operations across physical volumes.
3. Deciding whether a request is handled by the CDN or the Cache.
4. Checking whether a logical volume has become read-only due to capacity limits or operational reasons.

### Haystack Cache
The Cache receives HTTP requests for images either from the CDN or directly from the user's browser. It is organized as a distributed Hash Table, and the image ID is used to locate the image in the cache. If the image is not found (no hit), the Cache pulls the image from the specified Store according to the URL.
An image is cached in the Cache only when the following conditions are met:
1. Requests come directly from the browser and not from the CDN.
2. The image is pulled from a writable server.
This approach is beneficial because images are often read quickly after being written, and segregating reading and writing operations results in faster performance.

### Haystack Store

The Store component of Haystack is central to its design. Each physical volume within the Store is essentially a large file. Every Store machine locates an image quickly using a logical volume ID and offset. An important design feature of Haystack is that it can fetch an image's filename, offset, and size without accessing the hard disk.

Each physical volume in the Store consists of a superblock and a series of "needles". Each needle contains image meta-information and the image itself. ![[Screenshot 2023-05-30 at 5.10.10 PM.png]]
The layout of a Haystack store file would look like this:
```markdown
[Superblock][Needle 1][Needle 2]...[Needle n]
```
Each needle would have the following structure:
```markdown
[Flag][Cookie][Key][Alternate Key][Data Size][Data Checksum][Image Data][Padding]
```
To speed up image access, basic meta-information of all images is kept in memory. To accelerate the reconstruction of meta-information in memory after a machine reboot, the machine periodically takes snapshots of the in-memory meta-information, creating an index file.
The layout of a Haystack index file would look like this:
```markdown
[Superblock][Needle Index 1][Needle Index 2]...[Needle Index n]
```
Each Needle Index would have the following structure:
```markdown
[Key][Alternate Key][Offset][Size]
```
![[Screenshot 2023-05-30 at 5.11.15 PM.png]]
### APIs

#### Photo Read
A read request from the Cache carries the volume ID, key, alternate key, and cookie. The Store machine looks up related picture meta-information in memory, finds (file descriptor, offset, and size), reads the picture file and meta-information, performs a cookie comparison and checksum verification, and returns image data upon passing the verification.

#### Photo Write
A write request from the web server carries logical volume ID, key, alternate key, cookie, and image data. Each Store machine synchronously appends this information to the corresponding Store file. 

#### Photo Delete
Deleting files is performed by setting the flags corresponding to the image meta-information in memory and in the store file sequentially. 

#### The Index File
To manage consistency during downtime, there are some processes in place. For example, if a new file is written into the volume and memory but crashes before being written to the index file, the system will add the latest file to the index and memory upon restart.

#### File System
To minimize unnecessary disk reading, the traditional POSIX file system is replaced with XFS, which is more suited to large files.

### Error Recovery
Error recovery is managed by regular testing and timely recovery. A background task named "Pitchfork" periodically checks the health of each storage node. Once the health check fails, Pitchfork marks all volume IDs on the problematic machine as read-only. 

### Optimization
Common optimization methods include:
- **Periodic Compaction**: This is an online operation designed to reclaim space occupied by deleted and duplicate files. 
- **Memory Simplification**: It's wasteful to use the flag to mark deletion. Instead, the offset of the meta information corresponding to all deleted files in memory is set to 0.
- **Batch Uploading**: The average performance of a hard drive is better when performing large-scale writes, so batch uploading is preferred. 
By employing these methods, the efficiency of Haystack Store is greatly enhanced.