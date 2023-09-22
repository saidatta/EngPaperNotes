### Database Structure
- **Database to File Correspondence:** Each database (`db`) corresponds to a file. 
- **File Page Structure:** These files are divided into pages of a usual size of 4096 Bytes. The structure of these pages can be divided into three parts:
    - **Metadata Pages:** The first two pages of each file store metadata.
    - **Special Page for Freelist:** There's a special page that stores the freelist. This freelist consists of idle page IDs.
    - **B+ Tree Pages:** The remaining pages form a B+ tree structure.

### B+ Tree Structure
- **Buckets:** Each bucket is a complete B+ tree.
- **Nodes and Pages:** Each node in the B+ tree corresponds to one or more consecutive pages.

### Memory Management
- **Page Cache:** Due to memory being smaller than disk, it's practical to cache pages. A common way to implement this is using an LRU (Least Recently Used) algorithm. However, BoltDB doesn't use this approach. Instead, it uses `mmap()` to create a shared, read-only file mapping, and calls `madvise(MADV_RANDOM)`. This way, page caching is managed by the operating system.
  
### Transaction Management
- **Write-Ahead Logging:** BoltDB does not use Write-Ahead Logging (WAL). All operations within a transaction are performed in memory, and they are only written to disk when a transaction is committed.
- **Committing Transactions:** When transactions are committed, dirty pages (i.e., pages that have been modified) are written back to disk. This ensures that concurrent read transactions are not affected by these changes.
---
### B+ Tree Index
BoltDB makes use of a B+ tree structure for indexing. Its implementation is slightly different from the typical B+ tree.
- A **branch** in BoltDB's B+ tree consists of key/value pairs, where each pair points to a child node. The key defines the start of the child node range, and the value stores the child node page id.
- A **leaf** in BoltDB's B+ tree is a key/value pair that stores data, with no pointer to its sibling node. Typically, an extra pointer in B+ tree points to its sibling node, but this isn't the case in BoltDB.

Three structures in BoltDB are closely related to the B+ tree:
1. **Page:** The page size is generally 4096 bytes, corresponding to each page in the file. The read and write operations on files are performed in units of pages.
2. **Node:** A single node of the B+ tree. When accessing the node, its content is converted into memory as a node, and each node corresponds to one or more consecutive pages.
3. **Bucket:** Each Bucket is a complete B+ tree. All operations are directed at the Bucket.

### Search Process
A typical search process in BoltDB's B+ tree is as follows:
1. Find the root node of the Bucket, which is the root node of the B+ tree page id.
2. Read the corresponding page and convert it into a memory node.
3. If it's a branch node, find the page id using the key.
4. Repeat steps 2 and 3 until you find a leaf node, and then return the corresponding value in the node.

## Page Structure
Pages are organized by page size (4096 bytes). The structure of a page is as follows:
```go
type page struct {
	id       pgid // page id
	flags    uint16 // distinguishes different types of pages
	count    uint16 // number of data items in page
	overflow uint32 // allocates multiple pages if a single page size is not enough
	ptr      uintptr // starting address for saving page data
}
```

The `ptr` is the starting address for saving data. Different types of pages store different types of data. There are 4 types of pages, distinguished by the `flags`:
1. **Meta Page:** Stores db metadata.
2. **Freelist Page:** Stores ids of free pages.
3. **Branch Page:** Stores branch node data.
4. **Leaf Page:** Stores leaf node data.

## Page Cache
BoltDB doesn't implement page cache(bufferpool). Instead, it calls `mmap()` to map the entire file, and calls `madvise(MADV_RANDOM)` to be managed by the operating system page cache. All subsequent read operations on files on disk can be done on `db.data`, which simplifies the implementation.

## Node Structure and File Format

### Leaf Node
The format of a leaf node in the file is as follows:
```go
// leafPageElement represents a node on a leaf page.
type leafPageElement struct {
  flags uint32 // distinguishes subbucket and normal value
  pos   uint32 // offset from leafPageElement to the key
  ksize uint32
  vsize uint32
}
```
The data for a node is stored at the location of `page.ptr`. Following the `leafPageElement`, there are all the keys and values. 

### Branch Node
The format of a branch node in the file differs from that of a leaf node. The value of a branch node is a child node page id, which is stored in `branchPageElement`, and

 the storage of the key is also obtained via `pos`:
```go
// branchPageElement represents a node on a branch page.
type branchPageElement struct {
	pos   uint32
	ksize uint32
	pgid  pgid
}
```
By storing the item header and items separately, lookup time is reduced to O(1) for getting all item headers and a corresponding binary search item. If it were stored in an `[item header][item][...]` format, it would need to traverse and search in order.

### In-Memory Structures
BoltDB first deserializes pages to get the node structure, representing a node in a B+ tree:
```go
type node struct {
	bucket     *Bucket
	isLeaf     bool // Distinguishes branch node and leaf node
	unbalanced bool
	spilled    bool
	key        []byte // The start key of this node
	pgid       pgid
	parent     *node
	children   nodes
	inodes     inodes // Stores node's data
}
```

The `inodes` stores K/V data for a node:
```go
type inode struct {
	flags uint32 // Used for leaf node, distinguishes normal value from subbucket
	pgid  pgid   // Used for branch node, page id of child node
	key   []byte
	value []byte
}
```
The conversion between nodes and pages is handled by `node.read(p *page)` and `node.write(p *page)`.

### Buckets
Buckets represent namespaces (similar to tables), where each Bucket is a complete B+ tree. The structure is as follows:
```go
type Bucket struct {
	*bucket
	tx      *Tx 
	buckets map[string]*Bucket // Subbucket cache
	page    *page // Inline page reference
	rootNode *node // Materialized node for the root page
	nodes    map[pgid]*node // Node cache
	FillPercent float64 // Threshold for filling nodes when they split
}

type bucket struct {
	root     pgid // Page id of the bucket's root-level page
	sequence uint64 // Monotonically incrementing, used by NextSequence()
}
```

The operations `get` and `put` work by first finding the position corresponding to the key. This is done through a `Cursor`:
```go
type Cursor struct {
	bucket *Bucket
	stack  []elemRef
}

type elemRef struct {
	page  *page
	node  *node
	index int
}
```
The `Cursor` starts from the root page of a Bucket and recursively searches until it finds a leaf node. The `Cursor.stack` saves the search path. The node and the location of the key are stored at the top of the stack.

### Subbucket
BoltDB supports nested Buckets. For a subbucket, the flag `leafPageElement.flags = bucketLeafFlag` is set. The subbucket itself is a complete B+ tree.

The format of a subbucket stored in the parent bucket's leaf node is as follows:
```go
type bucket struct {
	root     pgid // Page id of the bucket's root-level page
	sequence uint64 // Monotonically incrementing, used by NextSequence()
}
```
BoltDB also supports inline buckets. If a Bucket is small enough, it can be stored inline in the value after the bucket header.

### B+ Tree Balancing
BoltDB performs all operations in a transaction in memory, such as deletion or addition of a key/value pair. The splitting and merging of the B+ tree doesn't occur until the transaction is committed.
The balance of the B+ tree is divided into 2 steps:
1. **Merge nodes (rebalance):** Any key that's been deleted and nodes whose size or number of keys doesn't meet requirements will be merged.
2. **Split nodes (spill):** Nodes larger than `page size * FillPercent` are split into multiple nodes.
## Transactions
BoltDB supports full transaction features (ACID) and uses MVCC for concurrency control. It allows multiple read transactions and one write transaction to execute concurrently, but read transactions may block write transactions. Here are the
- **Durability:** When a write transaction is committed, a new page is allocated for the data modified by the transaction, and written to the file.
- **Atomicity:** Uncommitted write transaction operations are all performed in memory; the committed write transaction writes data to the file in order.
- **Isolation:** A version number is obtained at the beginning of each read transaction, and the data involved in the read transaction will not be overwritten by the write transaction.

Transactions in BoltDB are represented by the `Tx` struct.
```go
type Tx struct {
	writable bool
	managed  bool
	db       *DB
	meta     *meta
	root     Bucket
	pages          map[pgid]*page
	stats          TxStats
	commitHandlers []func()
	WriteFlag int
}
```

### Writing a Transaction
The process of writing a transaction goes as follows:
1. Initialize the transaction according to the DB: a copy metadata, initialize root bucket, self-increment txid.
2. Starting from root bucket, traverse the B+tree to operate, and all modifications are performed in memory.
```go
// Initialize a write transaction
tx, err := db.Begin(true)
if err != nil {
    log.Fatal(err)
}

// Use the transaction...
_, err := tx.CreateBucket([]byte("MyBucket"))
if err != nil {
    log.Fatal(err)
}

// Commit the transaction
if err := tx.Commit(); err != nil {
    log.Fatal(err)
}
```
3. Submit the write transaction:
    - Allocate a new page for each modified node when splitting a B+tree
    - Assign a new page to freelist
    - Write B+tree data and freelist data to file
    - Write metadata to the file
### Reading a Transaction
The process of reading a transaction is as follows:
1. Initialize the transaction according to the DB: copy metadata, initialize root bucket.
2. Add the current read transaction to db.txs.
3. Starting from root bucket, traverse the B+tree to find.
```go
// Start a read-only transaction
tx, err := db.Begin(false)
if err != nil {
    log.Fatal(err)
}

// Use the transaction...
b := tx.Bucket([]byte("MyBucket"))
v := b.Get([]byte("answer"))

// If needed, manually rollback the transaction
tx.Rollback()
```
4. When finished, remove the transaction itself from db.txs.

### Durability
BoltDB ensures durability by storing metadata in the first two pages of the database, such that the information can be restored when the database is reloaded.
```go
type meta struct {
	magic    uint32
	version  uint32
	pageSize uint32
	flags    uint32
	root     bucket // root bucket page id
	freelist pgid // freelist page id
	pgid     pgid // total number of pages in file
	txid     txid // largest committed write transaction id
	checksum uint64
}
```
![[Screenshot 2023-06-09 at 8.52.16 AM.png]]
```
// freelist represents a list of all pages that are available for allocation.
// It also tracks pages that have been freed but are still in use by open transactions.
type freelist struct {
	ids     []pgid          // all free and available free page ids.
	pending map[txid][]pgid // mapping of soon-to-be free page ids by tx.
	cache   map[pgid]bool   // fast lookup of all free and pending page ids.
}
```
### Atomicity
BoltDB guarantees atomicity, which means that a transaction must be fully executed or not executed at all.
```go
tx, err := db.Begin(true)
if err != nil {
    log.Fatal(err)
}

bucket, err := tx.CreateBucketIfNotExists([]byte("bucket"))
if err != nil {
    tx.Rollback()
    return err
}

if err := bucket.Put([]byte("key"), []byte("value")); err != nil {
    tx.Rollback()
    return err
}

// Commit transaction.
if err := tx.Commit(); err != nil {
    log.Fatal(err)
}
```
### Isolation
BoltDB supports the concurrent execution of multiple read transactions and a single write transaction. The fundamental isolation model ensures that any newly allocated pages from a committed write transaction will not interfere with the ongoing read transactions.
#### Transactional Behavior
- When a write transaction is committed, the old pages are released and new ones are allocated. This process is designed to ensure that the newly allocated pages are not accessed by ongoing read transactions.
- The old pages released by the write transaction may still be accessed by read transactions. Consequently, they can't be immediately reused for the next allocation. To handle this, these pages are stored in a `freelist.pending`, and are only moved to `freelist.ids` for allocation when it is confirmed that no read transaction will use them.
#### Free List Management
- `freelist.pending`: This is a list that maintains the ids of the pages released by each write transaction.
- `freelist.ids`: This is a list that maintains the ids of the pages that are ready for distribution.

#### Transaction ID (txid)
- Each transaction has a `txid`. The `db.meta.txid` holds the largest committed write transaction id.
    - For a read transaction, `txid == db.meta.txid`.
    - For a write transaction, `txid == db.meta.txid + 1`.
- When a write transaction is successfully committed, the metadata is updated, including the `db.meta.txid`.
- The `txid` can be seen as a version number. Only read transactions with a lower version number can access the nodes released by write transactions with a higher version number. When no read transaction with a `txid` smaller than the write transaction's `txid` exists, the pending pages can be released for allocation.

#### Read Transaction Management
- All ongoing read transactions are stored in `db.txs`. When a read transaction is created, it is added to this list:
```go
// Keep track of transaction until it closes.
db.txs = append(db.txs, t)
```
- When a read transaction completes (calls `Tx.Rollback()`), it is removed from this list:
```go
tx.db.removeTx(tx)
```

#### Write Transaction Management
- When a write transaction is created, it identifies the smallest `txid` in `db.txs`, and releases all pages in `freelist.pending` with a `txid` smaller than it:
```go
// Free any pages associated with closed read-only transactions.
var minid txid = 0xFFFFFFFFFFFFFFFF
for _, t := range db.txs {
    if t.meta.txid < minid {
        minid = t.meta.txid
    }
}
if minid > 0 {
    //The minid - 1 is passed in here, it should be fine to pass in the minid, the transaction that reads the data of this version will not access the page released by the writing transaction of this version
    db.freelist.release(minid - 1) // Will merge the page released by the transaction whose txid is less than minid - 1 in pending into ids
}
```

#### Free List File Writing
It's important to note that when writing the `freelist` to a file, both `freelist.ids` and `freelist.pending` are written. Only the `freelist` page will be accessed upon restarting, and all read transactions will be closed, hence all `pending` pages can be used for allocation.

One might wonder if it would be possible to judge whether the old page can be used immediately, instead of doing so when the next write transaction begins. While this could be feasible, writing `freelist` and metadata would require that no new read transactions are initiated, making it less efficient than BoltDB's current implementation.

### Concurrency
##### Read transactions block write transactions
`boltdb`Because `mmap()`the entire file is mapped in using , and a read lock is added to the read transaction:
```go
// Obtain a read-only lock on the mmap. When the mmap is remapped it will
// obtain a write lock so all transactions must finish before it can be
// remapped.
db.mmaplock.RLock()
```
It needs to be allocated when the write transaction is committed `page`. If the current file is not enough `free page`, the file needs to be expanded and re-opened `mmap()`:
```go
// Resize mmap() if we're at the end.
p.id = db.rwtx.meta.pgid
var minsz = int((p.id+pgid(count))+1) * db.pageSize
if minsz >= db.datasz {
    if err := db.mmap(minsz); err != nil {
        return nil, fmt.Errorf("mmap allocate error: %s", err)
    }
}
```

`mmap()`And a write lock is added to:
```
db.mmaplock.Lock()
defer db.mmaplock.Unlock()
```

In this case the read transaction will block the write transaction. One way is to set large `Options.InitialMmapSize`, increasing the initial `mmap()`size :
```go
// InitialMmapSize is the initial mmap size of the database
// in bytes. Read transactions won't block write transaction
// if the InitialMmapSize is large enough to hold database mmap
// size. (See DB.Begin for more information)
//
// If <=0, the initial map size is 0.
// If initialMmapSize is smaller than the previous database size,
// it takes no effect.
InitialMmapSize int
```

BoltDB does not support multiple write transactions to be executed at the same time, but it provides a `Batch` function that can be used to batch multiple write operations into a single transaction.
-   If one of the transactions fails, the remaining transactions will be re-executed, so the transactions are required to be idempotent.
```go
// Obtain writer lock. This is released by the transaction when it closes.
// This enforces only one writer transaction at a time.
db.rwlock.Lock()
```
### Summary
This should give you a practical overview of how BoltDB works, along with some sample code for basic operations. Remember, BoltDB is a relatively low-level tool, and these examples are quite simple compared to the full range of operations you might perform with BoltDB.