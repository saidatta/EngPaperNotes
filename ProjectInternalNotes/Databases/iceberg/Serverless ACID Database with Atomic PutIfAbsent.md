https://notes.eatonphil.com/2024-09-29-build-a-serverless-acid-database-with-this-one-neat-trick.html

---
### **Introduction**
In this note, we will discuss how to build a **serverless ACID database** inspired by **Delta Lake**. The key concept behind achieving **ACID compliance** (Atomicity, Consistency, Isolation, Durability) in a serverless environment is the use of an **atomic PutIfAbsent** operation. This operation, combined with immutable data files and metadata-based transaction logs, ensures **snapshot isolation** and concurrency control without the need for complex infrastructure.

---
### **Problem Statement**
When building a serverless ACID-compliant database, the main goals are:
1. **Concurrent transactions**: Supporting multiple readers and writers simultaneously.
2. **Snapshot isolation**: Ensuring that each transaction sees a consistent view of the data, even with concurrent writes.
3. **Atomic writes**: Guaranteeing that a transaction either completes successfully or doesn't change the data at all.
In a typical serverless environment, these goals are achieved through the use of **blob storage** (e.g., S3, GCS, Azure Blob Storage) for data persistence. The key trick is to implement **atomic PutIfAbsent** for managing metadata files, which allows multiple transactions to safely write concurrently.
---
### **Delta Lake Overview**
**Delta Lake** is an open-source protocol that ensures **ACID properties** for serverless databases by organizing data into immutable files stored in blob storage. Its architecture revolves around:
- **Immutable data files** for each transaction.
- **Metadata log** that records the file names corresponding to a transaction.
- **Concurrency control** using an **atomic PutIfAbsent** operation on metadata files.
The core mechanism that Delta Lake uses for concurrency control is simple but effective:
- Metadata files include the **transaction ID** in the file name.
- When committing a transaction, if another transaction has already used the same transaction ID, the commit fails, thus ensuring **snapshot isolation**.
---
### **Atomic PutIfAbsent and Blob Storage**
The atomic PutIfAbsent operation is the key to managing concurrent transactions in a serverless database. It ensures that a metadata file is only written **if it does not already exist**, which is crucial for handling concurrent writers.
#### **How Atomic PutIfAbsent Works**:
- When a transaction attempts to write its metadata, it uses an atomic PutIfAbsent operation to write the metadata file.
- If another transaction has already written a file with the same name (i.e., same transaction ID), the operation fails, and the transaction must retry.
In most cloud blob storage providers, this can be implemented using **conditional writes**:
- **AWS S3**: Use the `If-None-Match` header.
- **Azure Blob Storage**: Use the `If-None-Match` condition.
- **Google Cloud Storage**: Use the `x-goog-if-generation-match` header.
For simplicity, we will demonstrate how to implement this on a **local filesystem** using POSIX atomic file creation (`O_CREAT | O_EXCL`).
### **Blob Storage Interface**
We start by defining an interface that abstracts the blob storage operations needed for atomic PutIfAbsent and metadata management.
```go
type objectStorage interface {
    putIfAbsent(name string, bytes []byte) error
    listPrefix(prefix string) ([]string, error)
    read(name string) ([]byte, error)
}
```
This interface supports three operations:
1. **putIfAbsent**: Atomically writes bytes to a file if it doesn't already exist.
2. **listPrefix**: Lists all files in storage with a specific prefix.
3. **read**: Reads the contents of a file.
---
### **Filesystem Implementation of Atomic PutIfAbsent**
On a **local filesystem**, we can implement atomic PutIfAbsent using POSIX file creation flags `O_CREAT | O_EXCL`, ensuring that the file is only created if it doesn't already exist.
```go
type fileObjectStorage struct {
    basedir string
}

func (fos *fileObjectStorage) putIfAbsent(name string, bytes []byte) error {
    filename := path.Join(fos.basedir, name)
    f, err := os.OpenFile(filename, os.O_WRONLY|os.O_EXCL|os.O_CREATE, 0644)
    if err != nil {
        return err
    }

    _, err = f.Write(bytes)
    if err != nil {
        os.Remove(filename)
        return err
    }

    err = f.Sync()
    if err != nil {
        os.Remove(filename)
        return err
    }

    return f.Close()
}

func (fos *fileObjectStorage) listPrefix(prefix string) ([]string, error) {
    files, err := os.ReadDir(fos.basedir)
    if err != nil {
        return nil, err
    }

    var matched []string
    for _, file := range files {
        if strings.HasPrefix(file.Name(), prefix) {
            matched = append(matched, file.Name())
        }
    }
    return matched, nil
}

func (fos *fileObjectStorage) read(name string) ([]byte, error) {
    return os.ReadFile(path.Join(fos.basedir, name))
}
```

Here, we use **POSIX atomic file creation** to implement `putIfAbsent`, ensuring that files (i.e., metadata) are only written if they do not already exist.

---
### **Transaction Management and Metadata**
Each transaction operates on **immutable data files** and writes metadata to a **log file**. The transaction metadata log contains actions such as creating tables, adding data objects, and changing table schemas. 

When committing a transaction, the system uses the `putIfAbsent` method to write the transaction log. If the log file already exists (indicating another transaction has committed), the transaction fails.
#### **Transaction Structure**:
```go
type transaction struct {
    Id               int
    previousActions  map[string][]Action
    Actions          map[string][]Action
    tables           map[string][]string
    unflushedData    map[string]*[DATAOBJECT_SIZE][]any
    unflushedPointer map[string]int
}
```
#### **Action Types**:
- **AddDataobjectAction**: Represents adding a new data object (e.g., writing rows to a table).
- **ChangeMetadataAction**: Represents a change in table schema (e.g., creating a table).
```go
type Action struct {
    AddDataobject  *DataobjectAction
    ChangeMetadata *ChangeMetadataAction
}
```
### **Snapshot Isolation via Atomic Metadata Management**
When a transaction commits, it creates a **metadata log file** named using the transaction ID. Each new transaction:
1. Scans existing metadata logs to determine the latest transaction ID.
2. Attempts to commit by creating a new metadata log file with an incremented transaction ID using `putIfAbsent`.
If the `putIfAbsent` fails (indicating a conflict), the transaction must retry. This guarantees **snapshot isolation**, as only one transaction can succeed in writing the metadata for a given ID.
### **Example Code: Committing a Transaction**
```go
func (d *client) commitTx() error {
    if d.tx == nil {
        return errNoTx
    }

    // Flush any remaining data
    for table := range d.tx.tables {
        err := d.flushRows(table)
        if err != nil {
            return err
        }
    }

    // Serialize the transaction metadata
    bytes, err := json.Marshal(d.tx)
    if err != nil {
        return err
    }

    // Attempt to write the metadata log atomically
    filename := fmt.Sprintf("_log_%020d", d.tx.Id)
    err = d.os.putIfAbsent(filename, bytes)
    if err != nil {
        return err // Transaction failed due to conflict
    }

    return nil
}
```

This method ensures that only one transaction can succeed in committing, while others will fail and need to retry.

---
### **ACID Guarantees**
The key to achieving **ACID guarantees** in a serverless system lies in:
1. **Atomic Writes**: Using the atomic PutIfAbsent operation ensures **Atomicity**.
2. **Consistency**: Immutable data and metadata logs ensure the database is always in a consistent state.
3. **Isolation**: Snapshot isolation is achieved via atomic metadata management.
4. **Durability**: Data is written to durable blob storage, ensuring it persists even in case of failure.

---
### **Conclusion**
By implementing **atomic PutIfAbsent** for managing metadata logs and leveraging **immutable data files**, we can build a **serverless ACID database** that supports concurrent transactions with **snapshot isolation**. This technique simplifies concurrency control, making it scalable and suitable for serverless architectures.

The **Delta Lake** architecture is an excellent inspiration for building serverless transactional databases with minimal dependencies, while still ensuring the **ACID properties** necessary for reliable data management.