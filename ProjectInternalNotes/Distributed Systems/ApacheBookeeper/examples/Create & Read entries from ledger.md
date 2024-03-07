Here is a simple example of how to create a ledger and write entries to it using the ledger API:

```java
import org.apache.bookkeeper.client.BookKeeper; 
import org.apache.bookkeeper.client.LedgerHandle;

// Create a BookKeeper client 
BookKeeper bkc = new BookKeeper("localhost:2181");   

// Create a ledger LedgerHandle
lh = bkc.createLedger(BookKeeper.DigestType.MAC, "password".getBytes()); 

// Write entries to the ledger 
byte[] data = "Hello, BookKeeper!".getBytes();
lh.addEntry(data);  

// Close the ledger 
lh.close();
```

Now, read entries from the ledger:

```java 
import org.apache.bookkeeper.client.BookKeeper; 
import org.apache.bookkeeper.client.LedgerHandle; 
import org.apache.bookkeeper.client.LedgerEntry;  

// Create a BookKeeper client
BookKeeper bkc = new BookKeeper("localhost:2181");

// Open the ledger LedgerHandle
lh = bkc.openLedger(ledgerId, BookKeeper.DigestType.MAC, "password".getBytes());  

// Read entries from the ledger 
Enumeration<LedgerEntry> entries = lh.readEntries(0, lh.getLastAddConfirmed());  

// Process the entries 
while(entries.hasMoreElements()) { 
	LedgerEntry entry = entries.nextElement();   
	System.out.println(new String(entry.getEntry())); 
}  

// Close the ledger 
lh.close();
```