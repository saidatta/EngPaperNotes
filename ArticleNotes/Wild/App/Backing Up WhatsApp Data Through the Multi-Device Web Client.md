### Introduction
Backing up WhatsApp messages and media in an open format is challenging. However, the multi-device beta makes the WhatsApp web client nearly as functional as the mobile client. This guide describes how the web client stores messages and media and introduces a proof-of-concept program, `wadump`, which can extract and display them.
### Background
The multi-device beta allows WhatsApp to be used on multiple devices without requiring the phone to be online. This independence means the web client must store messages and media locally, creating an opportunity for data extraction.
### System Overview
1. **WhatsApp Data Storage**:
   - **Metadata**: Stored in cleartext in IndexedDB.
   - **Text Messages**: Encrypted and stored in IndexedDB.
   - **Media Files**: Encrypted, stored in WhatsApp’s CDN, and cached locally.
2. **IndexedDB**: A persistent storage facility for web applications. It acts as an in-browser key-value store.
### IndexedDB Data Structure
- **Databases**: Created by web applications to store data persistently.
- **Object Stores**: Tables within IndexedDB that store JavaScript objects indexed by a key.
### Metadata Storage
- **Contacts**: Stored in the `contact` object store.
- **Groups**: Stored in the `group-metadata` object store.
- **Messages**: Stored in the `message` object store.
- **Chats**: Stored in the `chat` object store.
### Message Encryption and Storage
- **Text Messages**: Stored as JSON objects in the `message` object store.
  - Encrypted using AES-CBC.
  - Example JSON structure:
    ```json
    {
      "id": ...,
      "t": 1517917600,
      "from": ...,
      "to": ...,
      "msgRowOpaqueData": {
        "_keyId": 1,
        "iv": ArrayBuffer(16),
        "_data": ArrayBuffer(80)
      }
    }
    ```
  - **AES-CBC**: A symmetric encryption scheme using the same key for encryption and decryption.
### Key Storage and Derivation
- **Keys**: Stored in the `keys` object store in the `wawc_db_enc` IndexedDB database.
  - Example key structure:
    ```json
    {
      "id": 1,
      "key": CryptoKey,
      "_expiration": 1609459200000
    }
    ```
- **Key Derivation**: Uses the HKDF key derivation function to generate AES-CBC keys for message encryption.

### Retrieving Keys
- **Info Parameter**: Stored in browser `localStorage` under `WebEncKeySalt`.
- **Salt**: Sent in a WebSocket message when the client starts.
- **Key Derivation Function**:
  ```javascript
  crypto.subtle.deriveKey(
    {
      name: "HKDF",
      hash: "SHA-256",
      salt,
      info
    },
    key,
    { name: "AES-CBC", length: 128 },
    false,
    ["encrypt", "decrypt"]
  );
  ```
### Decrypting Messages
- **Monkey Patching**: Overwrite `crypto.subtle.decrypt` to intercept the decryption process and retrieve the key.
- **Decryption Process**:
  1. Read a message from the `message` object store.
  2. Replace `crypto.subtle.decrypt` with a function that tries to decode the message.
  3. Once the correct key is found, store it and use it to decrypt messages in bulk.
### Decrypting Media
- **Media Storage**:
  - Stored in WhatsApp’s CDN and cached locally.
  - Example media message structure:
    ```json
    {
      "id": ...,
      "t": 1627663880,
      "from": ...,
      "to": ...,
      "type": "image",
      "mimetype": "image/jpeg",
      "filehash": "...",
      "mediaKey": "...",
      "directPath": "...",
      "size": 165863,
      "height": 1281,
      "width": 1600
    }
    ```
- **Media Key Derivation**:
  - Key derived using HKDF with SHA-256.
  - Parameters:
    - Key: 32 bytes from `mediaKey`.
    - Info: "WhatsApp Image Keys" for images, "WhatsApp Video Keys" for videos, etc.
    - Salt: Not used (dummy salt filled with 0s).
  - Output Key Length: 112 bytes.
  - Key slicing:
    ```javascript
    const mediaKeys = {
      iv: key.slice(0, 16),
      encKey: key.slice(16, 48),
      macKey: key.slice(48, 80),
      refKey: key.slice(80, 112)
    };
    ```
### Media Decryption
- **Decryption Process**:
  ```javascript
  const key = await crypto.subtle.importKey("raw", mediaKeys.encKey, "AES-CBC", false, ["decrypt"]);
  bytes = bytes.slice(0, -10);
  const cleartext = await crypto.subtle.decrypt({ name: "AES-CBC", iv: mediaKeys.iv }, key, bytes);
  ```
### Media Caching
- **Cache Storage API**: Used to cache decrypted media files locally.
- **wadump**: Looks in the cache first before downloading media files from the CDN.
### wadump Utility
- **Data Dump**:
  - `message.json`: Contains decrypted text messages.
  - `chat.json`, `contact.json`, `group-metadata.json`: Dumps of respective object stores.
  - `media/<filehash>`: Media files retrieved from the CDN.
### Limitations and Future Work
- **Limitations**:
  - The method may breach WhatsApp’s terms of service.
  - Future changes to the web client may render this method obsolete.
  - wadump cannot reliably extract all media files, especially older ones.
- **Future Work**:
  - Develop a browser extension to sync WhatsApp web client data with the local file system.
  - Use the File System Access API to work around media download limitations.
### Closing Thoughts

Understanding how the WhatsApp web client stores and encrypts data allows for the creation of tools like `wadump` to back up messages and media. While the current implementation has limitations, it opens up possibilities for future enhancements and reliable data preservation.

### References

- Facebook Engineering Blog: Multi-device beta explanation
- Mozilla Developer Network: [SubtleCrypto](https://developer.mozilla.org/en-US/docs/Web/API/SubtleCrypto)
- GitHub Repository: [wadump](https://github.com/user/repo)

### Example Code

**Monkey Patching `crypto.subtle.decrypt`**:
```javascript
(function() {
  const originalDecrypt = crypto.subtle.decrypt;
  crypto.subtle.decrypt = async function(...args) {
    const [algorithm, key, data] = args;
    try {
      const decryptedData = await originalDecrypt.apply(this, args);
      // Store the key for future use
      window.decryptionKey = key;
      return decryptedData;
    } catch (e) {
      return originalDecrypt.apply(this, args);
    }
  };
})();
```

**Media Decryption**:
```javascript
async function decryptMedia(mediaMessage) {
  const mediaKeyBuffer = base64ToArrayBuffer(mediaMessage.mediaKey);
  const hkdfKey = await deriveHKDFKey(mediaKeyBuffer, mediaMessage.type);
  const decryptedData = await decryptWithAES(hkdfKey, mediaMessage.encryptedData);
  return decryptedData;
}

async function deriveHKDFKey(mediaKeyBuffer, mediaType) {
  const hkdfParams = {
    name: "HKDF",
    hash: "SHA-256",
    salt: new Uint8Array(32),
    info: new TextEncoder().encode(`WhatsApp ${mediaType} Keys`)
  };
  const importedKey = await crypto.subtle.importKey("raw", mediaKeyBuffer, "HKDF", false, ["deriveKey"]);
  const derivedKey = await crypto.subtle.deriveKey(hkdfParams, importedKey, { name: "AES-CBC", length: 256 }, false, ["decrypt"]);
  return derivedKey;
}

async function decryptWithAES(key, data) {
  const iv = data.slice(0, 16);
  const encryptedData = data.slice(16);
  const decryptedData = await crypto.subtle.decrypt({ name: "AES-CBC", iv: iv }, key, encryptedData);
  return decryptedData;
}
```

### ASCII Visualization

```
+------------------------------------+
| WhatsApp Web Client                |
|                                    |
| +-------------------------------+  |
| | IndexedDB                     |  |
| |                               |  |
| | +---------------------------+ |  |
| | | keys                      | |  |
| | | +-----------------------+ | |  |
| | | | id, CryptoKey,        | | |  |
| | | | _expiration           | | |  |
| | +---------------------------+ |  |
| |                               |  |
| | +---------------------------+ |  |
| | | message                   | |  |
| | | +-----------------------+ | |  |
| | | | id, t, from, to,      | | |  |
| | | | msgRowOpaqueData      | | |  |
| | +---------------------------+ |  |
| +-------------------------------+  |
+------------------------------------+
```

These detailed notes provide a comprehensive understanding of how to back up WhatsApp data through the

 multi-device web client, tailored for a Staff+ software engineer.