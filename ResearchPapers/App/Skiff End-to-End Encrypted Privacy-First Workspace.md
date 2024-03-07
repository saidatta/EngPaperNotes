https://skiff-org.github.io/whitepaper/Skiff_Whitepaper_2023.pdf
---
### 1. Overview
- **Purpose**: The whitepaper presents a comprehensive security framework for Skiff, focusing on privacy and encryption for workspace, email, and calendar functionalities.
- **Key Features**:
  - End-to-end encryption for all documents, files, messages, and calendar events.
  - No sensitive information is accessible to anyone besides the creator and their collaborators.
  - Implements robust authentication methods, out-of-band key verification, and two-step authentication.
  - Utilizes advanced privacy, cryptography, and decentralization technologies.
---
### 2. Threat Model
- **Assumptions**:
  - Adversaries can access any data sent over the network, even if encrypted.
  - Data stored on cloud services cannot be assumed confidential.
  - Skiff servers are "honest but curious", meaning they won't deny access or serve compromised applications but may be vulnerable to data compromise.
- **Security Measures**:
  - Use of subresource integrity and out-of-band public key verification to maintain honest client applications.
  - Native applications for iOS, Android, and macOS to mitigate network threats.
- **Privacy Protections**:
  - Design to prevent user enumeration and allow users to block or report shared content.
  - No collection of personally identifying information upon user signup.
---
### 3. Security Properties
- **End-to-End Encryption**:
  - All sensitive information remains encrypted and visible only to authorized users.
  - Emails and calendar events include encryption for titles, locations, descriptions, and external attendees.
- **Resilience**:
  - Resistant to man-in-the-middle attacks, ensuring privacy even over untrusted communication channels.
  - Measures to combat user abuse, impersonation, and phishing attacks.
  - Emphasis on usability to prevent users from switching to less secure alternatives.
---
### 4. Open Source Initiative
- **Impact**:
  - Encourages community engagement by allowing public review, use, and contributions.
- **Components**:
  - Skiff Mail client is entirely open-source.
  - Cryptography libraries and a typed envelope library for data versioning and authentication are available to developers.
  - UI library is also open-source, promoting a privacy-first design system.
- **Future Plans**:
  - Ongoing efforts to open-source the remainder of Skiff's products throughout 2023.
---
### Technical Notes
- **Cryptography**: Discusses the implementation of end-to-end encryption, highlighting the importance of privacy in communications. The paper specifies encryption of documents, emails, and calendar events, ensuring that sensitive information is safeguarded against unauthorized access.
- **Authentication and Verification**: Details robust authentication methods, including out-of-band key verification and two-step authentication, to enhance security and verify user identities effectively.
- **Decentralization and Privacy**: Emphasizes the use of decentralization technologies to maintain user privacy and ownership of data, a critical aspect of Skiff's design philosophy.
- **Open Source Philosophy**: The commitment to open-sourcing Skiff's components underlines the importance of transparency and community involvement in developing secure, privacy-focused software.

#### Conclusion

The Skiff whitepaper presents a holistic approach to building a privacy-first, encrypted workspace that prioritizes user security and data ownership. Through its threat model, security properties, and open-source initiatives, Skiff aims to offer a secure and user-friendly platform for communication and collaboration, setting a new standard in the field of privacy-focused software solutions.

---
#### 3. System Design
---
#### 3.1 Overview and Encryption Protocols
- **Encryption Schema**: Utilizes public-key authenticated encryption for secure sharing of encrypted documents. Each user receives a set of public and private keys for signing and encryption, generated using Curve25519.
- **Encryption Library**: Uses tweetnacl-js for asymmetric (tweetnacl.box) and secret-key (tweetnacl.secretbox) authenticated encryption, ensuring confidentiality and authenticity (AEAD).
- **Key Management**: Public keys are shareable for sending or verifying information, while private keys, crucial for decryption and signature generation, remain confidential.
---
#### 3.2 Login, Creating Accounts, and Private Key Storage
- **Account Creation Process**:
  1. Secure password generation and in-browser keypair generation using tweetnacl.js.
  2. Derivation of a symmetric key from the user's password using Argon2id and HKDF, creating separate keys for login and encrypting user secrets.
  3. Encryption of user's private keys (not the password) with the derived key for server storage, ensuring that decryption is only possible in-browser.
- **Authentication Protocol**: Uses Secure Remote Password (SRP) for login, enhancing security against brute-force and dictionary attacks. User login results in access to encrypted user data and a time-limited JWT for document access.
---
#### 3.3 Crypto Wallet Login
- **Integration with Crypto Wallets**: Supports login/signup using MetaMask and Brave Wallet, leveraging Ethereum public keys for authentication and encrypted password storage.
- **Authentication Model**: Employs a challenge-response mechanism via digital signatures to prevent impersonation and man-in-the-middle attacks.
- **Potential for Expansion**: Aims to introduce private collaboration tools to the crypto community, acknowledging limitations with wallets that lack native encryption/decryption capabilities.
---
#### 3.4 Two-Factor Authentication (2FA)
- **Implementation**: Offers optional setup of 2FA using authenticator apps like Authy, Google Authenticator, and Duo Mobile.
- **Security Process**: Involves generating a 2FA secret in-browser, which is then encrypted by Skiff and stored for future logins. This approach adds a layer of security against phishing and unauthorized access.
- **WebAuthN Integration**: Mentioned as a potential additional verification method, enhancing the security framework.
---
#### 3.5 Account Recovery and Password Changes
- **Recovery Mechanism**: Utilizes a recovery key (similar to the password_derived_secret) for encrypting user data, allowing for account recovery and password reset via email verification.
- **Security Considerations**: Emphasizes the distinct security treatment of the recovery key hash versus password hash, highlighting the uniqueness and unpredictability of the recovery key.
- **Future Directions**: Suggests the possibility of using Shamir Secret Sharing for enhanced security and usability in account recovery, proposing a 2-of-3 secret sharing model for increased flexibility.
---
### Technical Insights
- **Cryptography and Security**: The document details Skiff's advanced use of cryptographic protocols and libraries, emphasizing the importance of secure key management and authenticated encryption for user data privacy.
- **User Authentication and Security**: Describes a multifaceted approach to user authentication, incorporating both traditional password-based methods and innovative crypto wallet integration, all while prioritizing end-to-end encryption.
- **Two-Factor Authentication**: Highlights the implementation of 2FA as an additional security layer, crucial for protecting against sophisticated attacks and unauthorized access attempts.
- **Account Recovery**: Addresses the challenges of account recovery in encrypted systems, proposing secure and user-friendly mechanisms for password reset and account access restoration, with a nod towards future enhancements in usability and security.
#### Conclusion
The Skiff research paper's sections on system design, encryption protocols, and security measures provide a deep dive into the technological underpinnings of Skiff's privacy-first workspace. By focusing on end-to-end encryption, secure key management, and user-friendly security features, Skiff aims to deliver a secure and private platform for communication and collaboration, setting a benchmark in the realm of encrypted services.

---
#### 4. Skiff Pages and Drive: Documents and Files
#### 4.1 Document and File Encryption Model
- **Concept**: Skiff treats each document (Page) as a fundamental unit of collaboration, supporting rich text, PDFs, and folders, all of which are end-to-end encrypted.
- **Encryption Strategy**:
  - **Session Key**: Each document is encrypted with a short-term symmetric session key for content and metadata, enabling real-time collaboration.
  - **Asymmetric Hierarchical Keypair**: Facilitates unique encrypted session keys for each collaborator, derived from their public keys, ensuring private access to shared documents.
---
#### 4.2 Opening a Document (d1)
- **Procedure**:
  1. Bob retrieves his encrypted copy of `session_key_d_1`, encrypted with his public key.
  2. Using his private key, Bob decrypts this session key.
  3. With the decrypted session key, Bob can access and edit document `d1`.
---
#### 4.3 Sharing a Document (d2)
- **Process**:
  1. Alice intends to share document `d2` with Charlie and retrieves Charlie's public key.
  2. Alice encrypts `session_key_d2` with Charlie’s public key and sends it for Charlie's future access, along with a signature.
  3. Charlie, upon signing in, accesses the encrypted session key and, after decryption, can read and edit `d2`.
---
#### 5. Real-time Collaboration

- **Mechanism**: Utilizes CRDTs (Conflict-free Replicated Data Types) for seamless real-time collaboration, ensuring that document updates are synchronized among collaborators in an encrypted manner.
- **Collaboration Flow**:
  - Collaborators establish WebSocket connections to a shared "room," identified by the document's unique ID.
  - Updates are broadcasted and received through this connection, encrypted with the document's session key.
  - Each collaborator applies updates to their local copy, leading to a consistent final document version across all users.
---
### Technical Insights
- **Encryption Model**: Details the comprehensive approach to encrypting documents and metadata on Skiff, highlighting the importance of session keys and asymmetric hierarchical keypairs for secure document sharing and collaboration.
- **Key Management**: Explores the process of key encryption and decryption for accessing documents, emphasizing the security measures taken to protect user data.
- **Document Sharing and Access**: Describes a secure method for sharing documents with collaborators, ensuring that only authorized users can access the encrypted content.
- **Real-time Collaboration**: Discusses the use of CRDTs for encrypted real-time collaboration, showcasing the technology's ability to handle simultaneous updates without conflicts, preserving document integrity and user privacy.
#### Conclusion
The sections on Skiff Pages and Drive delve into the secure foundation of document and file management within the Skiff platform. By employing advanced encryption models and leveraging CRDTs for collaboration, Skiff ensures that users can safely create, share, and collaborate on documents in a privacy-first environment. This approach not only secures sensitive information against unauthorized access but also enhances the usability and efficiency of collaborative efforts, setting a high standard for privacy and security in digital workspaces.
#### 6. Building a Filesystem
---
#### 6.1 Scalable Sharing and Unsharing

- **Challenges**: Addresses the scalability issues in sharing a large number of documents, highlighting the impracticality of O(n) operations for large organizations.
- **Solution**: Introduces per-document hierarchical keys for scalable sharing and unsharing, using asymmetric key pairs to encrypt and decrypt child document keys, ensuring efficient and secure document access.
---
#### 6.2 Sharing a Document (dr) in a Recursive Filesystem
- **Process**: Alice can share a document root (dr) with Bob by encrypting dr's hierarchical keys with Bob's public key, enabling Bob to access all child documents recursively in an O(1) operation.
---
#### 6.3 Unsharing a Document (d3) in a Recursive Filesystem
- **Method**: Alice can unshare Bob from the root document (d3) by deleting Bob's encrypted key copies from d3's key register, achieving an O(1) operation from the client's perspective.
---
#### 6.4 Expiring Access
- **Functionality**: Allows documents to be shared with an expiration date. After the expiration, any modifications or access requests trigger re-encryption of the document, effectively revoking the expired user's access.
---
#### 6.5 Link Sharing a Document (d2)
- **Innovation**: Skiff's link sharing maintains end-to-end encryption, making it unique compared to traditional link sharing methods. The mechanism involves generating a shareable link that embeds encrypted session keys, preserving privacy and security.
---
#### 6.6 Email Invitation Links for Document (d3)
- **Invitation System**: Facilitates the invitation of new users to Skiff documents via temporary, secure links embedded in email invitations. New collaborators can create accounts and immediately begin collaborating, ensuring a seamless onboarding experience.
### Technical Insights
- **Scalable Document Sharing**: Explores the innovative approach to scalable document sharing and unsharing within Skiff's filesystem, emphasizing the use of hierarchical keys for efficient access management.
- **Access Expiration**: Discusses the mechanism for expiring access to shared documents, highlighting the balance between collaboration and security by re-encrypting documents upon access expiration.
- **Link Sharing**: Details the secure link sharing feature, showcasing how Skiff leverages end-to-end encryption to enhance privacy and security in document collaboration.
- **Invitation Links**: Describes the process of inviting new collaborators to Skiff, ensuring that even those without existing accounts can participate in secure collaboration efforts.
#### Conclusion
The sections on building a filesystem in the Skiff platform illustrate a comprehensive approach to managing document access and collaboration at scale. By integrating advanced cryptographic models, such as hierarchical keys and secure link sharing, Skiff addresses the challenges of privacy, security, and efficiency in digital collaboration. The platform's innovative solutions for sharing, unsharing, and expiring access underscore its commitment to creating a user-friendly, secure environment for collaborative work, paving the way for the future of privacy-first digital workspaces.

---
#### 7. Document Chunks and Expiration
- **Data Model**: Documents are divided into chunks, each encrypted with the document's session key. This division enhances efficiency and security for various document sizes and types.
- **Authentication**: Each chunk is authenticated, including its sequence number and a final chunk flag, to prevent tampering or incorrect sequencing by Skiff or external actors.
- **Expiring Content**: Chunks can have an `expiry_date` for content that should expire, with the server omitting expired content from downloads and deleting expired fields regularly.
---
#### 8. Skiff Mail
##### 8.1 Email Sending and Receiving - Skiff Mail to Skiff Mail
- **Process**: Similar to document sharing, emails between Skiff users are encrypted with a symmetric key, then this key is encrypted with the sender's and recipient's public keys, ensuring end-to-end encryption.
- **Privacy**: Ensures total privacy for Skiff Mail content, with messages accessible only to designated recipients and not even Skiff having access.
##### 8.2 Email Sending - Skiff Mail to External
- **Mechanism**: Emails to external addresses are encrypted with a temporary decryption service's public key, which processes and sends the email, then deletes the decryption key, maintaining privacy.
- **External Communication**: Allows Skiff users to securely communicate with non-Skiff email addresses while preserving the content's confidentiality.
##### 8.3 Email Receiving - External to Skiff Mail
- **Encryption Service**: External emails are immediately encrypted upon receipt with the recipient's Skiff public key, ensuring the email remains private and secure within the Skiff ecosystem.
- **End-to-End Encryption**: Maintains the integrity and privacy of emails received from external sources, ensuring they are accessible only to the intended Skiff recipient.
##### 8.4 Mail Threading
- **Functionality**: Uses the JWZ threading algorithm to maintain organized conversations, ensuring that replies and forwards are accurately grouped with their original messages.
- **User Experience**: Enhances usability and privacy by efficiently managing long email threads, making it easy for users to follow conversations.
### Technical Insights
- **Chunk-Based Document Model**: Discusses the benefits of splitting documents into chunks for encryption, improving data handling efficiency and security, especially for large documents.
- **Expiring Document Content**: Explores the concept of expirable document chunks, a feature that enhances privacy and control over document lifespan and access.
- **Skiff Mail Encryption**: Details the encryption processes for Skiff Mail, covering end-to-end encrypted communication within Skiff and with external email services, emphasizing the privacy-first approach.
- **Mail Threading**: Highlights the use of the JWZ algorithm for mail threading in Skiff Mail, illustrating Skiff's commitment to maintaining user-friendly and secure email communication.
#### Conclusion
The sections on document chunks, expiration, and the detailed breakdown of Skiff Mail's functionalities reveal Skiff's comprehensive approach to secure, private digital communication and document management. By leveraging advanced encryption techniques and efficient data models, Skiff ensures that users can collaborate and communicate with confidence in the privacy and security of their information. The platform's innovative features, such as expiring content and sophisticated mail threading, underscore its dedication to creating a user-centric, privacy-first workspace in the digital age.
#### 9. Skiff Calendar
- **Integration**: Designed as a privacy-first calendar that integrates seamlessly with Skiff Mail, Pages, and Drive, emphasizing the importance of keeping sensitive calendar information private.
- **Data Unit**: Each calendar has its own Curve25519 public-private keypair for encrypting all event information, ensuring privacy for titles, locations, notes, and external recipients.
- **Sharing Mechanics**: Sharing calendars involves sharing the calendar's private key, allowing shared users to decrypt events, similar to document sharing in Skiff.
##### 9.1 Calendar Events
- **Encryption**: Events are encrypted with the calendar's keypair, with personal preferences like event color and notifications also kept encrypted.
- **User Privacy**: Ensures all sensitive information and user preferences are end-to-end encrypted and only accessible to authorized users.
##### 9.2 Calendar Invites - Skiff to Skiff
- **Process**: Invites are encrypted with the receiving user's primary calendar public key, akin to sending an encrypted email within Skiff.
##### 9.3 Calendar Invites - External
- **Handling**: External invites use the iCalendar format, sent as encrypted emails, keeping the list of external users and event details private and encrypted.
##### 9.4 Calendar Sharing
- **Key Sharing**: Similar to sharing a document or file, sharing a calendar grants access to all events by sharing the calendar's private key.
#### 10. Import
##### 10.1 Pages Import
- **Client-side Encryption**: Files and Google Docs are encrypted client-side using Skiff's document model, ensuring end-to-end encryption during the import process.
##### 10.2 Mail Import
- **Server-side Processing**: Imports from Gmail and EML files are encrypted by a Skiff-run service, offering scalability for large mail imports while maintaining privacy.
##### 10.3 Calendar Import
- **Client-side Processing**: ICS files and Google Calendars are imported client-side, parsed, and encrypted before storage, preserving privacy for event information.
#### 11. Public Key Verification
- **Trust Mechanism**: Skiff facilitates the verification of other users' public signing keys via "verification phrases," enhancing trust and security in user interactions.
- **Verification Storage**: Verified public signing keys are stored in encrypted user data, ensuring accuracy and privacy in future communications and collaborations.
#### 12. Conclusion
- **Security Model**: Skiff aims to offer a decentralized, scalable, and end-to-end encrypted platform for collaboration and communication, safeguarding sensitive information across all services.
- **Privacy Vision**: Challenges the norm of neglected privacy in digital products, proposing an innovative, privacy-centric ecosystem supported by new security technologies like crypto wallets and client-side indexing.
- **Future Potential**: Envisions expanding Skiff's platform to build more private, performant, and user-friendly software, leveraging ongoing technological advancements.
### Technical Insights
- **Skiff Calendar's Unique Encryption**: Highlights the application of encryption to calendar events, mirroring the privacy measures in Skiff's document and mail services.
- **Import Processes**: Discusses the encryption and privacy considerations in importing documents, mails, and calendar events into Skiff, emphasizing the platform's comprehensive approach to user data protection.
- **Public Key Verification**: Explores Skiff's method for enhancing user trust and security through public key verification, ensuring secure and private interactions within the platform.
#### Conclusion
The final sections of the Skiff research paper detail the comprehensive encryption and privacy features of Skiff Calendar, alongside the import functionalities and public key verification process. Skiff's dedication to building a secure, decentralized, and user-centric collaboration platform is evident, with each feature designed to protect user privacy and data integrity. As digital privacy concerns continue to grow, Skiff's innovative approach sets a new standard for secure digital communication and collaboration tools, promising a future where user privacy is prioritized and protected.