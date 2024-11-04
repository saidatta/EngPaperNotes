## Table of Contents
1. [Overview](#overview)
2. [Key Concepts in TSDB Encodings](#key-concepts-in-tsdb-encodings)
    - [MultiRollupValue](#multirollupvalue)
    - [Point Codec](#point-codec)
    - [Gamut](#gamut)
    - [Sequence](#sequence)
3. [TSDB Encoding Implementations](#tsdb-encoding-implementations)
    - [Brick and Block Structures](#brick-and-block-structures)
    - [Block Formats](#block-formats)
    - [Bundle Encoding](#bundle-encoding)
4. [Codec Factories](#codec-factories)
5. [Code Examples](#code-examples)
6. [Equations and Algorithms](#equations-and-algorithms)
7. [ASCII Diagrams](#ascii-diagrams)
8. [Conclusion](#conclusion)

---
![[Screenshot 2024-10-03 at 6.46.28 PM 1.png]]
## Overview

TSDB encodings are critical for efficient storage and transmission of time series data. The goal of these encodings is to serialize Java objects representing time series data into compact binary formats for storage or wire transmission and to deserialize them back efficiently.

This document covers different types of encodings used in a Time Series Database (TSDB), focusing on how metric data is stored, transmitted, and optimized for performance. Topics include multi-rollup encodings, point codecs, sequence codecs, and the usage of bricks and bundles.

---

## Key Concepts in TSDB Encodings

### MultiRollupValue

MultiRollupValue is an interface in Java used for storing multiple rollup values (SUM, COUNT, AVERAGE, etc.) for a single data point. It is often associated with `MultiRollupTimeValue`, which includes a timestamp.

#### Fields in `ArrayMultiRollupValue` Implementation

```java
private long longMask = 0;    // Tracks which rollups have long values
private long doubleMask = 0;  // Tracks which rollups have double values
private long nullMask = 0;    // Tracks which rollups are explicitly set to null
private long[] values = new long[NUM_ROLLUPS];  // Stores actual values
```

- **longMask**: Indicates rollups where long values are defined (e.g., COUNT, SUM).
- **doubleMask**: Indicates rollups where double values are defined (e.g., LATEST).
- **nullMask**: Tracks which rollups are explicitly set to null.
- **values[]**: The actual rollup values are stored in this array. For double values, Java’s `Double.doubleToRawLongBits()` is used to store 64-bit IEEE 754 representations in a long array.

#### Example: Rollup Masks

- A rollup that tracks `SUM` and `COUNT` might have a `longMask` of `0b110`, meaning both rollups store long values.
- A rollup tracking `LATEST` and `SUM` might have a `doubleMask` of `0b0101`, indicating that these two rollups store double values.

### Point Codec

Point codecs serialize and deserialize single data points (`MultiRollupValue` instances). The encoding involves calculating deltas (difference between current and previous values) to optimize storage.

#### Point3
Point3 is a delta-based codec. It saves space by encoding differences between consecutive values. Point3 uses variable-length encoding to store only the minimum necessary bytes, with types like `int8`, `int16`, or `int64` depending on the value’s magnitude.

#### Point3 Encoding Layout

- **rollupGamutSize bits**: Condensed gamut mask representing enabled rollups.
- **4 bits**: Type (e.g., `int8`, `int32`, or `float` versions).
- **1 bit**: Delta flag (`0` for actual value, `1` for delta).
- **0-64 bits**: Encoded value using the least number of bits required for the type.

### Gamut

Gamut refers to the subset of rollups being encoded. Gamut encoding allows for space optimization by encoding only relevant rollups.

#### Gamut Encoding Example
For a rollup containing `SUM`, `COUNT`, `MIN`, `MAX`, `LAG`:
- Full mask: `0b10_0110_1010`
- Condensed gamut mask: `0b1_1111` (only stores enabled rollups)

This optimization reduces the number of bits required for encoding the rollup.

---

## TSDB Encoding Implementations

### Brick and Block Structures

Time series data is organized into **bricks** and **blocks** to optimize for read/write performance by grouping datapoints over a fixed time range.

- **Brick**: Encodes a time range of data points for a single MTS (Metric Time Series).
    - **startTime**: Start time of the brick in epoch ms.
    - **endTime**: End time of the brick.
    - **length**: Number of bits contained in the brick.
    - **size**: Number of datapoints in the brick.

- **Block**: A wrapper around a brick, containing additional metadata such as MTS ID, start time, and time resolution.
    - **Block3**: Used for legacy TSDB, binary compatible with modern encodings.
    - **Block10**: Extended format with additional metadata fields like org ID.

#### Block3 Encoding Layout

- **Header**: Includes metadata such as size, resolution, and timestamp.
- **Payload**: Contains the brick encoded using `Point3`.

### Bundle Encoding

A **bundle** encodes data from multiple MTS within the same blob. This reduces the number of `GET` and `PUT` requests when interacting with storage systems like Cassandra or S3.

- **Bundle Layout**:
  - **MTS ID**: Maps to a sequence of time series data.
  - **Metadata**: Optional key-value pairs associated with the MTS.
  - **Value**: Encoded bricks representing data points for the MTS.

#### Bundle Encoding Example

```
<8 bits>: Bundle flag (0 if bundle is empty)
-- For each MTS ID --
<64 bits>: MTS ID
<32 bits>: Buffer offset for next key
<32 bits>: Offset for value linked list
-- For each value --
<32 bits>: Brick index
<Variable>: Encoded MultiRollupValues in the brick
<Padding>: To align to full bytes
```
---
## Codec Factories
Codec factories are used to create codecs for encoding and decoding blocks, bundles, and sequences. Each factory is resolution-specific, meaning different factories are required for different time resolutions.
### Example: Using CodecFactory
```java
CodecFactory codecFactory = codecFactoryProvider.getCodecFactory(resolutionMs);
BlockCodec blockCodec = codecFactory.getBlockCodec(version);
```
- **CodecFactory**: Retrieves the appropriate codec based on the time resolution and codec version.
---
## Code Examples
### Example 1: MultiRollupValue Encoding with Point3
```java
MultiRollupValue current = ...;
MultiRollupValue previous = ...;

Point3Encoder encoder = new Point3Encoder(rollupGamutMask);
byte[] encodedBytes = encoder.encode(current, previous);
```

- **Delta Compression**: Encodes only the difference between `current` and `previous` values.
- **Variable-Length Encoding**: Only uses the required number of bytes based on value type.

### Example 2: Bundle Creation for Multiple MTS

```java
Map<Long, Sequence> mtsData = new HashMap<>();
Bundle bundle = new Bundle();
for (Long mtsId : mtsData.keySet()) {
    bundle.addMTS(mtsId, mtsData.get(mtsId));
}
```

- **MTS Data Storage**: Multiple MTS sequences are stored within the same bundle for efficiency.
---
## Equations and Algorithms

### Delta Encoding for Point3

The delta is computed as:

\[
\Delta = \text{current\_value} - \text{previous\_value}
\]

Where:
- `current_value`: The current data point.
- `previous_value`: The previous data point.

The encoded value is stored using variable-length encoding, which minimizes the space required.

### Gamut Mask Calculation

For a rollup containing `SUM(2)`, `COUNT(4)`, `LAG(9)`:

\[
\text{rollupMask} = 0b10\_0100\_0010
\]

The condensed gamut mask would be:

\[
\text{gamutMask} = 0b111
\]

---

## ASCII Diagrams

### Brick and Block Relationship

```
	+-----------------------+
|        Block           |
| +-------------------+ |
| |       Brick        | |
| |  +--------------+  | |
| |  | Datapoint 1   |  | |
| |  | Datapoint 2   |  | |
| |  |     ...       |  | |
| |  +--------------+  | |
| +-------------------+ |
+-----------------------+
```

---

## Conclusion

Efficient encoding is essential for scalable time series databases. The use of delta encoding, gamut masking, and sequence codecs ensures minimal storage and transmission overhead while maintaining flexibility and accuracy in rollup calculations. Understanding these encoding mechanisms is crucial for anyone working on TSDB internals, as they form the foundation of data storage and query performance optimization.