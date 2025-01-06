## Overview

**Andrew Hutchings** discusses the challenges and solutions related to quantizing data points, managing rollups, and handling histogram counts within their system. The primary focus is on ensuring accurate count rollups for both normal data points and histograms, addressing conflicts arising from overloaded meanings of count rollups, and maintaining backward compatibility during system enhancements.

## Problem Statement

### Dual Meaning of Count Rollups

- **Normal Data Points**:
  - **Count Rollup**: Represents the number of raw data points processed within a specific time window.
  - **Example**:
    - Metric emitting once per second:
      - **1-second resolution**: Count = 1
      - **1-minute resolution**: Count = 60
      - **5-minute resolution**: Count = 300

- **Histogram Data Points**:
  - **Count Rollup**: Represents the number of observations in the histogram, not necessarily tied to the number of raw data points.
  - **Issue**: Overloading the `count` rollup causes conflicts in analytics, as `count = 0` for histograms might be misinterpreted as no data points received.
### Aggregation Temporality
- **Types**:
  - **Delta**: Represents the change since the last measurement.
  - **Cumulative**: Represents a running total from a starting point.
- **Problem with Cumulative Temporality**:
  - Systems emitting cumulative points without changes lead to redundant data points with `count = 0` after conversion to delta, causing analytics issues.
## Detailed Explanation
### Quantizer and Rollups
- **Quantizer**:
  - Processes raw data points and publishes quantized data with various rollups.
  - **Rollups**: Time-based aggregations over different resolutions (e.g., 1 second, 1 minute).
- **Rollup Types**:
  - **Count Rollup**: Number of data points processed within a time window.
  - **Other Rollups**: Sum, mid, max, etc.

### Histogram Handling

- **Histogram Data Points**:
  - **Fixed Bucket Histograms**: Data points contain bucket boundaries, counts per bucket, and summary data (sum, count, min, max).
  - **Count Interpretation**: Total number of observations, matching the sum of bucket counts.
- **Conflict Issue**:
  - Histogram `count` is an observation count, differing from normal data point `count`.
  - Analytics may misinterpret histogram `count = 0` as no data points, leading to incorrect analytics results.
### Aggregation Temporality Conversion
- **Cumulative to Delta**:
  - Received cumulative data points are converted to delta by subtracting the previous point.
  - **Result**: If no new data, delta count = 0, causing analytics to potentially ignore or misinterpret the point.
### Impact on Analytics
- **Symptoms**:
  - Gaps in charts.
  - Incorrect extrapolation.
  - Misinterpretation of data due to `count = 0`.
## Solution Approach
### Storing Raw Count in Tsdb
- **Objective**: Provide accurate raw data point counts to analytics by storing the raw count separately.
- **Implementation Steps**:
  1. **Emit Raw Count**: Include the count of raw points in the quantizer's emitted data points.
  2. **Persist in Tsdb**: Store the raw count in the Time Series Database (Tsdb) for accurate querying.
  3. **Streaming to Analytics**: Ensure the raw count is available for real-time analytics streaming.
### Encoder and Decoder Adjustments
- **Multi Rollup Value**:
  - Structure that holds multiple rollups for a data point.
  - **Supported Rollups**: Must include the raw count rollup.
- **Encoders**:
  - **Existing**: Support normal and histogram data points.
  - **Modification**:
    - Add support for the raw count rollup.
    - Ensure backward compatibility by introducing versioning.
- **Decoders**:
  - Handle incoming data points, recognizing and correctly interpreting the raw count rollup based on version.
### Configuration Management
- **Configuration Storage**: Managed via Zookeeper.
- **Updating Configurations**:
  - **Tools**:
    - **SFC UI**: Graphical interface for managing configurations.
    - **Command Line**: `sfc config` commands for direct updates.
  - **Versioning**:
    - Introduce new versions for encoders to handle raw counts.
    - Use configuration flags to toggle between old and new encoder versions.

### Backward Compatibility

- **Versioning Strategy**:
  - **Old Encoder**: Remains unchanged to support existing data points.
  - **New Encoder**: Introduces raw count rollup with a new version number.
  - **Configuration Flags**: Toggle between encoders to prevent disruption.

- **Analytics Service Update**:
  - Update analytics libraries to handle new encoder versions.
  - Ensure analytics can parse both old and new data formats during the transition.

## Implementation Details

### Key Classes and Components
- **Quantizer**:
  - Manages the processing and emission of quantized data points.
- **MultiRollupValue**:
  - Java object representing multiple rollups for a data point.
  - **Supported Rollups**: Must include `raw_count` alongside existing rollups.
- **Encoders**:
  - **QuantizedMultiRollupDeltaDatumEncoder**:
    - Encodes normal and histogram data points with delta temporality.
  - **QuantizedHistogramDatumEncoder**:
    - Encodes histogram-specific data points, now including raw counts.
- **Decoders**:
  - **QuantizedHistogramDatumDecoder**:
    - Decodes incoming histogram data points, recognizing raw count rollups based on version.
- **Configuration Interfaces**:
  - Managed via Disco Config project.
  - Backed by Zookeeper for dynamic updates.

### Code Snippets

#### Example: MultiRollupValue with Raw Count

```java
public class MultiRollupValue {
    private Set<RollupType> supportedRollups;

    public MultiRollupValue(Set<RollupType> rollups) {
        this.supportedRollups = rollups;
    }

    public void addRollup(RollupType rollup) throws UnsupportedOperationException {
        if (!supportedRollups.contains(rollup)) {
            throw new UnsupportedOperationException("Rollup not supported: " + rollup);
        }
        // Add rollup logic
    }

    // Other methods...
}

enum RollupType {
    COUNT,
    SUM,
    MID,
    MAX,
    RAW_COUNT, // Newly added rollup
    // Other rollups...
}
```

#### Example: Encoder Versioning

```java
public abstract class DatumEncoder {
    protected int version;

    public DatumEncoder(int version) {
        this.version = version;
    }

    public abstract byte[] encode(MultiRollupValue value);
}

public class QuantizedMultiRollupDeltaDatumEncoder extends DatumEncoder {
    public QuantizedMultiRollupDeltaDatumEncoder(int version) {
        super(version);
    }

    @Override
    public byte[] encode(MultiRollupValue value) {
        // Encoding logic based on version
        if (version == 1) {
            // Original encoding
        } else if (version == 2) {
            // New encoding with raw_count
        }
        // ...
    }
}
```

### Configuration Management with Zookeeper

- **Accessing Configurations**:

```bash
# Using SFC UI
open_sfc_ui --navigate-to-config

# Using Command Line
sfc config get quantizer.alpha.raw_count_enabled
sfc config set quantizer.alpha.raw_count_enabled true
```

- **Default Configuration**:

  ```yaml
  quantizer:
    alpha:
      raw_count_enabled: false # Default value
      # Other configurations...
  ```

### Handling Encoding and Decoding

- **Encoding Process**:
  1. **Quantizer** processes raw data points.
  2. **MultiRollupValue** includes `raw_count` if enabled.
  3. **Encoder** serializes the `MultiRollupValue` based on version.
  4. **Tsdb** stores the encoded data.

- **Decoding Process**:
  1. **Analytics Service** subscribes to data points via Kafka.
  2. **Decoder** interprets `MultiRollupValue` based on version.
  3. **Analytics** utilizes accurate `raw_count` for computations.

## Steps to Implement the Solution

1. **Update Encoders**:
   - Add `RAW_COUNT` to supported rollups.
   - Implement versioning in encoders to handle new rollup.

2. **Update Decoders**:
   - Modify decoders to recognize and correctly parse `RAW_COUNT` based on version.

3. **Modify Quantizer Emission**:
   - Emit `raw_count` in `MultiRollupValue` when enabled via configuration.

4. **Configuration Management**:
   - Define new configuration flags in Disco Config.
   - Ensure configurations are dynamically loaded from Zookeeper.

5. **Backward Compatibility**:
   - Use version numbers and configuration flags to toggle encoder behavior.
   - Ensure analytics can handle both old and new data formats.

6. **Testing**:
   - Implement unit and integration tests for new encoder/decoder versions.
   - Validate accurate `raw_count` handling in both Tsdb and analytics.

7. **Deployment**:
   - Roll out encoder updates behind feature flags.
   - Coordinate with analytics team to update their libraries.
   - Monitor system behavior to ensure accurate data processing.

## Example Math: Delta Calculation

Given cumulative counts:
- **Previous Point**: Count = 100
- **Current Point**: Count = 110

**Delta Count** = Current Count - Previous Count = 110 - 100 = 10

If no new data points:
- **Previous Point**: Count = 110
- **Current Point**: Count = 110

**Delta Count** = 110 - 110 = 0

## Code Example: Handling Raw Count in Encoder

```java
public class QuantizedMultiRollupDeltaDatumEncoderV2 extends DatumEncoder {
    public QuantizedMultiRollupDeltaDatumEncoderV2() {
        super(2); // Version 2
    }

    @Override
    public byte[] encode(MultiRollupValue value) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        
        try {
            // Write version
            dos.writeShort(version);
            
            // Write supported rollups
            dos.writeInt(value.getSupportedRollups().size());
            for (RollupType rollup : value.getSupportedRollups()) {
                dos.writeUTF(rollup.name());
            }
            
            // Write rollup values
            for (RollupType rollup : value.getSupportedRollups()) {
                switch (rollup) {
                    case COUNT:
                        dos.writeLong(value.getCount());
                        break;
                    case RAW_COUNT:
                        dos.writeLong(value.getRawCount());
                        break;
                    // Handle other rollups...
                }
            }
        } catch (IOException e) {
            // Handle exception
        }
        
        return baos.toByteArray();
    }
}
```

## Configuration Management

### Adding New Configuration

1. **Define Configuration Interface**:

    ```java
    @Config
    public interface QuantizerConfig {
        @ConfigKey("quantizer.alpha.raw_count_enabled")
        @DefaultValue("false")
        boolean isRawCountEnabled();
        
        @ConfigKey("quantizer.alpha.other_setting")
        @DefaultValue("default_value")
        String getOtherSetting();
        
        // Additional configurations...
    }
    ```

2. **Update Zookeeper Configuration**:

    ```bash
    # Using SFC UI or Command Line
    sfc config set quantizer.alpha.raw_count_enabled true
    ```

### Reading Configuration in Code

```java
public class Quantizer {
    private QuantizerConfig config;

    public Quantizer(QuantizerConfig config) {
        this.config = config;
    }

    public void emitDataPoint(DataPoint dp) {
        MultiRollupValue rollup = new MultiRollupValue(getSupportedRollups());
        if (config.isRawCountEnabled()) {
            rollup.addRollup(RollupType.RAW_COUNT);
            rollup.setRawCount(dp.getRawCount());
        }
        // Add other rollups...
        // Encode and emit
    }
}
```

## Testing Strategy

- **Unit Tests**:
  - Validate encoder outputs for both versions.
  - Ensure decoders correctly interpret different versions.

- **Integration Tests**:
  - Verify end-to-end data flow from quantizer to Tsdb and analytics.
  - Test configuration toggling and its impact on data emission.

- **Regression Tests**:
  - Ensure existing functionalities remain unaffected by new changes.

## Potential Challenges

- **Backward Compatibility**:
  - Ensuring older analytics services can still process data without `raw_count`.
  
- **Configuration Management**:
  - Properly managing and propagating configuration changes via Zookeeper.

- **Version Handling**:
  - Accurately distinguishing and processing different encoder versions in decoders.

## Action Items

1. **Implement Raw Count Rollup**:
   - Update encoder and decoder classes to support `RAW_COUNT`.
   
2. **Manage Configuration**:
   - Define and implement new configuration flags.
   - Update Zookeeper configurations accordingly.

3. **Update Analytics Libraries**:
   - Ensure analytics services can parse and utilize `RAW_COUNT` rollups.

4. **Testing and Validation**:
   - Develop comprehensive tests for new functionalities.
   - Perform integration testing to ensure system stability.

5. **Documentation and Communication**:
   - Document changes in system architecture and data flow.
   - Communicate updates to relevant teams for seamless integration.

## References

- **Classes**:
  - `QuantizedMultiRollupDeltaDatumEncoder`
  - `QuantizedHistogramDatumEncoder`
  - `MultiRollupValue`
  - `RollupType`
  - `QuantizerConfig`
  - `Quantizer`

- **Configuration Tools**:
  - **SFC UI**: Interface for managing configurations.
  - **Command Line**: `sfc config` commands for direct manipulation.

- **Projects**:
  - **Metric Data Client**
  - **Ts Router**
  - **Disco Config**

## Appendix

### Example: Raw Count Calculation

```math
\text{Delta Count} = \text{Current Count} - \text{Previous Count}
```

- **Scenario 1**:
  - **Previous Count**: 100
  - **Current Count**: 110
  - **Delta Count**: 10

- **Scenario 2**:
  - **Previous Count**: 110
  - **Current Count**: 110
  - **Delta Count**: 0
```