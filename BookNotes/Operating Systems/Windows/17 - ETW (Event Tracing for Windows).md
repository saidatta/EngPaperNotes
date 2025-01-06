https://www.youtube.com/watch?v=-i_xAF7JqyA


**Audience:**  
These notes are for experienced Windows engineers who are familiar with kernel concepts, system-wide instrumentation, and performance troubleshooting. We will deep-dive into ETW—its architecture, capabilities, how to configure and consume it, tools available, and ways to analyze its output.

---

## What is ETW?

**Event Tracing for Windows (ETW)** is a high-performance, low-overhead tracing mechanism built into the Windows operating system. Introduced in Windows 2000, ETW provides a system-wide logging infrastructure that can capture events from:

- **Kernel components:** (process/thread creation, context switches, I/O, network activities, memory operations, and more)
- **User-mode components:** (application-defined events, diagnostics from services like IIS, SQL Server, and countless other providers)

Events are defined by **providers**—individual sources of telemetry that know how to describe and emit their data. ETW can capture and record these events to files (ETL) or deliver them to real-time consumers.

**Key Advantages:**
- **High Performance:** Minimal overhead even under heavy event loads.
- **Rich Data Model:** Strongly typed events with schemas. Each event can have multiple properties and complex structures.
- **System-wide Scope:** ETW is not limited to a single process; it can capture events from the entire system.

---

## ETW Architecture

1. **Providers:**  
   Entities that generate events. They can be kernel components, user-mode apps, or system services. Each provider is identified by a GUID and can have a schema describing events and their properties.

2. **Sessions:**  
   A controller creates **ETW sessions**, which define:
   - Which providers are active
   - How events are buffered
   - The output destination (e.g., a file or real-time)
   
   Sessions run in the kernel, managing buffers that store events before flushing them to disk or delivering them to consumers.

3. **Consumers:**
   - **File consumers:** Events stored in `.etl` files.
   - **Real-time consumers:** A process can listen to events as they are flushed from buffers, typically with a small delay.

4. **Controllers:**
   - They start/stop sessions, enable providers, and configure parameters (buffer sizes, logging mode).

**Lifecycle:**
- A controller (e.g., `logman`, `xperf`, `wpr`) starts a session.
- The session enables certain providers.
- Providers log events into per-CPU buffers managed by the kernel.
- Events are delivered to consumers or written to files.

---

## Memory and Performance Details

- **Buffering:**
  ETW uses kernel buffers (non-paged pool) for events. This ensures minimal interference with normal system operations.

- **Delay:**
  Typically, events appear to the consumer after a short delay (1–3 seconds) due to buffering. This is acceptable for performance tuning and diagnostics, but might be less ideal for real-time security alerts.

- **Overhead:**
  ETW was designed for always-on scenarios. Even thousands of events per second typically add negligible overhead.

---

## Tools for ETW

### Built-in Tools

1. **`logman`** (Command Line):
   - Enumerate providers: `logman query providers`
   - Start a session: `logman start MySession –p "MyProvider" –o c:\trace.etl`
   - Stop a session: `logman stop MySession`

2. **`traceview`** (From WDK/SDK):
   - GUI tool to start/stop sessions and open existing ETL files.
   - Shows events in a table format.

3. **`xperf`** (Windows Performance Toolkit):
   - More advanced command-line tool for controlling sessions.
   - `xperf –on ProviderName –f trace.etl` to start.
   - `xperf –stop` to end and flush data.
   - Can enable kernel events (like CPU sampling, context switches) easily.

4. **`wpr`** (Windows Performance Recorder):
   - GUI or command line.
   - Simplifies enabling common scenarios (CPU sampling, I/O analysis).
   - `wpr -start GeneralProfile -filemode c:\trace.etl` then `wpr -stop c:\trace.etl`.

5. **`tracepkt`, `tracerpt`**:
   - `tracerpt` converts ETL to CSV or XML:
     ```bash
     tracerpt c:\trace.etl –o c:\out.csv –of CSV
     ```

### Third-Party or Additional Tools

- **Windows Performance Analyzer (WPA)**:
  - GUI tool to open ETL files.
  - Rich visualization (CPU usage, I/O, GPU, memory).
  - Ideal for analyzing complex performance problems, correlations, and timelines.

- **ETW Explorer** (community tool, as shown by the transcript):
  - Enumerates and inspects providers, events, schemas.
  - Helps understand which providers/events might be useful.

- **PerfView**:
  - Excellent for .NET analysis, supports ETW events.
  - Good for CPU stacks, GC events, and more.

- **Process Monitor X** (community example):
  - Uses ETW to capture system events (files, registry, network, memory) in real-time, though with a slight delay.

---

## Providers and Events

**Providers**:
- Identified by a GUID and optionally a symbolic name.
- Can be kernel (e.g., `Microsoft-Windows-Kernel-Process`, `Microsoft-Windows-Kernel-File`).
- Can be user-mode (e.g., `Microsoft-Windows-DotNETRuntime`, `Microsoft-Windows-IIS`).

**Events**:
- Each event has an ID, opcode, version.
- Each event defines a set of fields with names and types.
- Examples:
  - **Kernel Process Provider:** Events for process start/stop, image load, etc.
  - **Kernel Thread Provider:** Context switches, thread creations.
  - **Kernel Memory Provider:** Page faults, memory allocations.
  - **Network Providers:** TCP/IP send/recv events, TCP connection life cycle.

---

## Using ETW Programmatically

**ETW APIs:**
- `StartTrace`, `EnableTrace`, `ControlTrace` for session control.
- `OpenTrace`, `ProcessTrace` for offline consumption of ETL files.
- `EventWrite` for user-mode providers writing events.
- `TraceLogging` API (simplified user-mode provider API).
  
**Consumer Example (C++ Pseudocode):**
```cpp
#include <evntrace.h>
#include <iostream>

static PEVENT_RECORD_CALLBACK MyEventCallback = [](PEVENT_RECORD rec) {
    // Process the event
};

int main() {
    // Open ETL file
    EVENT_TRACE_LOGFILE logFile = {0};
    logFile.LogFileName = L"C:\\trace.etl";
    logFile.ProcessTraceMode = PROCESS_TRACE_MODE_EVENT_RECORD | PROCESS_TRACE_MODE_RAW_TIMESTAMP;
    logFile.EventRecordCallback = MyEventCallback;

    TRACEHANDLE hTrace = OpenTrace(&logFile);
    if (hTrace == INVALID_PROCESSTRACE_HANDLE) {
        std::wcerr << L"Failed to open trace\n";
        return 1;
    }

    // Process all events in the file
    ProcessTrace(&hTrace, 1, NULL, NULL);
    CloseTrace(hTrace);

    return 0;
}
```

**Provider Example (User-mode EventWrite):**
```c
// Provider code snippet using TraceLogging (C):
#include <TraceLoggingProvider.h>

TRACELOGGING_DECLARE_PROVIDER(MyProvider);
TRACELOGGING_DEFINE_PROVIDER(MyProvider, "MyCompany.MyProvider", /* {GUID} */ );

int main() {
    TraceLoggingRegister(MyProvider);
    TraceLoggingWrite(
        MyProvider,
        "MyEvent",
        TraceLoggingString("Hello from ETW", "Message")
    );
    TraceLoggingUnregister(MyProvider);
    return 0;
}
```

---

## Best Practices and Considerations

1. **Filter Before Analyze:**
   Huge volumes of events → target the providers you truly need.
   
2. **Buffer Sizing:**
   Large ETW sessions can consume memory. Adjust buffer sizes if you get buffer lost events.

3. **Security and Privileges:**
   Controlling ETW sessions typically requires admin or certain privileges. Sensitive data might appear in traces.

4. **Real-Time vs Offline:**
   Real-time consumers can quickly become overwhelmed. Often, it’s better to record to a file and analyze offline with WPA or PerfView.

5. **Schema Awareness:**
   Understanding event schemas from tools like ETW Explorer helps decode event properties meaningfully.

---

## Summary

- **ETW**: A flexible, low-overhead telemetry system at the heart of Windows.
- **Huge Ecosystem**: Thousands of providers, each with structured events.
- **Tools**: `logman`, `xperf`, `wpr`, `tracerpt`, `traceview`, WPA, PerfView, ETW Explorer.
- **Scenarios**: Performance tuning, debugging complex kernel interactions, memory analysis, networking insight, security auditing (with some limitations due to delay).

ETW is a cornerstone for advanced Windows troubleshooting and performance analysis. By learning how to start/stop sessions, pick providers, and use analysis tools, engineers can gain deep insights into system behavior and resolve complex issues more efficiently.
```