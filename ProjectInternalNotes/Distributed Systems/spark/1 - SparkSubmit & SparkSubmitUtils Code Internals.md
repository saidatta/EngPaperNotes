Focusing on the [`SparkSubmit`](https://github.com/apache/spark/blob/master/launcher/src/main/java/org/apache/spark/deploy/SparkSubmit.scala) class and its associated components.
>  
> **Note-taking Goals**:
> - Understand how Spark's internal submit mechanism works.
> - See annotated code blocks.
> - Diagram the flow of control.
> - Provide examples and references.
> - Include visual ASCII or sequence diagrams.
> - Provide comprehensive coverage to facilitate deep learning of Spark internals.
>  
> **Source License**: This code is from the Apache Spark project, licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).
## Table of Contents

1. [Full Source Code](#full-source-code)
2. [High-Level Overview](#high-level-overview)
3. [Key Classes & Objects](#key-classes--objects)
4. 1. [SparkSubmitAction (Enumeration)](#sparksubmitaction-enumeration)
5. 2. [SparkSubmit (Class)](#sparksubmit-class)
6. 3. [SparkSubmitUtils (Object)](#sparksubmitutils-object)
7. [Detailed Walkthrough](#detailed-walkthrough)
8. 1. [Entry Points and Flow](#entry-points-and-flow)
9. 2. [Argument Parsing](#argument-parsing)
10. 3. [The `doSubmit` Method](#the-dosubmit-method)
11. 4. [The `parseArguments` Method](#the-parsearguments-method)
12. 5. [Submit, Kill, Request Status, Print Version](#submit-kill-request-status-print-version)
13. 6. [Preparing Submit Environment](#preparing-submit-environment)
14. 7. [The `runMain` Method](#the-runmain-method)
15. 8. [Maven Dependency Resolution Mechanics](#maven-dependency-resolution-mechanics)
16. [Sequence Diagram of `SparkSubmit`](#sequence-diagram-of-sparksubmit)
17. [Examples & Usage](#examples--usage)
18. 1. [Example: Submitting a Scala/Java JAR](#example-submitting-a-scalajava-jar)
19. 2. [Example: Submitting a Python Script](#example-submitting-a-python-script)
20. [Additional Observations](#additional-observations)
21. [Summary](#summary)

<br>

---

## Full Source Code

```scala

....

/**
 * Whether to submit, kill, or request the status of an application.
 * The latter two operations are currently supported only for standalone and Mesos cluster modes.
 */
private[deploy] object SparkSubmitAction extends Enumeration {
  type SparkSubmitAction = Value
  val SUBMIT, KILL, REQUEST_STATUS, PRINT_VERSION = Value
}

/**
 * Main gateway of launching a Spark application.
 *
 * This program handles setting up the classpath with relevant Spark dependencies and provides
 * a layer over the different cluster managers and deploy modes that Spark supports.
 */
private[spark] class SparkSubmit extends Logging {

  import DependencyUtils._
  import SparkSubmit._

  def doSubmit(args: Array[String]): Unit = {
    // Initialize logging ...
    val uninitLog = initializeLogIfNecessary(true, silent = true)

    val appArgs = parseArguments(args)
    if (appArgs.verbose) {
      logInfo(appArgs.toString)
    }
    appArgs.action match {
      case SparkSubmitAction.SUBMIT => submit(appArgs, uninitLog)
      case SparkSubmitAction.KILL => kill(appArgs)
      case SparkSubmitAction.REQUEST_STATUS => requestStatus(appArgs)
      case SparkSubmitAction.PRINT_VERSION => printVersion()
    }
  }

  protected def parseArguments(args: Array[String]): SparkSubmitArguments = {
    new SparkSubmitArguments(args)
  }

  private def kill(args: SparkSubmitArguments): Unit = {
    if (RestSubmissionClient.supportsRestClient(args.master)) {
      new RestSubmissionClient(args.master)
        .killSubmission(args.submissionToKill)
    } else {
      val sparkConf = args.toSparkConf()
      sparkConf.set("spark.master", args.master)
      SparkSubmitUtils
        .getSubmitOperations(args.master)
        .kill(args.submissionToKill, sparkConf)
    }
  }

  private def requestStatus(args: SparkSubmitArguments): Unit = {
    if (RestSubmissionClient.supportsRestClient(args.master)) {
      new RestSubmissionClient(args.master)
        .requestSubmissionStatus(args.submissionToRequestStatusFor)
    } else {
      val sparkConf = args.toSparkConf()
      sparkConf.set("spark.master", args.master)
      SparkSubmitUtils
        .getSubmitOperations(args.master)
        .printSubmissionStatus(args.submissionToRequestStatusFor, sparkConf)
    }
  }

  private def printVersion(): Unit = {
    logInfo("""Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version %s
      /_/
    """.format(SPARK_VERSION))
    logInfo("Using Scala %s, %s, %s"
      .format(Properties.versionString, Properties.javaVmName, Properties.javaVersion))
    logInfo(s"Branch $SPARK_BRANCH")
    logInfo(s"Compiled by user $SPARK_BUILD_USER on $SPARK_BUILD_DATE")
    logInfo(s"Revision $SPARK_REVISION")
    logInfo(s"Url $SPARK_REPO_URL")
    logInfo("Type --help for more information.")
  }

  @tailrec
  private def submit(args: SparkSubmitArguments, uninitLog: Boolean): Unit = {

    def doRunMain(): Unit = {
      if (args.proxyUser != null) {
        val isKubernetesClusterModeDriver = args.master.startsWith("k8s") &&
          "client".equals(args.deployMode) &&
          args.toSparkConf().getBoolean("spark.kubernetes.submitInDriver", false)
        if (isKubernetesClusterModeDriver) {
          logInfo("Running driver with proxy user. Cluster manager: Kubernetes")
          SparkHadoopUtil.get.runAsSparkUser(() => runMain(args, uninitLog))
        } else {
          val proxyUser = UserGroupInformation.createProxyUser(
            args.proxyUser,
            UserGroupInformation.getCurrentUser())
          try {
            proxyUser.doAs(new PrivilegedExceptionAction[Unit]() {
              override def run(): Unit = {
                runMain(args, uninitLog)
              }
            })
          } catch {
            case e: Exception =>
              if (e.getStackTrace().length == 0) {
                error(s"ERROR: ${e.getClass().getName()}: ${e.getMessage()}")
              } else {
                throw e
              }
          }
        }
      } else {
        runMain(args, uninitLog)
      }
    }

    if (args.isStandaloneCluster && args.useRest) {
      try {
        logInfo("Running Spark using the REST application submission protocol.")
        doRunMain()
      } catch {
        case e: SubmitRestConnectionException =>
          logWarning(s"Master endpoint ${args.master} was not a REST server. " +
            "Falling back to legacy submission gateway instead.")
          args.useRest = false
          submit(args, false)
      }
    } else {
      doRunMain()
    }
  }

  private[deploy] def prepareSubmitEnvironment(
      args: SparkSubmitArguments,
      conf: Option[HadoopConfiguration] = None)
      : (Seq[String], Seq[String], SparkConf, String) = {
    val childArgs = new ArrayBuffer[String]()
    val childClasspath = new ArrayBuffer[String]()
    val sparkConf = args.toSparkConf()
    if (sparkConf.contains("spark.local.connect")) sparkConf.remove("spark.remote")
    var childMainClass = ""

    val clusterManager: Int = args.maybeMaster match {
      case Some(v) =>
        assert(args.maybeRemote.isEmpty || sparkConf.contains("spark.local.connect"))
        v match {
          case "yarn" => YARN
          case m if m.startsWith("spark") => STANDALONE
          case m if m.startsWith("mesos") => MESOS
          case m if m.startsWith("k8s") => KUBERNETES
          case m if m.startsWith("local") => LOCAL
          case _ =>
            error("Master must either be yarn or start with spark, mesos, k8s, or local")
            -1
        }
      case None => LOCAL
    }

    val deployMode: Int = args.deployMode match {
      case "client" | null => CLIENT
      case "cluster" => CLUSTER
      case _ =>
        error("Deploy mode must be either client or cluster")
        -1
    }

    ...

    // The method is quite large, see the detailed commentary in the notes below.

    ...
  }

  private def runMain(args: SparkSubmitArguments, uninitLog: Boolean): Unit = {
    val (childArgs, childClasspath, sparkConf, childMainClass) = prepareSubmitEnvironment(args)

    if (uninitLog) {
      Logging.uninitialize()
    }

    if (args.verbose) {
      logInfo(s"Main class:\n$childMainClass")
      logInfo(s"Arguments:\n${childArgs.mkString("\n")}")
      logInfo(s"Spark config:\n${Utils.redact(sparkConf.getAll.toMap).sorted.mkString("\n")}")
      logInfo(s"Classpath elements:\n${childClasspath.mkString("\n")}")
      logInfo("\n")
    }

    val loader = getSubmitClassLoader(sparkConf)
    for (jar <- childClasspath) {
      addJarToClasspath(jar, loader)
    }

    var mainClass: Class[_] = null

    try {
      mainClass = Utils.classForName(childMainClass)
    } catch {
      case e: ClassNotFoundException =>
        logError(s"Failed to load class $childMainClass.")
        throw new SparkUserAppException(CLASS_NOT_FOUND_EXIT_STATUS)
      case e: NoClassDefFoundError =>
        logError(s"Failed to load $childMainClass: ${e.getMessage()}")
        throw new SparkUserAppException(CLASS_NOT_FOUND_EXIT_STATUS)
    }

    val app: SparkApplication = if (classOf[SparkApplication].isAssignableFrom(mainClass)) {
      mainClass.getConstructor().newInstance().asInstanceOf[SparkApplication]
    } else {
      new JavaMainApplication(mainClass)
    }

    @tailrec
    def findCause(t: Throwable): Throwable = t match {
      case e: UndeclaredThrowableException =>
        if (e.getCause() != null) findCause(e.getCause()) else e
      case e: InvocationTargetException =>
        if (e.getCause() != null) findCause(e.getCause()) else e
      case e: Throwable =>
        e
    }

    try {
      app.start(childArgs.toArray, sparkConf)
    } catch {
      case t: Throwable =>
        throw findCause(t)
    } finally {
      if (args.master.startsWith("k8s") && !isShell(args.primaryResource) && ...) {
        ...
      }
    }
  }

  private def error(msg: String): Unit = throw new SparkException(msg)

}

/**
 * This entry point is used by the launcher library to start in-process Spark applications.
 */
private[spark] object InProcessSparkSubmit {
  def main(args: Array[String]): Unit = {
    val submit = new SparkSubmit()
    submit.doSubmit(args)
  }
}

object SparkSubmit extends CommandLineUtils with Logging {
  ...
  // Implementation includes the main method for spark-submit, constants, helper methods, etc.
  ...
}

private[spark] object SparkSubmitUtils extends Logging {
  ...
  // Utilities for dependency resolution, parsing, logging, etc.
  ...
}

```

> **Note**: The code above is abridged in some sections (`...`) for brevity. Refer to the original code snippet for the full details.

<br>

---

## High-Level Overview

Spark’s `SparkSubmit` class is the **primary entry point** for launching Spark applications via the command line (e.g. `spark-submit`). Its main tasks are:

1. **Parsing command-line arguments** to figure out how the application should be launched.  
2. **Deciding on the cluster manager** (e.g. YARN, Kubernetes, Mesos, Standalone, local).  
3. **Preparing the environment** (classpath, local vs. remote resources, proxy user, etc.).  
4. **Invoking the actual main class** of the user application (or a specialized runner for Python/R apps).  

The internal design includes helper objects like `SparkSubmitUtils` for dependency resolution and the `SparkSubmitAction` enumeration for the different actions (submit/kill/status/version).

---

## Key Classes & Objects

### SparkSubmitAction (Enumeration)

```scala
private[deploy] object SparkSubmitAction extends Enumeration {
  type SparkSubmitAction = Value
  val SUBMIT, KILL, REQUEST_STATUS, PRINT_VERSION = Value
}
```

- **Purpose**: Enumerates possible actions that `spark-submit` can perform:
  1. **SUBMIT**: Submit a new Spark job.
  2. **KILL**: Terminate an existing Spark job (for Standalone or Mesos).
  3. **REQUEST_STATUS**: Check status of a submission.
  4. **PRINT_VERSION**: Print Spark version info.

### SparkSubmit (Class)

```scala
private[spark] class SparkSubmit extends Logging {
  def doSubmit(args: Array[String]): Unit = { ... }
  protected def parseArguments(args: Array[String]): SparkSubmitArguments = { ... }
  private def kill(args: SparkSubmitArguments): Unit = { ... }
  private def requestStatus(args: SparkSubmitArguments): Unit = { ... }
  private def printVersion(): Unit = { ... }
  private def submit(args: SparkSubmitArguments, uninitLog: Boolean): Unit = { ... }
  private[deploy] def prepareSubmitEnvironment(...): (Seq[String], Seq[String], SparkConf, String)
  private def runMain(args: SparkSubmitArguments, uninitLog: Boolean): Unit = { ... }
  private def error(msg: String): Unit = throw new SparkException(msg)
}
```

- **Main tasks**:
  1. **`doSubmit`**: Orchestrates the entire submission process.
  2. **`parseArguments`**: Creates an instance of `SparkSubmitArguments` from command-line input.
  3. **`submit(...)`**: Handles either the direct submission or fallback from REST to legacy submission methods for certain cluster types.
  4. **`prepareSubmitEnvironment`**: Sets up the environment (classpath, resources, dependencies) for the submission.
  5. **`runMain`**: Actually runs the application’s main class in the correct classloader context.

### SparkSubmitUtils (Object)

```scala
private[spark] object SparkSubmitUtils extends Logging {
  // Contains utility methods for:
  //  - Resolving Maven dependencies (via Ivy).
  //  - Parsing spark config properties.
  //  - Loading or building Ivy settings.
  //  - Downloading remote resources.
}
```

- **Key Methods**:
  - `resolveMavenCoordinates(...)`: Downloads JARs from Maven coordinates.
  - `buildIvySettings(...)`: Creates Ivy settings with custom resolvers (like `spark-packages`).
  - `createExclusion(...)`: Helps skip certain dependencies that Spark already includes.

---
## Detailed Walkthrough

### 1. Entry Points and Flow

1. **`SparkSubmit.main(args)`** is what the `spark-submit` script calls.  
2. It constructs a `SparkSubmit` instance and invokes `doSubmit(args)`.  
3. `doSubmit`:
   - Initializes logging (if needed).
   - Parses arguments into a `SparkSubmitArguments` instance.
   - Decides which action to take (`SUBMIT`, `KILL`, `REQUEST_STATUS`, `PRINT_VERSION`).  
4. For `SUBMIT`, it calls `submit(...)` (which might pass to REST or legacy submission).

### 2. Argument Parsing

- Handled by `SparkSubmitArguments`, not shown in detail here, but it extracts things like:
  - `--master`, `--class`, `--deploy-mode`
  - `--conf`, `--packages`, `--files`, `--jars`
  - `--py-files`, `--archives`
  - `--verbose`
  - etc.
- It also merges with defaults from the Spark configuration.

### 3. The `doSubmit` Method

```scala
def doSubmit(args: Array[String]): Unit = {
  val uninitLog = initializeLogIfNecessary(true, silent = true)
  val appArgs = parseArguments(args)
  if (appArgs.verbose) {
    logInfo(appArgs.toString)
  }
  appArgs.action match {
    case SparkSubmitAction.SUBMIT => submit(appArgs, uninitLog)
    case SparkSubmitAction.KILL => kill(appArgs)
    case SparkSubmitAction.REQUEST_STATUS => requestStatus(appArgs)
    case SparkSubmitAction.PRINT_VERSION => printVersion()
  }
}
```

**Key Points**:
- `uninitLog` keeps track of whether the logging should be reset before the application’s main class.
- `appArgs.action` is decided by the `SparkSubmitArguments` object.

### 4. The `parseArguments` Method

```scala
protected def parseArguments(args: Array[String]): SparkSubmitArguments = {
  new SparkSubmitArguments(args)
}
```
- Simply creates a new `SparkSubmitArguments` object. If you look inside `SparkSubmitArguments`, you’ll find how each command-line flag is mapped.

### 5. Submit, Kill, Request Status, Print Version

- `kill(appArgs)`: If the master supports the REST client, use `RestSubmissionClient`. Otherwise, use a specific `SparkSubmitOperation`.
- `requestStatus(appArgs)`: Similarly checks status.
- `printVersion()`: Logs version info.

### 6. Preparing Submit Environment

**`prepareSubmitEnvironment`** is quite large. Here are the highlights:

```scala
private[deploy] def prepareSubmitEnvironment(
    args: SparkSubmitArguments,
    conf: Option[HadoopConfiguration] = None)
    : (Seq[String], Seq[String], SparkConf, String) = {
  // 1) Create a SparkConf from the arguments
  // 2) Determine cluster manager (YARN, STANDALONE, MESOS, KUBERNETES, LOCAL)
  // 3) Decide deploy mode (client or cluster)
  // 4) Check validity for python or R submissions
  // 5) Possibly resolve Maven coordinates & handle transitive dependencies
  // 6) Download remote resources
  // 7) Possibly adjust classpath entries
  // 8) Set up SparkConf keys for further usage
  // 9) Return a tuple: (childArgs, childClasspath, sparkConf, mainClass)
}
```

1. **Cluster manager logic**: `args.maybeMaster.map(...) => YARN | STANDALONE | MESOS | LOCAL | KUBERNETES`.
2. **Deploy mode**: `client` or `cluster`.
3. **Resolve dependencies**: 
   - If user passed `--packages`, calls `DependencyUtils.resolveMavenDependencies(...)`.
   - Merges them with existing JARs/py-files as needed.
4. **Download**:
   - If user references remote `--jars`, `--files`, etc. these may be downloaded locally (especially in `client` mode).
5. **In the case of `yarn-cluster`** or `k8s` cluster, sets up the main class differently (like `org.apache.spark.deploy.yarn.YarnClusterApplication`).
6. Adjusts environment, merges Spark conf, sets the final main class to run.

### 7. The `runMain` Method

```scala
private def runMain(args: SparkSubmitArguments, uninitLog: Boolean): Unit = {
  // Step 1: prepare environment
  val (childArgs, childClasspath, sparkConf, childMainClass) = prepareSubmitEnvironment(args)

  // Step 2: Possibly uninitialize Spark's logging, so user app can re-init logs
  if (uninitLog) {
    Logging.uninitialize()
  }

  // Step 3: If verbose, log environment details
  // Step 4: Set up classloader and add all jars to it
  val loader = getSubmitClassLoader(sparkConf)
  for (jar <- childClasspath) { addJarToClasspath(jar, loader) }

  // Step 5: Load the final main class & instantiate if needed
  var mainClass: Class[_] = null
  try {
    mainClass = Utils.classForName(childMainClass)
  } catch {
    case e: ClassNotFoundException => ...
  }

  val app: SparkApplication = if (classOf[SparkApplication].isAssignableFrom(mainClass)) {
    mainClass.getConstructor().newInstance().asInstanceOf[SparkApplication]
  } else {
    new JavaMainApplication(mainClass)
  }

  // Step 6: Actually run app.start(childArgs, sparkConf)
}
```

**Key Points**:
- The new `SparkConf` is used.  
- The class is loaded reflectively, ensuring the user’s main class can be invoked.  
- If the main class implements `SparkApplication`, its `start` method is called; otherwise, it uses `JavaMainApplication` to run a standard main method.

### 8. Maven Dependency Resolution Mechanics

The `SparkSubmitUtils` object has methods like `resolveMavenCoordinates` which use **Ivy** under the hood. Steps typically include:

1. Building an **Ivy settings** object that points to standard repositories (`Central`, `spark-packages`, local `.m2`, etc.).
2. Creating a temporary `DefaultModuleDescriptor`.
3. Adding dependencies (from `--packages`) and applying exclusion rules for Spark’s own artifacts.
4. Resolving & retrieving the JARs.
5. Merging these jars with the user’s existing list.

<br>

---

## Sequence Diagram of `SparkSubmit`

```plaintext
 +----------------------+           +-------------------------+          +------------------+
 |  spark-submit (CLI)  |           |    SparkSubmit Class   |          |   Spark Cluster  |
 +-----------+----------+           +------------+------------+          +--------+---------+
             |                                           |                      |
             | 1) spark-submit [args] -> calls main()    |                      |
             |------------------------------------------>|                      |
             |                                           |                      |
             | 2) doSubmit(args)                         |                      |
             |------------------------------------------>|                      |
             |                                           |                      |
             |   parseArguments -> SparkSubmitArguments  |                      |
             |------------------------------------------>|                      |
             |   action match (SUBMIT, etc.)             |                      |
             |------------------------------------------>|                      |
             |                                           |                      |
             | 3) submit(...) -> prepare environment     |                      |
             |------------------------------------------>|                      |
             |   check cluster manager / resources / etc.|                      |
             |------------------------------------------>|                      |
             |   runMain(...)                            |                      |
             |------------------------------------------>|                      |
             |   load user main class & invoke           |                      |
             |------------------------------------------>|                      |
             |                                           |    4) connect to     |
             |                                           |--------> cluster mgmt|
             |                                           |           service    |
             |                                           |                      |
             | 5) app logic runs in driver / cluster     |<---------------------|
             |-------------------------------------------|                      |
             |                                           |                      |
 +-----------+----------+           +------------+------------+          +--------+---------+
 |     End of CLI       |           |  spark-submit ends   |            |  Cluster executes|
 +----------------------+           +-----------------------+            +------------------+
```

<br>

---

## Examples & Usage

### 1. Example: Submitting a Scala/Java JAR

```bash
spark-submit \
  --class org.mycompany.MyApp \
  --master spark://spark-master:7077 \
  --deploy-mode cluster \
  --jars additional.jar,dependency.jar \
  --executor-memory 4G \
  path/to/my-application.jar \
  arg1 arg2
```

**Notes**:
- The code in `SparkSubmit` looks at the `--class` to know which main class to invoke.
- The `--jars` are merged with existing classpath.

### 2. Example: Submitting a Python Script

```bash
spark-submit \
  --master local[4] \
  --py-files helper.zip \
  my_script.py \
  --script-arg1 foo
```

- Internally, `isPython()` is `true`, so `SparkSubmit` runs `org.apache.spark.deploy.PythonRunner` with `my_script.py`.
- `--py-files` are added to `PYTHONPATH`.

<br>

---

## Additional Observations

1. **Proxy User**: There's special logic for running as a proxy user for certain cluster managers (`UserGroupInformation.createProxyUser`).  
2. **Kubernetes**: Special code handles “cluster-mode” driver logic by checking `spark.kubernetes.submitInDriver`.  
3. **Logging**: The standard Spark logging is partially re-initialized so the user’s application can have its own logging.  

<br>

---

## Summary

The `SparkSubmit` class is the backbone of Spark’s submission flow. It carefully orchestrates:
- **Argument parsing**  
- **Dependency resolution**  
- **Resource downloading** (in `client` mode)  
- **Classloader creation**  
- **Invocation** of the actual main method (or special runner for Python/R)

This design allows Spark to **abstract** away the complexities of different cluster managers (YARN, Kubernetes, Mesos, Standalone, local), while exposing consistent user-facing arguments (`spark-submit`) that lead to the final application launching procedure.

By exploring each method’s responsibilities, you gain an inside look at how Spark transitions from a command-line invocation to a running cluster application with the correct environment, dependencies, and logic.

> **Next Steps**:
> - Dive deeper into `SparkSubmitArguments` to see how each CLI flag is parsed.
> - Explore `SparkHadoopUtil` and how Spark deals with `UserGroupInformation` for security.
> - Look at `SparkSubmitUtils` in more depth to understand the Ivy-based dependency resolution.

<br>

---

**License**: [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)  

```text
© 2023 The Apache Software Foundation.
This document is provided for educational purposes, summarizing code licensed under Apache 2.0.
```
