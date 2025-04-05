
https://juejin.cn/post/7230468104617328700
## Preface

This article, authored by Mars, provides an in-depth look into using Quartz for scheduled tasks in a distributed architecture. Quartz is a powerful, open-source job scheduling library that can be integrated into virtually any Java application. Unlike monolithic architectures, distributed architectures require more sophisticated scheduling solutions, and Quartz delivers with its clear structure and ease of use.

## Character Introduction

To get started with Quartz, you need to understand three main components:

1. **Scheduler**: Manages the scheduling of jobs.
2. **Job**: Represents the task to be performed. This is an interface that requires implementing the `execute` method.
3. **Trigger**: Determines the schedule of job execution. Key types include `SimpleTrigger` for straightforward schedules and `CronTrigger` for cron expression-based schedules.

## Official Examples

Quartz provides comprehensive examples to illustrate its usage. Here is an overview of a simple example with two Java files: `HelloJob` and `SimpleExample`.

### HelloJob.java

```java
package org.quartz.examples.example1;

import java.util.Date;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.quartz.Job;
import org.quartz.JobExecutionContext;
import org.quartz.JobExecutionException;

/**
 * A simple job that says "Hello" to the world.
 */
public class HelloJob implements Job {

    private static Logger _log = LoggerFactory.getLogger(HelloJob.class);

    public HelloJob() {
    }

    public void execute(JobExecutionContext context) throws JobExecutionException {
        _log.info("Hello World! - " + new Date());
    }
}
```

### SimpleExample.java

```java
package org.quartz.examples.example1;

import org.quartz.JobDetail;
import org.quartz.Scheduler;
import org.quartz.SchedulerFactory;
import org.quartz.Trigger;
import org.quartz.impl.StdSchedulerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Date;

import static org.quartz.DateBuilder.evenMinuteDate;
import static org.quartz.JobBuilder.newJob;
import static org.quartz.TriggerBuilder.newTrigger;

public class SimpleExample {

    public void run() throws Exception {
        Logger log = LoggerFactory.getLogger(SimpleExample.class);

        log.info("------- Initializing ----------------------");

        // 1. Create a Scheduler
        SchedulerFactory sf = new StdSchedulerFactory();
        Scheduler sched = sf.getScheduler();

        log.info("------- Initialization Complete -----------");

        Date runTime = evenMinuteDate(new Date());

        log.info("------- Scheduling Job  -------------------");

        // 2. Specify a Job
        JobDetail job = newJob(HelloJob.class).withIdentity("job1", "group1").build();

        // 3. Specify a Trigger
        Trigger trigger = newTrigger().withIdentity("trigger1", "group1").startAt(runTime).build();

        // 4. Bind Job and Trigger
        sched.scheduleJob(job, trigger);
        log.info(job.getKey() + " will run at: " + runTime);

        // 5. Execute
        sched.start();

        log.info("------- Started Scheduler -----------------");

        log.info("------- Waiting 65 seconds... -------------");
        try {
            Thread.sleep(65L * 1000L);
        } catch (Exception e) {
        }

        // Shutdown the scheduler
        log.info("------- Shutting Down ---------------------");
        sched.shutdown(true);
        log.info("------- Shutdown Complete -----------------");
    }

    public static void main(String[] args) throws Exception {
        SimpleExample example = new SimpleExample();
        example.run();
    }
}
```

## Key Steps in Using Quartz

1. **Create a Scheduler**: Instantiate a scheduler using a `SchedulerFactory`.
2. **Create a Job**: Define a job by implementing the `Job` interface and passing it to a `JobDetail` object.
3. **Create a Trigger**: Define the schedule for the job using a `Trigger` object.
4. **Bind Job and Trigger**: Schedule the job with the trigger using the scheduler.
5. **Execute the Scheduler**: Start the scheduler to begin job execution.

## Quartz in Distributed Systems

### How is Quartz Distributed?

Quartz achieves distributed scheduling through database-backed locks. The core tables used in Quartz are:

| Table Name | Functional Description |
|------------|------------------------|
| `QRTZ_CALENDARS` | Stores Quartz Calendar information |
| `QRTZ_CRON_TRIGGERS` | Stores `CronTrigger`, including cron expression and time zone information |
| `QRTZ_FIRED_TRIGGERS` | Stores status information related to triggered `Trigger` and associated `Job` execution |
| `QRTZ_PAUSED_TRIGGER_GRPS` | Stores information about paused trigger groups |
| `QRTZ_SCHEDULER_STATE` | Stores information about the state of the scheduler and other scheduler instances |
| `QRTZ_LOCKS` | Stores pessimistic locking information |
| `QRTZ_JOB_DETAILS` | Stores detailed information about each configured job |
| `QRTZ_SIMPLE_TRIGGERS` | Stores simple triggers, including repetition count and interval |
| `QRTZ_TRIGGERS` | Stores information about configured triggers |

### Example SQL Table Creation

Find the appropriate SQL script for your database and execute it to create the necessary tables.

### Backtracking from `HelloJob.execute`

1. **Execute Method**: The entry point for job execution.
2. **JobRunShell**: The core component calling the `execute` method.
3. **QuartzSchedulerThread**: Handles trigger acquisition and job scheduling.

### Trigger Processing
The `JobStore` interface is key for trigger management. Implementations include `RAMJobStore` (for in-memory storage) and `JobStoreSupport` (for database-backed storage).

```java
protected <T> T executeInNonManagedTXLock(
        String lockName, 
        TransactionCallback<T> txCallback, final TransactionValidator<T> txValidator) throws JobPersistenceException {
    boolean transOwner = false;
    Connection conn = null;
    try {
        if (lockName != null) {
            if (getLockHandler().requiresConnection()) {
                conn = getNonManagedTXConnection();
            }
            transOwner = getLockHandler().obtainLock(conn, lockName);
        }

        if (conn == null) {
            conn = getNonManagedTXConnection();
        }

        final T result = txCallback.execute(conn);
        try {
            commitConnection(conn);
        } catch (JobPersistenceException e) {
            rollbackConnection(conn);
            if (txValidator == null || !retryExecuteInNonManagedTXLock(lockName, new TransactionCallback<Boolean>() {
                @Override
                public Boolean execute(Connection conn) throws JobPersistenceException {
                    return txValidator.validate(conn, result);
                }
            })) {
                throw e;
            }
        }

        Long sigTime = clearAndGetSignalSchedulingChangeOnTxCompletion();
        if (sigTime != null && sigTime >= 0) {
            signalSchedulingChangeImmediately(sigTime);
        }

        return result;
    } catch (JobPersistenceException e) {
        rollbackConnection(conn);
        throw e;
    } catch (RuntimeException e) {
        rollbackConnection(conn);
        throw new JobPersistenceException("Unexpected runtime exception: " + e.getMessage(), e);
    } finally {
        try {
            releaseLock(lockName, transOwner);
        } finally {
            cleanupConnection(conn);
        }
    }
}
```
### Combined Process
1. **Lock Acquisition**: Obtain a lock in the `QRTZ_LOCKS` table.
2. **Job Execution**: Execute the `JobRunShell` to perform the task.
3. **Lock Release**: Release the lock after task completion.
Each scheduler instance in the cluster follows this process, ensuring distributed coordination.
## Summary
Quartz is a robust scheduling framework suitable for distributed systems. By using database-backed locks, Quartz ensures that only one scheduler instance accesses the `Trigger` at a time, maintaining system performance and reliability. The distributed scheduling principle is implemented through these database locks and a pessimistic locking strategy.
### Key Points

- **Scheduler**: Manages job scheduling.
- **Job**: Represents the task to be executed.
- **Trigger**: Determines the execution schedule.
- **Distributed Scheduling**: Achieved through database locks and a pessimistic locking strategy.

Quartz provides a clear and powerful framework for scheduling tasks in both monolithic and distributed systems, ensuring high performance and reliability.

---