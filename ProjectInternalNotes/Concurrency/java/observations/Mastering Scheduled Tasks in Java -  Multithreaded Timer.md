## Preface
The default Timer class in Java doesn't support concurrent execution of multiple tasks. To address the concurrency problem in multi-task scheduling, we can leverage multithreading and thread pools. This chapter delves into how to handle these issues using the `ScheduledExecutorService`.
## ScheduledExecutorService
`ScheduledExecutorService` is an interface that extends `ExecutorService`. To create a thread pool for scheduled tasks, we can use the `Executors` utility class:

```java
ScheduledExecutorService scheduledExecutorService = Executors.newScheduledThreadPool(5);
```

## Rewriting the Timer Example

Previously, we encountered blocking issues with the `Timer`. Let's modify the example to use a thread pool instead:
### Original Example Using `Timer`

```java
import java.util.Date;
import java.util.Timer;
import java.util.TimerTask;

public class MarsTimer {
    public static void main(String[] args) {
        Timer timer = new Timer();
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                System.out.println("当前时间:" + new Date());
            }
        }, 1000, 5000);
    }
}
```
### Modified Example Using `ScheduledExecutorService`
```java
import java.util.Date;
import java.util.TimerTask;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class MarsTimer {
    public static void main(String[] args) {
        // Create the first task
        TimerTask tta = new TimerTask() {
            @Override
            public void run() {
                System.out.println(">> 这是a任务：当前时间：" + new Date());
                try {
                    Thread.sleep(5000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        };

        // Create the second task
        TimerTask ttb = new TimerTask() {
            @Override
            public void run() {
                System.out.println("<< 这是b任务：当前毫秒：" + System.currentTimeMillis());
            }
        };

        // Create a thread pool with 5 core threads
        ScheduledExecutorService scheduledExecutorService = Executors.newScheduledThreadPool(5);

        // Schedule the first task with a 5-second period
        scheduledExecutorService.scheduleAtFixedRate(tta, 1, 5, TimeUnit.SECONDS);

        // Schedule the second task with a 5-second period
        scheduledExecutorService.scheduleAtFixedRate(ttb, 1, 5, TimeUnit.SECONDS);
    }
}
```

### Result
Both tasks run independently without blocking each other, solving the issue of blocking caused by shared queues.
## Handling Complex Scheduling
If you need to perform tasks on a monthly or yearly basis, `ScheduledExecutorService` may not suffice due to its limited support for cron expressions. To handle more complex schedules, cron expressions are required.
## Cron Expressions
A cron expression is a string representing a schedule in six or seven fields separated by spaces, with the format `X X X X X X`. The last field (year) is optional. Each field can have multiple values, special characters, and ranges.
### Example
`0 10 1 ? * * 2023` - Runs at 1:10 AM every day in 2023.
### Field Placeholder Values

| Special Character | Meaning                                          | Example                                                     |
| ----------------- | ------------------------------------------------ | ----------------------------------------------------------- |
| `*`               | All possible values                              | `*` in the month field means every month.                   |
| `,`               | Enumeration values                               | `5,20` in the minute field means at 5 and 20 minutes.       |
| `-`               | Range                                            | `5-20` in the minute field means every minute from 5 to 20. |
| `/`               | Increment                                        | `0/15` in the minute field means every 15 minutes.          |
| `?`               | No specific value (date and weekday fields only) | In date field, `?` means no specific value.                 |
| `L`               | Last day                                         | `L` in the date field means the last day of the month.      |
| `W`               | Nearest weekday                                  | `5W` means the nearest weekday to the 5th.                  |
| `#`               | Specific day of the week                         | `4#2` means the second Thursday of the month.               |
### Implementing Cron Expressions in Java
To support cron-like scheduling in Java, we might need a library such as `quartz-scheduler` which supports cron expressions.
## Example with Quartz Scheduler
```java
import org.quartz.*;
import org.quartz.impl.StdSchedulerFactory;

import static org.quartz.JobBuilder.newJob;
import static org.quartz.TriggerBuilder.newTrigger;
import static org.quartz.CronScheduleBuilder.cronSchedule;

public class QuartzExample {
    public static void main(String[] args) throws SchedulerException {
        // Define a job and bind it to our job class
        JobDetail job = newJob(MyJob.class)
                .withIdentity("job1", "group1")
                .build();

        // Define a trigger that runs according to a cron expression
        Trigger trigger = newTrigger()
                .withIdentity("trigger1", "group1")
                .withSchedule(cronSchedule("0 0 12 * * ?"))
                .build();

        // Schedule the job with the trigger
        Scheduler scheduler = new StdSchedulerFactory().getScheduler();
        scheduler.start();
        scheduler.scheduleJob(job, trigger);
    }

    public static class MyJob implements Job {
        public void execute(JobExecutionContext context) throws JobExecutionException {
            System.out.println("Executing job at " + new Date());
        }
    }
}
```
### Result
The job will run every day at noon.
## Summary
- **ScheduledExecutorService**: Suitable for simple periodic tasks with millisecond precision.
- **Cron Expressions**: Needed for complex schedules (e.g., monthly, yearly).
- **Quartz Scheduler**: Supports cron expressions and provides advanced scheduling capabilities.
By leveraging these tools, you can handle multithreaded, concurrent scheduled tasks effectively in Java.