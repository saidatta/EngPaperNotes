The `ThreadLocal` class in Java enables you to create variables that can only be read and written by the same thread. Thus, even if two threads are executing the same code, and the code has a reference to a `ThreadLocal` variable, then the two threads cannot see each other's `ThreadLocal` variables.

This feature of `ThreadLocal` can be useful in multi-threaded programming when you need to maintain thread confinement, where data is only accessible from a single thread, avoiding potential synchronization issues.

## Basic Usage of ThreadLocal

A `ThreadLocal` is like a "box" that can hold a single item, and it gives you access to put an item into the box and later get it back.

Here is a simple example:

```java
public class ThreadLocalExample {

    public static class MyRunnable implements Runnable {

        private ThreadLocal<Integer> threadLocal = new ThreadLocal<Integer>();

        @Override
        public void run() {
            threadLocal.set((int) (Math.random() * 100D));

            try {
                Thread.sleep(2000);
            } catch (InterruptedException e) {
            }

            System.out.println(threadLocal.get());
        }
    }

    public static void main(String[] args) {
        MyRunnable myRunnable = new MyRunnable();

        Thread thread1 = new Thread(myRunnable);
        Thread thread2 = new Thread(myRunnable);

        thread1.start();
        thread2.start();
    }
}
```

In this example, two threads are started and both run the same `Runnable` instance. The `Runnable` sets a random integer to the `ThreadLocal` instance and then sleeps for two seconds before printing the integer value to the console. Because the `ThreadLocal` keeps a separate value for each thread, the two threads will always print out different values, despite them sharing the same `ThreadLocal` instance from the `Runnable`.

## ThreadLocal and Memory Leaks

While `ThreadLocal` is a powerful feature, it can also introduce memory leaks if not used properly. When a thread terminates but has a reference to a `ThreadLocal` object, the `ThreadLocal` value will be kept in memory until the entire thread is garbage collected, which might not happen if the Thread is contained in a thread pool.

To prevent potential memory leaks, `ThreadLocal` provides a `remove()` method that removes the value for the current thread. You should call this method whenever you're done with using the `ThreadLocal` value in the current thread, like in a `finally` block after the work of the thread is done.

```java
try {
    // ... work with threadLocal
} finally {
    threadLocal.remove();
}
```

## Initial ThreadLocal Value

By default, a `ThreadLocal` variable starts off as `null`. You can change this by overriding the `initialValue()` method. This method will be called once for each thread the first time you access the value in each thread, whether you do it via `get()` or `set()`. Here is an example:

```java
private ThreadLocal<Integer> threadLocal = new ThreadLocal<Integer>() {
    @Override protected Integer initialValue() {
        return new Integer((int) (Math.random() * 100D));
    }
};
```

In this code, the `initialValue()` method has been overridden to return a random integer. 

## Key Takeaways

- `ThreadLocal` is a useful class for maintaining thread confinement, where each thread maintains its own instance of a variable.
- Always remember to clean up with `remove()` after using a `ThreadLocal`, especially in environments with long-lived threads, to prevent memory leaks.
- You can provide an initial value to a `Thread

Local` by overriding the `initialValue()` method.

Sure, let's dive into those usecases:

## Usecase #1 - Per thread instances for memory efficiency and thread-safety

Often, we may have some resources that are expensive to create, like database connections, date formatters, random number generators etc. Also, these resources may not be thread-safe. To tackle this, we can create a new instance of such a resource for each thread. This ensures thread-safety as well as makes our program memory efficient because the resources are reused among the tasks executed by the same thread.

Consider a simple example of a `DateFormat` object. It's not thread-safe, and we might need to use it in multiple threads:

```java
public class DateFormatterThreadLocal {

    // ThreadLocal with initial value
    private static final ThreadLocal<DateFormat> df = ThreadLocal.withInitial(() -> new SimpleDateFormat("yyyy-MM-dd"));

    public String format(Date date) {
        return df.get().format(date); // each thread will have its own DateFormat
    }
}
```

In this example, `df` is a static `ThreadLocal` variable that holds a `DateFormat` object. Each thread will have its own instance of `DateFormat`, ensuring thread-safety. This is a memory efficient way to use instances that are expensive to create.

## Usecase #2 - Per thread context (thread-safety + perf)

Sometimes we have certain objects that need to be used across different methods, or even different classes, within the same thread. Typically, these are context-like objects, like user session data.

Consider a system where a request to our server might pass through several services. Each service might need to know who the current user is. Passing the User object through every method would be cumbersome. Instead, we can attach the User object to the thread so that it can be accessed from anywhere in the thread, much like how Spring Framework uses `ThreadLocal` for request attributes in `RequestContextHolder`.

Here's an example of how we might do that:

```java
public class UserService {
    private static final ThreadLocal<User> currentUser = new ThreadLocal<>();

    public void setCurrentUser(User user) {
        currentUser.set(user);
    }

    public User getCurrentUser() {
        return currentUser.get();
    }

    public void clear() {
        currentUser.remove();
    }
}

public class Service1 {
    private UserService userService;
    
    public void process() {
        User user = userService.getCurrentUser();
        // process something with user
    }
}

public class Service2 {
    private UserService userService;
    
    public void process() {
        User user = userService.getCurrentUser();
        // process something with user
    }
}
```

In this case, `UserService` provides a way to set and get the current user within the thread context. This user object can then be accessed from any services (like `Service1` and `Service2`) in the same thread without passing it through each method.

Remember to clear the `ThreadLocal` variable once it is no longer needed to prevent memory leaks, especially in an environment where threads are pooled.

## References

- Oracle's Java Tutorials: [Thread Objects](https://docs.oracle.com/javase/tutorial/essential/concurrency/threads.html)
- Oracle's Java Docs: [ThreadLocal](https://docs.oracle.com/javase/8/docs/api/java/lang/ThreadLocal.html)

