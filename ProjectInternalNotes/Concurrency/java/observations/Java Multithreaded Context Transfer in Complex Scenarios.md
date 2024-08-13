## I. Introduction
When expanding an overseas mall, starting from India and gradually moving to other countries, it is essential to support multiple languages, countries, time zones, and localization. A critical challenge is to propagate the identified country information through different layers of the system, especially in multithreaded scenarios.
## II. Background Technology
### 2.1 ThreadLocal
`ThreadLocal` is a simple way to store per-thread data. It allows us to associate state with a thread without passing this state explicitly through the code. Here’s a brief overview of how `ThreadLocal` works:
```java
public void set(T value) {
    Thread t = Thread.currentThread();
    ThreadLocalMap map = getMap(t);
    if (map != null)
        map.set(this, value);
    else
        createMap(t, value);
}

public T get() {
    Thread t = Thread.currentThread();
    ThreadLocalMap map = getMap(t);
    if (map != null) {
        ThreadLocalMap.Entry e = map.getEntry(this);
        if (e != null) {
            @SuppressWarnings("unchecked")
            T result = (T)e.value;
            return result;
        }
    }
    return setInitialValue();
}

ThreadLocalMap getMap(Thread t) {
    return t.threadLocals;
}

private Entry getEntry(ThreadLocal<?> key) {
    int i = key.threadLocalHashCode & (table.length - 1);
    Entry e = table[i];
    if (e != null && e.get() == key)
        return e;
    else
        return getEntryAfterMiss(key, i, e);
}
```

Each `Thread` has its own `ThreadLocalMap` containing a weakly referenced `Entry` (ThreadLocal, Object).
- The `get` method retrieves the value from the `ThreadLocalMap`.
- The `set` method updates the value in the `ThreadLocalMap`.
### 2.2 InheritableThreadLocal
`InheritableThreadLocal` allows child threads to inherit values from their parent thread.
```java
public class InheritableThreadLocal<T> extends ThreadLocal<T> {
    protected T childValue(T parentValue) {
        return parentValue;
    }

    ThreadLocalMap getMap(Thread t) {
       return t.inheritableThreadLocals;
    }

    void createMap(Thread t, T firstValue) {
        t.inheritableThreadLocals = new ThreadLocalMap(this, firstValue);
    }
}
```
When a new thread is created, it checks if the parent thread has `inheritableThreadLocals`. If it does, it copies the data to the child thread.
### Issues with Thread Pools
When using thread pools, `InheritableThreadLocal` doesn’t work as expected because threads are reused. The inherited values are only copied during thread creation.
### 2.3 TransmittableThreadLocal
`TransmittableThreadLocal` (TTL) solves the problem of passing thread-local variables when using thread pools. It ensures that thread-local variables are copied from the parent thread to the child thread every time a task is submitted.

```java
static TransmittableThreadLocal<String> transmittableThreadLocal = new TransmittableThreadLocal<>();

public static void main(String[] args) throws InterruptedException {
    ExecutorService executorService = Executors.newFixedThreadPool(1);
    executorService = TtlExecutors.getTtlExecutorService(executorService);

    transmittableThreadLocal.set("I am a transmittable parent");
    executorService.execute(() -> {
        System.out.println(transmittableThreadLocal.get());
        transmittableThreadLocal.set("I am an old transmittable parent");
    });
    System.out.println(transmittableThreadLocal.get());

    TimeUnit.SECONDS.sleep(1);
    transmittableThreadLocal.set("I am a new transmittable parent");

    executorService.execute(() -> {
        System.out.println(transmittableThreadLocal.get());
    });
}
```

## III. Practical Application of TTL in Overseas Shopping Malls
### 3.1 Data Row + SpringMVC
1. **HTTP Request Handling**: Extract country information from URL or cookies and store it in `TransmittableThreadLocal`.
```java
public class ShopShardingHelperUtil {
    private static TransmittableThreadLocal<String> countrySet = new TransmittableThreadLocal<>();

    public static String getCountry() {
        return countrySet.get();
    }

    public static void setCountry(String country) {
        countrySet.set(country.toLowerCase());
    }

    public static void clear() {
        countrySet.remove();
    }
}
```

2. **Custom Thread Pool**: Use `TtlExecutors` to wrap the original custom thread pool.
```java
public static Executor getExecutor() {
    if (executor == null) {
        synchronized (TransmittableExecutor.class) {
            if (executor == null) {
                executor = TtlExecutors.getTtlExecutor(initExecutor());
            }
        }
    }
    return executor;
}
```
3. **MyBatis Interceptor**: Use `TransmittableThreadLocal` to get the country information and modify SQL queries accordingly.
```java
public Object intercept(Invocation invocation) throws Throwable {
    StatementHandler statementHandler = (StatementHandler) invocation.getTarget();
    BoundSql boundSql = statementHandler.getBoundSql();
    String originalSql = boundSql.getSql();
    Statement statement = (Statement) CCJSqlParserUtil.parse(originalSql);
    String threadCountry = ShopShardingHelperUtil.getCountry();

    if (StringUtils.isNotBlank(threadCountry)) {
        // Modify the statement based on country
    }

    Field boundSqlField = BoundSql.class.getDeclaredField("sql");
    boundSqlField.setAccessible(true);
    boundSqlField.set(boundSql, statement.toString());
    return invocation.proceed();
}
```
### 3.2 Database + SpringBoot
1. **Async Configuration**: Configure custom thread pools in SpringBoot with TTL.
```java
@Bean
public ThreadPoolTaskExecutor threadPoolTaskExecutor(){
    return TtlThreadPoolExecutors.getAsyncExecutor();
}
```
2. **Localization Context**: Use `TransmittableThreadLocal` for localization context.
```java
public class LocalizationContextHolder {
    private static TransmittableThreadLocal<LocalizationContext> localizationContextHolder = new TransmittableThreadLocal<>();

    public static LocalizationContext getLocalizationContext() {
        return localizationContextHolder.get();
    }

    public static void setLocalizationContext(LocalizationContext localizationContext) {
        localizationContextHolder.set(localizationContext);
    }
}
```
3. **HTTP Request Interceptor**: Set country information in the localization context.
```java
@Override
public LocaleContext resolveLocaleContext(final HttpServletRequest request) {
    parseLocaleCookieIfNecessary(request);
    LocaleContext localeContext = new TimeZoneAwareLocaleContext() {
        @Override
        public Locale getLocale() {
            return (Locale) request.getAttribute(LOCALE_REQUEST_ATTRIBUTE_NAME);
        }
        @Override
        public TimeZone getTimeZone() {
            return (TimeZone) request.getAttribute(TIME_ZONE_REQUEST_ATTRIBUTE_NAME);
        }
    };

    String country = localeContext.getLocale().getCountry().toLowerCase();
    ShopShardingHelperUtil.setCountry(country);
    return localeContext;
}
```
## IV. Conclusion
This article demonstrates the use of `ThreadLocal`, `InheritableThreadLocal`, and `TransmittableThreadLocal` to handle context transfer in complex multithreaded scenarios, such as supporting multiple countries in an overseas mall. By leveraging `TransmittableThreadLocal` with custom thread pools, we ensure correct context propagation, enabling proper handling of localization and country-specific data in a multithreaded environment.