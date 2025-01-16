#Note - got it from 
A `trace` is a group of [transactions](https://www.elastic.co/guide/en/apm/guide/current/data-model-transactions.html "Transactions") and [spans](https://www.elastic.co/guide/en/apm/guide/current/data-model-spans.html "Spans") with a common root. Each `trace` tracks the entirety of a single request. When a `trace` travels through multiple services, as is common in a microservice architecture, it is known as a distributed trace.

### Why is distributed tracing important?
Distributed tracing enables you to analyze performance throughout your microservice architecture by tracing the entirety of a request — from the initial web request on your front-end service all the way to database queries made on your back-end services.

Tracking requests as they propagate through your services provides an end-to-end picture of where your application is spending time, where errors are occurring, and where bottlenecks are forming. Distributed tracing eliminates individual service’s data silos and reveals what’s happening outside of service borders.
For supported technologies, distributed tracing works out-of-the-box, with no additional configuration required.

### How distributed tracing works
Distributed tracing works by injecting a custom `traceparent` HTTP header into outgoing requests. This header includes information, like `trace-id`, which is used to identify the current trace, and `parent-id`, which is used to identify the parent of the current span on incoming requests or the current span on an outgoing request.

When a service is working on a request, it checks for the existence of this HTTP header. If it’s missing, the service starts a new trace. If it exists, the service ensures the current action is added as a child of the existing trace, and continues to propagate the trace.

#### Trace propagation examples
In this example, Elastic’s Ruby agent communicates with Elastic’s Java agent. Both support the `traceparent` header, and trace data is successfully propagated.
![How traceparent propagation works](https://www.elastic.co/guide/en/apm/guide/current/images/dt-trace-ex1.png)

In this example, Elastic’s Ruby agent communicates with OpenTelemetry’s Java agent. Both support the `traceparent` header, and trace data is successfully propagated.
![How traceparent propagation works](https://www.elastic.co/guide/en/apm/guide/current/images/dt-trace-ex2.png)

In this example, the trace meets a piece of middleware that doesn’t propagate the `traceparent` header. The distributed trace ends and any further communication will result in a new trace.
![How traceparent propagation works](https://www.elastic.co/guide/en/apm/guide/current/images/dt-trace-ex3.png)

### Visualize distributed tracing
The APM app’s timeline visualization provides a visual deep-dive into each of your application’s traces:
![Distributed tracing in the APM UI](https://www.elastic.co/guide/en/apm/guide/current/images/apm-distributed-tracing.png)

### Manual distributed tracing
Elastic agents automatically propagate distributed tracing context for supported technologies. If your service communicates over a different, unsupported protocol, you can manually propagate distributed tracing context from a sending service to a receiving service with each agent’s API.

#### Add the `traceparent` header to outgoing requests
Sending services must add the `traceparent` header to outgoing requests.
Go iOS Java .NET Node.js PHP Python Ruby
1.  Start a transaction with [`startTransaction`](https://www.elastic.co/guide/en/apm/agent/java/current/public-api.html#api-start-transaction), or a span with [`startSpan`](https://www.elastic.co/guide/en/apm/agent/java/current/public-api.html#api-span-start-span).
2.  Inject the `traceparent` header into the request object with [`injectTraceHeaders`](https://www.elastic.co/guide/en/apm/agent/java/current/public-api.html#api-transaction-inject-trace-headers)

Example of manually instrumenting an RPC framework:
```java
// Hook into a callback provided by the RPC framework that is called on outgoing requests
public Response onOutgoingRequest(Request request) throws Exception {
  Span span = ElasticApm.currentSpan() [](https://www.elastic.co/guide/en/apm/guide/current/apm-distributed-tracing.html#CO12-4)
          .startSpan("external", "http", null)
          .setName(request.getMethod() + " " + request.getHost());
  try (final Scope scope = transaction.activate()) {
      span.injectTraceHeaders((name, value) -> request.addHeader(name, value)); [](https://www.elastic.co/guide/en/apm/guide/current/apm-distributed-tracing.html#CO12-5)
      return request.execute();
  } catch (Exception e) {
      span.captureException(e);
      throw e;
  } finally {
      span.end(); [](https://www.elastic.co/guide/en/apm/guide/current/apm-distributed-tracing.html#CO12-6)
  }
}
```

#### Parse the `traceparent` header on incoming requests
Receiving services must parse the incoming `traceparent` header, and start a new transaction or span as a child of the received context.

Go iOS Java .NET Node.js PHP Python Ruby
1.  Create a transaction as a child of the incoming transaction with [`startTransactionWithRemoteParent()`](https://www.elastic.co/guide/en/apm/agent/java/current/public-api.html#api-transaction-inject-trace-headers).
2.  Start and name the transaction with [`activate()`](https://www.elastic.co/guide/en/apm/agent/java/current/public-api.html#api-transaction-activate) and [`setName()`](https://www.elastic.co/guide/en/apm/agent/java/current/public-api.html#api-set-name).

Example:
```java
// Hook into a callback provided by the framework that is called on incoming requests
public Response onIncomingRequest(Request request) throws Exception {
    // creates a transaction representing the server-side handling of the request
    Transaction transaction = ElasticApm.startTransactionWithRemoteParent(request::getHeader, request::getHeaders); [](https://www.elastic.co/guide/en/apm/guide/current/apm-distributed-tracing.html#CO18-1)
    try (final Scope scope = transaction.activate()) { [](https://www.elastic.co/guide/en/apm/guide/current/apm-distributed-tracing.html#CO18-2)
        String name = "a useful name like ClassName#methodName where the request is handled";
        transaction.setName(name); [](https://www.elastic.co/guide/en/apm/guide/current/apm-distributed-tracing.html#CO18-3)
        transaction.setType(Transaction.TYPE_REQUEST); [](https://www.elastic.co/guide/en/apm/guide/current/apm-distributed-tracing.html#CO18-4)
        return request.handle();
    } catch (Exception e) {
        transaction.captureException(e);
        throw e;
    } finally {
        transaction.end(); [](https://www.elastic.co/guide/en/apm/guide/current/apm-distributed-tracing.html#CO18-5)
    }
}
```

Eventually, end the transaction

### Distributed tracing with RUM
Some additional setup may be required to correlate requests correctly with the Real User Monitoring (RUM) agent.

See the [RUM distributed tracing guide](https://www.elastic.co/guide/en/apm/agent/rum-js/current/distributed-tracing-guide.html) for information on enabling cross-origin requests, setting up server configuration, and working with dynamically-generated HTML.