
https://slack.engineering/how-we-design-our-apis-at-slack/
the Slack API design process:

1. **API Lifecycle Diagram**

```
+-------------+       +-----------+       +---------+       +------------+
| Specification  ---->  |  Review   ---->  | Testing  ---->  | Deployment  |
+-------------+       +-----------+       +---------+       +------------+
       |
       v
+-------------+
|  Feedback   |
+-------------+
       |
       v
   (Iterate)
```

2. **API Design Principles Diagram**

```
+-------------------+       +---------------------+       +------------------+
| Do one thing well |       | Fast & Easy Start   |       | Design for Scale |
+-------------------+       +---------------------+       +------------------+
         |                           |                           |
         v                           v                           v
+-------------------+       +---------------------+       +------------------+
| Simplify API,     |       | "Hello World" in    |       | Paginate,        |
| focus on single   |       | <15 minutes         |       | rate limit, etc. |
| function          |       +---------------------+       +------------------+
+-------------------+       
```

3. **Example API Flowchart (rtm.start to rtm.connect)**

```
+----------+       +--------------+       +----------------+       +--------------+
| rtm.start | ----> | High Payload | ----> | Scaling Issues | ----> | rtm.connect  |
+----------+       +--------------+       +----------------+       +--------------+
  |                                                                                 ^
  |                                                                                 |
  +---------------------------------------------------------------------------------+
      Simplify to only essential data (WebSocket URL)
```

4. **Error Handling and Feedback Loop Diagram**

```
+---------+       +----------------+       +-------------+
|  Error  | ----> | Short & Long   | ----> | Developer   |
| Detected|       | Form Messages  |       | Feedback    |
+---------+       +----------------+       +-------------+
      |                      |                   |
      v                      v                   v
+-----------+           +------------+     +--------------+
| Correction|           | Improvement|     | Re-iteration |
+-----------+           +------------+     +--------------+
```

5. **Rate Limiting and Pagination Diagram**

```
+-------------+       +-------------+       +---------------+
| Large API   | ----> | Pagination  | ----> | Manage Data   |
| Call        |       | Implemented |       | Flow          |
+-------------+       +-------------+       +---------------+
      |                      |                    |
      v                      v                    v
+-------------+       +-------------+       +---------------+
| Rate Limit  | ----> | System      | ----> | Maintain      |
| Set         |       | Reliability |       | Performance   |
+-------------+       +-------------+       +---------------+
```

These diagrams provide a basic visual overview using ASCII art, making complex processes easier to grasp at a glance. Each diagram focuses on a core aspect of the API design and lifecycle as practiced at Slack, adapted to a simple and digestible format.