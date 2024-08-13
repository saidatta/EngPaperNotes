### Data Characteristics
- **Read: Write Ratio:** 5:1, indicating a write-heavy system where client updates to scores are frequent.
### Database Schema Design
#### Relational Schema Overview
**Figure 1: Leaderboard Relational Database Schema**
- **Entities**:
  - **Players**: Stores player details.
  - **Games**: Information about games.
  - **Leaderboards**: Contains scores and references to players and games.
  - **Friends**: Associative table for player relationships.
- **Relationships**:
  - **Players to Games**: One-to-many (a player can participate in multiple games).
  - **Games to Leaderboards**: One-to-many (multiple leaderboards per game).
  - **Players to Leaderboards**: One-to-many (players can appear on multiple leaderboards).
  - **Players to Friends**: Many-to-many (via associative entity).

#### Sample SQL Schemas and Data

```sql
-- Games Table
CREATE TABLE Games (
    game_id INT PRIMARY KEY,
    name VARCHAR(255),
    created_at TIMESTAMP
);

-- Players Table
CREATE TABLE Players (
    player_id INT PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255),
    created_at TIMESTAMP,
    last_login TIMESTAMP,
    profile_image TEXT,
    location VARCHAR(255),
    game_id INT,
    FOREIGN KEY (game_id) REFERENCES Games(game_id)
);

-- Leaderboards Table
CREATE TABLE Leaderboards (
    leaderboard_id INT,
    score INT,
    created_at TIMESTAMP,
    game_id INT,
    player_id INT,
    PRIMARY KEY (leaderboard_id, player_id),
    FOREIGN KEY (game_id) REFERENCES Games(game_id),
    FOREIGN KEY (player_id) REFERENCES Players(player_id)
);

-- Friends Table
CREATE TABLE Friends (
    player_id1 INT,
    player_id2 INT,
    created_at TIMESTAMP,
    PRIMARY KEY (player_id1, player_id2),
    FOREIGN KEY (player_id1) REFERENCES Players(player_id),
    FOREIGN KEY (player_id2) REFERENCES Players(player_id)
);
```
#### Redis Schema Overview
**Figure 2: Leaderboard Redis Schema**
- **Entities**:
  - **Leaderboards**: Uses sorted sets to rank players by score.
  - **Players**: Hashes store player metadata.

- **Key-Value Structure**:
  - **Sorted Sets for Leaderboards**: `ZADD leaderboard_id score player_id`
  - **Hashes for Player Metadata**: `HSET player_id field value`

### SQL Queries for Common Operations
- **Insert a new player score**:
  ```sql
  INSERT INTO leaderboards (leaderboard_id, score, created_at, game_id, player_id)
  VALUES ('apex_legends', 1, '2050-08-22', 1, 42);
  ```
- **Update player score**:
  ```sql
  UPDATE leaderboards
  SET score = score + 1
  WHERE player_id = 42;
  ```

- **Fetch total score of a player for the current month**:
  ```sql
  SELECT SUM(score)
  FROM leaderboards
  WHERE player_id = 42 AND created_at >= '2025-03-01';
  ```

- **Calculate scores and rank the top 10 players**:
  ```sql
  SELECT player_id, SUM(score) AS total
  FROM leaderboards
  GROUP BY player_id
  ORDER BY total DESC
  LIMIT 10;
  ```

- **Calculate the rank of a specific player**:
  ```sql
  SELECT player_id, RANK() OVER (ORDER BY score DESC)
  FROM leaderboards
  WHERE game_id = 1;
  ```
### Type of Data Store Considerations
- **Relational Databases**:
  - Good for smaller scale, complex queries.
  - Performance issues with scaling due to the need for joins and complex transactions.

- **NoSQL/In-Memory (Redis)**:
  - Scalable, performant for real-time leaderboards.
  - Uses sorted sets for efficient score ranking.
### Capacity Planning and Traffic Estimates
- **Daily Active Users (DAU)**: 50 million (write)
- **Query Per Second (QPS)**:
  - Write: 600
  - Read: 3000
  - Peak Read: 3600

### Storage and Memory Requirements

- **Total Storage**: Approximately 2.920 TB over 5 years.
- **Memory**: Estimated at 2.2 GB for all player records.

## High-Level System Design for Leaderboards

### Leaderboard Update and Display Workflows

#### Update Workflow
- **WebSockets** for real-time communication.
- **Load Balancer** routes requests based on proximity.
- **Cache-Aside Pattern** used to update score in both cache and database.

#### Display Workflow


- **Cache Hit**: Leaderboard data served directly.
- **Cache Miss**: Data fetched from relational database, then cached.

### Challenges and Solutions

- **Scalability**: Use of in-memory databases like Redis.
- **High Availability**: Designing for redundancy and quick recovery.
- **Real-Time Updates**: Utilizing WebSockets for immediate client updates.

### Further Learning Resources

- **Sign up for a system design newsletter** to receive a comprehensive system design template.

-----
Sure, here are ASCII visualizations for the high-level use cases of updating and displaying the leaderboard data. These diagrams will help in understanding the flow and interaction between components in the system.

### 1. Score Update Workflow
This workflow illustrates the process when a player updates their score:

```
+-------------+     +-----------------+     +------------------+     +-------------+
|             |     |                 |     |                  |     |             |
|   Client    +-----> Load Balancer   +----->  Application     +----->  Database   |
|             |     |                 |     |  Server          |     |             |
+-------------+     +-----------------+     +---------+--------+     +-------------+
                                                     |
                                                     v
                                              +------+-+
                                              | Cache  |
                                              +--------+
```

- **Client**: Initiates the score update.
- **Load Balancer**: Directs the request to the appropriate application server based on factors like server load and geographic proximity.
- **Application Server**: Processes the update and interacts with both the database and cache.
- **Database**: Stores the updated score persistently.
- **Cache**: Temporary storage updated to reflect new scores for quick access.

### 2. Display Leaderboard Workflow
This workflow illustrates the process when a player requests to view the leaderboard:

```
+-------------+     +-----------------+     +------------------+     +--------+
|             |     |                 |     |                  |     |        |
|   Client    +-----> Load Balancer   +----->  Application     +-----> Cache  |
|             |     |                 |     |  Server          |     |        |
+-------------+     +-----------------+     +---------+--------+     +----+---+
                                                     |                   ^
                                                     v                   |
                                              +------+--------+          |
                                              |  Database     +----------+
                                              +---------------+
```

- **Client**: Sends a request to view the leaderboard.
- **Load Balancer**: Routes the request to the best-suited application server.
- **Application Server**: Checks the cache for the requested data.
- **Cache**: First point of data retrieval. If a cache miss occurs, the request is forwarded to the database.
- **Database**: Serves as the source of truth. Data fetched from the database is then cached to improve subsequent access times.

These visualizations provide a simplified yet descriptive outline of how leaderboard updates and display operations are handled within a scalable gaming system. This setup uses caching strategically to reduce database load and improve the responsiveness of the system.