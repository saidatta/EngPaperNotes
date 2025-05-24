Below is a walkthrough of what this code does, how the functions and classes are organized, and some usage examples. This code primarily deals with:

1. **Initializing** and **migrating** the Airflow metadata database.
2. **Creating default connections** and pools.
3. **Handling** global database locks to ensure migrations run in a single process.
4. **Reflecting** tables for advanced tasks, such as verifying or dropping old schema objects.

The snippet is part of Airflow’s internal utilities for managing and upgrading the Airflow metadata database. Much of the functionality is invoked by commands like `airflow db init`, `airflow db upgrade`, and `airflow db reset`.

---

## Table of Contents

1. [Imports and Constants](#imports-and-constants)  
2. [Utility Functions and Explanation](#utility-functions-and-explanation)  
   2.1 [\_format_airflow_moved_table_name](#_format_airflow_moved_table_name)  
   2.2 [merge_conn](#merge_conn)  
   2.3 [add_default_pool_if_not_exists](#add_default_pool_if_not_exists)  
   2.4 [create_default_connections](#create_default_connections)  
   2.5 [\_get_flask_db](#_get_flask_db)  
   2.6 [\_create_db_from_orm](#_create_db_from_orm)  
   2.7 [initdb](#initdb)  
   2.8 [\_get_alembic_config](#_get_alembic_config)  
   2.9 [\_get_script_object](#_get_script_object)  
   2.10 [\_get_current_revision](#_get_current_revision)  
   2.11 [check_migrations](#check_migrations)  
   2.12 [\_configured_alembic_environment (context manager)](#_configured_alembic_environment-context-manager)  
   2.13 [check_and_run_migrations](#check_and_run_migrations)  
   2.14 [synchronize_log_template](#synchronize_log_template)  
   2.15 [reflect_tables](#reflect_tables)  
   2.16 [upgradedb](#upgradedb)  
   2.17 [resetdb](#resetdb)  
   2.18 [downgrade](#downgrade)  
   2.19 [drop_airflow_models](#drop_airflow_models)  
   2.20 [drop_airflow_moved_tables](#drop_airflow_moved_tables)  
   2.21 [check](#check)  
   2.22 [DBLocks (Enum)](#dblocks-enum)  
   2.23 [create_global_lock (context manager)](#create_global_lock-context-manager)  
   2.24 [compare_type, compare_server_default](#compare_type-compare_server_default)  
   2.25 [get_sqla_model_classes](#get_sqla_model_classes)  
   2.26 [get_query_count, get_query_count_async, check_query_exists, exists_query](#get_query_count-and-related)  
   2.27 [LazySelectSequence (class)](#lazyselectsequence-class)  

---

## 1. Imports and Constants

```python
from __future__ import annotations

import collections.abc
import contextlib
import enum
import itertools
import json
import logging
import os
import sys
import time
import warnings
from collections.abc import Generator, Iterable, Iterator, Sequence
from tempfile import gettempdir
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    TypeVar,
    overload,
)

import attrs
from sqlalchemy import (
    Table,
    exc,
    func,
    inspect,
    literal,
    or_,
    select,
    text,
)

import airflow
from airflow import settings
from airflow.configuration import conf
from airflow.exceptions import AirflowException
from airflow.models import import_all_models
from airflow.utils import helpers
from airflow.utils.db_manager import RunDBManager
from airflow.utils.session import NEW_SESSION, create_session, provide_session  # noqa: F401
from airflow.utils.task_instance_session import get_current_task_instance_session
```

- **Imports**: Bring in the standard libraries (`logging`, `os`, `time`, etc.) and various Airflow modules needed to manage database connections, the Airflow settings, and session handling.
- `attrs` is a library used to create “plain old Python object” classes with less boilerplate.
- The `sqlalchemy` imports handle the actual database schema (creating tables, reflecting them, etc.).
- `contextlib` is used here primarily for context managers.

```python
T = TypeVar("T")

log = logging.getLogger(__name__)

_REVISION_HEADS_MAP: dict[str, str] = {
    "2.7.0": "405de8318b3a",
    "2.8.0": "10b52ebd31f7",
    ...
}
```
- `T` is a generic type variable for use in typed function signatures.
- `_REVISION_HEADS_MAP` is a dictionary that maps Airflow versions to particular database revision hashes. Airflow uses Alembic for migrations, and each release may bump the revision.

---

## 2. Utility Functions and Explanation

Below are the functions broken down with an explanation of what each does.

---
### 2.1 `_format_airflow_moved_table_name(source_table, version, category)`

```python
def _format_airflow_moved_table_name(source_table, version, category):
    return "__".join([settings.AIRFLOW_MOVED_TABLE_PREFIX, version.replace(".", "_"), category, source_table])
```

- **Purpose**: When Airflow migrates or re-names tables (for example, in major database schema changes), it can keep the old table around under a new name. This function standardizes how that “moved” table is named.
- `AIRFLOW_MOVED_TABLE_PREFIX` is a constant (usually `airflow_moved_`) from Airflow settings.

**Example Usage**:
```python
moved_table_name = _format_airflow_moved_table_name("dag_run", "2.6.0", "backup")
# -> "airflow_moved__2_6_0__backup__dag_run"
```
---
### 2.2 `merge_conn(conn: Connection, session: Session = NEW_SESSION)`
```python
@provide_session
def merge_conn(conn: Connection, session: Session = NEW_SESSION):
    """Add new Connection."""
    if not session.scalar(select(1).where(conn.__class__.conn_id == conn.conn_id)):
        session.add(conn)
        session.commit()
```

- **Purpose**: Inserts a new connection row into the `connection` table only if one with the same `conn_id` doesn’t already exist.
- A `Connection` object has fields such as `conn_id`, `host`, `login`, etc.
- The check `session.scalar(select(1).where(...))` effectively checks if the connection already exists by ID.
- `@provide_session` is a decorator to handle session scope. `session.commit()` is needed after adding.

**Example Usage**:
```python
from airflow.models.connection import Connection

my_conn = Connection(conn_id="my_aws_connection", conn_type="aws")
merge_conn(my_conn)  # merges if doesn't exist, or does nothing otherwise
```
---
### 2.3 `add_default_pool_if_not_exists(session: Session = NEW_SESSION)`

```python
@provide_session
def add_default_pool_if_not_exists(session: Session = NEW_SESSION):
    """Add default pool if it does not exist."""
    from airflow.models.pool import Pool

    if not Pool.get_pool(Pool.DEFAULT_POOL_NAME, session=session):
        default_pool = Pool(
            pool=Pool.DEFAULT_POOL_NAME,
            slots=conf.getint(section="core", key="default_pool_task_slot_count"),
            description="Default pool",
            include_deferred=False,
        )
        session.add(default_pool)
        session.commit()
```

- **Purpose**: Creates Airflow’s “default” pool if it isn’t already present in the database. Pools are used to limit concurrency for groups of tasks.
- The default pool’s size (number of slots) is taken from the `core` config key: `default_pool_task_slot_count`.

**Example Usage** (often called automatically during `airflow db init`):
```python
add_default_pool_if_not_exists()
```

---

### 2.4 `create_default_connections(session: Session = NEW_SESSION)`

```python
@provide_session
def create_default_connections(session: Session = NEW_SESSION):
    ...
    merge_conn(
        Connection(
            conn_id="airflow_db",
            conn_type="mysql",
            host="mysql",
            login="root",
            password="",
            schema="airflow",
        ),
        session,
    )
    ...
```

- **Purpose**: Creates a set of commonly used connections in Airflow if they do not already exist.  
- These include connections like `"airflow_db"`, `"mysql_default"`, `"postgres_default"`, `"google_cloud_default"`, and more.  
- Great for out-of-the-box testing and examples—users can reference these default IDs in their DAGs.

**Usage**: Called as part of database initialization. Rarely called directly by end-users, but you can if needed.

---

### 2.5 `_get_flask_db(sql_database_uri)`

```python
def _get_flask_db(sql_database_uri):
    from flask import Flask
    from flask_sqlalchemy import SQLAlchemy
    from airflow.www.session import AirflowDatabaseSessionInterface

    flask_app = Flask(__name__)
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = sql_database_uri
    flask_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db = SQLAlchemy(flask_app)
    AirflowDatabaseSessionInterface(app=flask_app, db=db, table="session", key_prefix="")
    return db
```

- **Purpose**: Airflow’s webserver uses Flask; this function creates a lightweight Flask context to hold a `SQLAlchemy` object configured for the `sql_database_uri`.
- It also configures the “Flask session” table for storing webserver sessions in the DB, if desired.

**Usage**: Internally called to set up the “Flask session” table during DB initialization.

---

### 2.6 `_create_db_from_orm(session)`

```python
def _create_db_from_orm(session):
    log.info("Creating Airflow database tables from the ORM")
    from alembic import command
    from airflow.models.base import Base

    def _create_flask_session_tbl(sql_database_uri):
        db = _get_flask_db(sql_database_uri)
        db.create_all()

    with create_global_lock(session=session, lock=DBLocks.MIGRATIONS):
        engine = session.get_bind().engine
        Base.metadata.create_all(engine)
        _create_flask_session_tbl(engine.url)
        # stamp the migration head
        config = _get_alembic_config()
        command.stamp(config, "head")
        log.info("Airflow database tables created")
```

- **Purpose**: Creates all core Airflow tables based on the ORM definitions, then “stamps” the Alembic migration history as “head,” meaning it matches the current schema.  
- Also sets up the Flask session table by calling `_create_flask_session_tbl`.

**Usage**: Called the first time you initialize a brand-new Airflow DB (no prior migrations exist).

---

### 2.7 `initdb(session: Session = NEW_SESSION, load_connections: bool = True)`

```python
@provide_session
def initdb(session: Session = NEW_SESSION, load_connections: bool = True):
    """Initialize Airflow database."""
    ...
    external_db_manager = RunDBManager()
    external_db_manager.validate()

    import_all_models()

    db_exists = _get_current_revision(session)
    if db_exists:
        upgradedb(session=session)
    else:
        _create_db_from_orm(session=session)

    external_db_manager.initdb(session)
    if conf.getboolean("database", "LOAD_DEFAULT_CONNECTIONS") and load_connections:
        create_default_connections(session=session)
    add_default_pool_if_not_exists(session=session)
    synchronize_log_template(session=session)
```

- **Purpose**: High-level command to initialize the Airflow DB:
  1. Checks if it’s a fresh DB or if migrations exist.
  2. If migrations exist, runs an upgrade. Otherwise, creates the schema from scratch.
  3. Creates default connections (unless configured not to) and the default pool.
  4. Synchronizes the log template config (for logging).
- Usually corresponds to the CLI command: `airflow db init`.

**Usage**:
```bash
airflow db init
```
Or programmatically:
```python
initdb()
```

---

### 2.8 `_get_alembic_config()`

```python
def _get_alembic_config():
    from alembic.config import Config

    package_dir = os.path.dirname(airflow.__file__)
    directory = os.path.join(package_dir, "migrations")
    alembic_file = conf.get("database", "alembic_ini_file_path")
    ...
    config.set_main_option("script_location", directory.replace("%", "%%"))
    config.set_main_option("sqlalchemy.url", settings.SQL_ALCHEMY_CONN.replace("%", "%%"))
    return config
```

- **Purpose**: Constructs an Alembic `Config` object, specifying where Alembic migration scripts live (`airflow/migrations`) and the database connection URL.  
- This is used throughout other functions to apply, stamp, or generate migration scripts.

---

### 2.9 `_get_script_object(config=None)`

```python
def _get_script_object(config=None) -> ScriptDirectory:
    from alembic.script import ScriptDirectory
    ...
    return ScriptDirectory.from_config(config)
```

- **Purpose**: Given the Alembic `Config` (or by building a new one if not provided), returns the “script” object containing all of the migration scripts known to Alembic.

---

### 2.10 `_get_current_revision(session)`

```python
def _get_current_revision(session):
    from alembic.migration import MigrationContext

    conn = session.connection()
    migration_ctx = MigrationContext.configure(conn)
    return migration_ctx.get_current_revision()
```

- **Purpose**: Retrieves the current Alembic revision stamped in the metadata database.  
- `None` indicates the database is not stamped with any revision (i.e., it’s brand-new or not recognized by Alembic).

---

### 2.11 `check_migrations(timeout)`

```python
def check_migrations(timeout):
    ...
    for ticker in range(timeout):
        source_heads = set(env.script.get_heads())
        db_heads = set(context.get_current_heads())
        if source_heads == db_heads and external_db_manager.check_migration(settings.Session()):
            return
        time.sleep(1)
    raise TimeoutError(...)
```

- **Purpose**: Poll (for up to `timeout` seconds) whether the database’s Alembic revision heads match the source code’s revision heads. If they do not match after `timeout`, raises `TimeoutError`.
- Typically used in environments where multiple Airflow instances might be racing to apply migrations.

---

### 2.12 `_configured_alembic_environment() (context manager)`

```python
@contextlib.contextmanager
def _configured_alembic_environment() -> Generator[EnvironmentContext, None, None]:
    from alembic.runtime.environment import EnvironmentContext
    ...
    with (
        EnvironmentContext(
            config,
            script,
        ) as env,
        settings.engine.connect() as connection,
    ):
        ...
        yield env
```

- **Purpose**: A context manager that sets up an Alembic environment (i.e., loads migrations, configures a `connection` to the DB, etc.) so that you can run queries like `get_heads` or `get_current_heads`.
- Used in functions like `check_migrations`.

---

### 2.13 `check_and_run_migrations()`

```python
def check_and_run_migrations():
    ...
    source_heads = set(env.script.get_heads())
    db_heads = set(context.get_current_heads())
    ...
    if len(db_heads) < 1:
        db_command = initdb
        command_name = "init"
        verb = "initialize"
    elif source_heads != db_heads:
        db_command = upgradedb
        command_name = "migrate"
        verb = "migrate"
    ...
```

- **Purpose**: In a *TUI/CLI* context, checks if the DB needs to be created or migrated. If it sees mismatches in revision heads, it prompts the user interactively to confirm whether to run the migrations automatically.

---

### 2.14 `synchronize_log_template(session)`

```python
@provide_session
def synchronize_log_template(*, session: Session = NEW_SESSION) -> None:
    """
    Synchronize log template configs with table.
    ...
    """
```

- **Purpose**: Airflow stores a "log filename template" and "elasticsearch log template" in config. This function ensures the DB has the correct record of these templates so old logs remain accessible. If the current template differs from what’s stored in the table, it inserts a new row.

---

### 2.15 `reflect_tables(tables, session)`

```python
def reflect_tables(tables: list[MappedClassProtocol | str] | None, session):
    ...
    metadata = sqlalchemy.schema.MetaData()
    metadata.reflect(bind=bind, ...)
    return metadata
```

- **Purpose**: Uses SQLAlchemy’s reflection to dynamically read the structure of existing tables from the database. This is helpful to see if certain tables or columns exist, or to drop them if needed.

---

### 2.16 `upgradedb(...)`

```python
@provide_session
def upgradedb(
    *,
    to_revision: str | None = None,
    from_revision: str | None = None,
    show_sql_only: bool = False,
    reserialize_dags: bool = True,
    session: Session = NEW_SESSION,
):
    ...
    # Using Alembic to apply migrations
    from alembic import command
    import_all_models()
    config = _get_alembic_config()

    ...
    command.upgrade(config, revision=to_revision or "heads")
    ...
    synchronize_log_template(session=session)
```

- **Purpose**: Performs the main database schema upgrade to the specified revision or “head” if none is specified.  
- Has a “dry-run” mode (`show_sql_only=True`) to print out the SQL that would be run without actually applying it.  
- Re-applies some standard items like log template sync.

**Usage**:
```bash
airflow db upgrade
```
Or programmatically:
```python
upgradedb(to_revision="abc123")  # up to a specific revision
```

---

### 2.17 `resetdb(session, skip_init=False)`

```python
@provide_session
def resetdb(session: Session = NEW_SESSION, skip_init: bool = False):
    """Clear out the database."""
    ...
    with create_global_lock(session=session, lock=DBLocks.MIGRATIONS), connection.begin():
        drop_airflow_models(connection)
        drop_airflow_moved_tables(connection)
        ...
        external_db_manager.drop_tables(session, connection)

    if not skip_init:
        initdb(session=session)
```

- **Purpose**: **DESTROYS** all Airflow tables, then re-initializes them if `skip_init=False`.  
- Useful for a full reset in a dev or test environment.

**Usage**:
```bash
airflow db reset
```

> **Warning**: This is destructive; it drops your entire Airflow metadata DB.

---

### 2.18 `downgrade(...)`

```python
@provide_session
def downgrade(*, to_revision, from_revision=None, show_sql_only=False, session: Session = NEW_SESSION):
    ...
    if show_sql_only:
        _offline_migration(command.downgrade, config=config, revision=revision_range)
    else:
        command.downgrade(config, revision=to_revision, sql=show_sql_only)
```

- **Purpose**: Moves the DB schema to a *previous* Alembic revision (downgrades).  
- Typically used only for development or in special cases.  
- Supports a “SQL-only” mode to dump SQL statements instead of executing them.

---

### 2.19 `drop_airflow_models(connection)`

```python
def drop_airflow_models(connection):
    from airflow.models.base import Base
    Base.metadata.drop_all(connection)
    db = _get_flask_db(connection.engine.url)
    db.drop_all()
    ...
    version.drop(connection)
```

- **Purpose**: Drops all tables associated with the Airflow ORM base classes, as well as the “Flask session” table, and the Alembic “version” table that tracks migrations.

---

### 2.20 `drop_airflow_moved_tables(connection)`

```python
def drop_airflow_moved_tables(connection):
    from airflow.models.base import Base
    from airflow.settings import AIRFLOW_MOVED_TABLE_PREFIX

    tables = set(inspect(connection).get_table_names())
    to_delete = [Table(x, Base.metadata) for x in tables if x.startswith(AIRFLOW_MOVED_TABLE_PREFIX)]
    for tbl in to_delete:
        tbl.drop(settings.engine, checkfirst=False)
        Base.metadata.remove(tbl)
```

- **Purpose**: Finds any tables that were previously renamed/archived (those that begin with `AIRFLOW_MOVED_TABLE_PREFIX`) and drops them.

---

### 2.21 `check(session)`

```python
@provide_session
def check(session: Session = NEW_SESSION):
    """Check if the database works."""
    session.execute(text("select 1 as is_alive;"))
    log.info("Connection successful.")
```

- **Purpose**: A simple “heartbeat” query to ensure the DB connection is live.

**Usage**: 
```bash
airflow db check
```
Or programmatically:
```python
check()
```

---

### 2.22 `DBLocks (Enum)`

```python
@enum.unique
class DBLocks(enum.IntEnum):
    MIGRATIONS = enum.auto()
    SCHEDULER_CRITICAL_SECTION = enum.auto()

    def __str__(self):
        return f"airflow_{self._name_}"
```

- **Purpose**: Defines global locks (e.g. MIGRATIONS) so that they can be acquired by name or ID, preventing concurrency issues when multiple schedulers or processes perform DB operations.

---

### 2.23 `create_global_lock(session, lock, lock_timeout=1800) (context manager)`

```python
@contextlib.contextmanager
def create_global_lock(
    session: Session,
    lock: DBLocks,
    lock_timeout: int = 1800,
) -> Generator[None, None, None]:
    """Contextmanager that will create and teardown a global db lock."""
    ...
    if dialect.name == "postgresql":
        conn.execute(text("SET LOCK_TIMEOUT to :timeout"), {"timeout": lock_timeout})
        conn.execute(text("SELECT pg_advisory_lock(:id)"), {"id": lock.value})
        ...
    elif dialect.name == "mysql" ...
    try:
        yield
    finally:
        ...
        # Unlock the advisory lock
```

- **Purpose**: Creates a “named” or “advisory” lock in the database so that only one process can perform certain critical actions (like migrations) at a time.  
- Implementation details differ depending on the backend. For PostgreSQL, it uses `pg_advisory_lock`. For MySQL, it uses `GET_LOCK(...)`.  
- `yield` ensures code within the `with` statement runs while locked, and once it exits the block, the lock is released.

**Example**:
```python
with create_global_lock(session, DBLocks.MIGRATIONS):
    # do something critical that shouldn't be parallelized
    upgradedb()
```

---

### 2.24 `compare_type` and `compare_server_default`

```python
def compare_type(context, inspected_column, metadata_column, inspected_type, metadata_type):
    ...
def compare_server_default(
    context, inspected_column, metadata_column, inspected_default, metadata_default, rendered_metadata_default
):
    ...
```

- **Purpose**: These are helper functions that might be passed into Alembic migrations to handle comparisons between the “declared” type in the ORM vs. the “inspected” type in the DB.  
- For example, MySQL can store textual columns with slightly different encodings or collations; these functions can override or refine how differences are handled.

---

### 2.25 `get_sqla_model_classes()`

```python
def get_sqla_model_classes():
    ...
    from airflow.models.base import Base

    try:
        return [mapper.class_ for mapper in Base.registry.mappers]
    except AttributeError:
        return Base._decl_class_registry.values()
```

- **Purpose**: Returns a list of all classes mapped by Airflow’s SQLAlchemy models.  
- The logic covers different SQLAlchemy versions (older vs. newer).

---

### 2.26 `get_query_count, get_query_count_async, check_query_exists, exists_query`

- **Purpose**: Utility functions to do quick SELECT queries that check how many results a query returns or whether at least one row matches some conditions, without duplicating query logic.

**Short Summaries**:

```python
def get_query_count(query_stmt: Select, *, session: Session) -> int:
    """Return the total number of rows (SELECT COUNT(*)) that match query_stmt."""
```
```python
async def get_query_count_async(statement: Select, *, session: AsyncSession) -> int:
    """Async version of get_query_count."""
```
```python
def check_query_exists(query_stmt: Select, *, session: Session) -> bool:
    """Return True if at least one row matches query_stmt, False otherwise."""
```
```python
def exists_query(*where: ClauseElement, session: Session) -> bool:
    """Another variant: SELECT 1 WHERE <conditions> LIMIT 1."""
```

---

### 2.27 `LazySelectSequence (class)`

```python
@attrs.define(slots=True)
class LazySelectSequence(Sequence[T]):
    _select_asc: ClauseElement
    _select_desc: ClauseElement
    _session: Session = attrs.field(...)
    _len: int | None = attrs.field(init=False, default=None)

    @classmethod
    def from_select(cls, select: Select, *, order_by: Sequence[ClauseElement], session: Session | None = None) -> Self:
        ...

    @staticmethod
    def _rebuild_select(stmt: TextClause) -> Select:
        """Rebuild textual SQL into an ORM SELECT statement."""
        raise NotImplementedError

    @staticmethod
    def _process_row(row: Row) -> T:
        """Convert a row from a SELECT to a typed object."""
        raise NotImplementedError

    ...
```

- **Purpose**: A specialized, “lazy” Python sequence interface over an SQL query.  
- It defers actually pulling items from the database until the sequence is iterated or sliced. This is helpful in places where you need a large dataset but want to handle pagination or slicing.  
- `_select_asc` and `_select_desc` store two queries—one ascending, one descending—so that slicing negative indices can fetch in reverse effectively.
- Subclasses must implement `_rebuild_select` (to handle the textual representation of a `SELECT` statement) and `_process_row`.

**Key Points**:

- **`__getitem__`**: Overridden to allow things like `lazy_seq[10]`, `lazy_seq[:5]`, etc. This triggers an actual SQL query with `LIMIT` and `OFFSET`.
- **`__len__`**: The length is fetched via a separate `SELECT COUNT(...)`.
- **`__iter__`** and `__reversed__`**: Provide forward or backward iteration.

**Example** (hypothetical pseudo-usage):

```python
# Suppose we have a select statement:
stmt = select(MyModel).order_by(MyModel.id)

# Create a lazy sequence, using the "id" column for ordering:
lazy_seq = MyLazySequence.from_select(stmt, order_by=[MyModel.id])

# Access an item:
item = lazy_seq[5]        # triggers a SQL query with OFFSET 5 LIMIT 1
length = len(lazy_seq)    # triggers a SQL COUNT query
some_slice = lazy_seq[:10]  # triggers a SQL query with LIMIT 10
```

---

## Summary

In summary, this large snippet:

- **Initializes** the Airflow database schema.
- **Handles** Alembic-based migrations (upgrade/downgrade).
- **Sets up** default connections and the default pool.
- **Ensures** global locks to prevent multiple concurrent migrations.
- **Provides** helper utilities for reflection, queries, and custom lazy sequences.

Most end-users will primarily interact with these features via CLI commands such as:

```bash
airflow db init
airflow db upgrade
airflow db downgrade
airflow db reset
airflow db check
```

But under the hood, these functions coordinate with Alembic, manage locks, and keep everything consistent in Airflow’s metadata store.