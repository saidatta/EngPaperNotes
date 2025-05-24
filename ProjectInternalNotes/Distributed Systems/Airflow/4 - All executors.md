Below is a comprehensive explanation of what this code does, how it fits into the Airflow architecture, and example usage. We’ll walk through each key piece of functionality within this **“All executors”** module.

---
# Table of Contents

1. [Overview](#overview)  
2. [Key Concepts](#key-concepts)  
3. [Important Classes and Data Structures](#important-classes-and-data-structures)  
   3.1 [\_alias_to_executors, \_module_to_executors, \_team_id_to_executors, \_classname_to_executors](#alias_and_related_dicts)  
   3.2 [\_executor_names (list\[ExecutorName\])](#executor_names_list)  
   3.3 [ExecutorLoader](#executorloader-class)  
4. [Workflow within ExecutorLoader](#workflow-within-executorloader)  
   4.1 [\_get_executor_names](#_get_executor_names)  
   4.2 [_get_team_executor_configs](#_get_team_executor_configs)  
   4.3 [block_use_of_multi_team](#block_use_of_multi_team)  
   4.4 [get_default_executor_name, get_default_executor](#get_default_executor_name-and-get_default_executor)  
   4.5 [init_executors](#init_executors)  
   4.6 [lookup_executor_name_by_str](#lookup_executor_name_by_str)  
   4.7 [load_executor](#load_executor)  
   4.8 [import_executor_cls](#import_executor_cls)  
   4.9 [import_default_executor_cls](#import_default_executor_cls)  
5. [Usage Examples](#usage-examples)  
   5.1 [Specifying a Core Executor via airflow.cfg](#specifying-core-executor)  
   5.2 [Specifying a Custom Executor (Module Path)](#custom-executor-module-path)  
   5.3 [Using Executor Aliases](#using-executor-aliases)  
6. [Visual Diagram](#visual-diagram)  

---

## 1. Overview

Airflow supports multiple **executors** to determine *where* and *how* Airflow tasks run. You might have:

- **LocalExecutor** (spawns processes locally)  
- **CeleryExecutor** (distributes tasks to Celery workers)  
- **KubernetesExecutor** (launches tasks as Kubernetes pods)  
- …plus custom/external executors.

This module:
- Maintains a **registry** of executors (both built-in and user-defined).  
- Loads executors from configuration in `airflow.cfg` (the `[core] executor = ...` setting, or additional multi-team setups).  
- Ensures you can specify executors by *alias*, by *module path*, or by *class name*.

---

## 2. Key Concepts

1. **ExecutorName**: A small data class representing how to load a particular executor.  
2. **Core Executors**: Built into Airflow: `LocalExecutor`, `SequentialExecutor`, `CeleryExecutor`, etc.  
3. **Multiple Executors**: Airflow 2 typically allows only one executor. However, upcoming or dev features may load multiple executors (one per “team,” etc.).  

---

## 3. Important Classes and Data Structures

### 3.1 `_alias_to_executors, _module_to_executors, _team_id_to_executors, _classname_to_executors`

```python
_alias_to_executors: dict[str, ExecutorName] = {}
_module_to_executors: dict[str, ExecutorName] = {}
_team_id_to_executors: dict[str | None, ExecutorName] = {}
_classname_to_executors: dict[str, ExecutorName] = {}
```

- **Purpose**: Internal dictionaries to quickly map different keys to an `ExecutorName` instance.
  - `_alias_to_executors`: Maps a short alias (e.g., `"LocalExecutor"`) to its `ExecutorName` record.  
  - `_module_to_executors`: Maps a Python module path (e.g. `"airflow.executors.local_executor.LocalExecutor"`) to `ExecutorName`.  
  - `_team_id_to_executors`: Maps a “team_id” to a particular executor (for multi-team scenarios).  
  - `_classname_to_executors`: Maps a raw class name (e.g., `"LocalExecutor"`) to `ExecutorName`.

### 3.2 `_executor_names (list[ExecutorName])`

```python
_executor_names: list[ExecutorName] = []
```

- **Purpose**: Stores all executor configurations **after** they’re first parsed from `airflow.cfg`. The module caches them so it doesn’t redo parsing repeatedly.

### 3.3 `ExecutorLoader` Class

```python
class ExecutorLoader:
    """Keeps constants for all the currently available executors."""

    executors = {
        LOCAL_EXECUTOR: "airflow.executors.local_executor.LocalExecutor",
        SEQUENTIAL_EXECUTOR: "airflow.executors.sequential_executor.SequentialExecutor",
        CELERY_EXECUTOR: "airflow.providers.celery.executors.celery_executor.CeleryExecutor",
        KUBERNETES_EXECUTOR: "airflow.providers.cncf.kubernetes.executors.kubernetes_executor.KubernetesExecutor",
        DEBUG_EXECUTOR: "airflow.executors.debug_executor.DebugExecutor",
    }
    ...
```

- Holds built-in executors in a dict mapping short aliases (e.g. `LOCAL_EXECUTOR = "LocalExecutor"`) to the full module path (e.g. `"airflow.executors.local_executor.LocalExecutor"`).  
- Contains static methods like `get_default_executor`, `init_executors`, etc., used by Airflow’s code base to load the configured executor.

---

## 4. Workflow within `ExecutorLoader`

The main logic for reading and parsing Airflow configuration is in `_get_executor_names()`, which calls `_get_team_executor_configs()`. Let’s break down each method.

### 4.1 `_get_executor_names`

```python
@classmethod
def _get_executor_names(cls) -> list[ExecutorName]:
    from airflow.configuration import conf

    if _executor_names:
        return _executor_names  # If we've already computed, just return it

    all_executor_names: list[tuple[None | str, list[str]]] = [
        (None, conf.get_mandatory_list_value("core", "EXECUTOR"))
    ]
    all_executor_names.extend(cls._get_team_executor_configs())
    ...
```

1. Checks if `_executor_names` is already populated (caching).  
2. Reads `[core] EXECUTOR` from `airflow.cfg` as a **list**. (Airflow can store multiple executor definitions in a single line if you specify them as a list.)  
3. Calls `_get_team_executor_configs()` to see if there are multiple “teams” configured.

Then it builds `ExecutorName` objects from each string. Example strings from config could be:

- `"LocalExecutor"` → recognized as a core alias.  
- `"myalias:my.custom.Executor"` → alias + custom Python module path.  
- `"airflow.executors.debug_executor.DebugExecutor"` → direct module path (no alias).  

### 4.2 `_get_team_executor_configs`

```python
@classmethod
def _get_team_executor_configs(cls) -> list[tuple[str, list[str]]]:
    from airflow.configuration import conf

    team_config = conf.get("core", "multi_team_config_files", fallback=None)
    configs = []
    if team_config:
        cls.block_use_of_multi_team()
        for team in team_config.split(","):
            (_, team_id) = team.split(":")
            configs.append((team_id, conf.get_mandatory_list_value("core", "executor", team_id=team_id)))
    return configs
```

- For multi-team scenarios, users might define multiple `[core]`-like sections with different `executor` values. This code reads them, returning a list of `(team_id, [executor_string, ...])` pairs.  
- Currently, “multi_team_config_files” is not fully supported; if set, `block_use_of_multi_team()` raises an exception unless special dev mode is enabled.

### 4.3 `block_use_of_multi_team`

```python
@classmethod
def block_use_of_multi_team(cls):
    team_dev_mode: str | None = os.environ.get("AIRFLOW__DEV__MULTI_TEAM_MODE")
    if not team_dev_mode or team_dev_mode != "enabled":
        raise AirflowConfigException(
            "Configuring multiple team based executors is not yet supported!"
        )
```

- **Prevents** usage of multiple executors until the feature is fully mature.

### 4.4 `get_default_executor_name` and `get_default_executor`

```python
@classmethod
def get_default_executor_name(cls) -> ExecutorName:
    return cls._get_executor_names()[0]

@classmethod
def get_default_executor(cls) -> BaseExecutor:
    default_executor = cls.load_executor(cls.get_default_executor_name())
    return default_executor
```

- **`get_default_executor_name()`**: Returns the first configured executor from `_get_executor_names()`.  
- **`get_default_executor()`**: Actually instantiates that executor class.

### 4.5 `init_executors`

```python
@classmethod
def init_executors(cls) -> list[BaseExecutor]:
    executor_names = cls._get_executor_names()
    loaded_executors = []
    for executor_name in executor_names:
        loaded_executor = cls.load_executor(executor_name)
        ...
        loaded_executors.append(loaded_executor)

    return loaded_executors
```

- Loads **all** configured executors (in case there are multiple) and returns a list of executor instances.  
- Typically, Airflow only uses a single executor from `[core]` — but if multiple are declared, they get loaded here.

### 4.6 `lookup_executor_name_by_str`

```python
@classmethod
def lookup_executor_name_by_str(cls, executor_name_str: str) -> ExecutorName:
    if not _classname_to_executors or not _module_to_executors or not _alias_to_executors:
        cls._get_executor_names()

    if executor_name := _alias_to_executors.get(executor_name_str):
        return executor_name
    elif executor_name := _module_to_executors.get(executor_name_str):
        return executor_name
    elif executor_name := _classname_to_executors.get(executor_name_str):
        return executor_name
    else:
        raise UnknownExecutorException(f"Unknown executor being loaded: {executor_name_str}")
```

- Given a string like `"LocalExecutor"`, `"airflow.executors.local_executor.LocalExecutor"`, or `"LocalExecutorClassName"`, returns the corresponding `ExecutorName` object.  
- If not found in any dictionary, raises `UnknownExecutorException`.

### 4.7 `load_executor`

```python
@classmethod
def load_executor(cls, executor_name: ExecutorName | str | None) -> BaseExecutor:
    if not executor_name:
        _executor_name = cls.get_default_executor_name()
    elif isinstance(executor_name, str):
        _executor_name = cls.lookup_executor_name_by_str(executor_name)
    else:
        _executor_name = executor_name

    try:
        executor_cls, import_source = cls.import_executor_cls(_executor_name)
        log.debug("Loading executor %s from %s", _executor_name, import_source.value)
        if _executor_name.team_id:
            executor = executor_cls(team_id=_executor_name.team_id)
        else:
            executor = executor_cls()
    except ImportError as e:
        ...
    log.info("Loaded executor: %s", _executor_name)

    executor.name = _executor_name
    return executor
```

1. If `executor_name` is a string, tries to convert it to an `ExecutorName` object. If `None`, uses default.  
2. Calls `import_executor_cls` to retrieve the actual Python class type.  
3. Instantiates it, sets `executor.name` to the `ExecutorName` used, and returns it.  

### 4.8 `import_executor_cls`

```python
@classmethod
def import_executor_cls(cls, executor_name: ExecutorName) -> tuple[type[BaseExecutor], ConnectorSource]:
    return import_string(executor_name.module_path), executor_name.connector_source
```

- Uses `import_string(...)` to dynamically import the executor’s class from its module path.  
- Returns the Python class and a note about the source (core executor vs. custom module).

### 4.9 `import_default_executor_cls`

```python
@classmethod
def import_default_executor_cls(cls) -> tuple[type[BaseExecutor], ConnectorSource]:
    executor_name = cls.get_default_executor_name()
    executor, source = cls.import_executor_cls(executor_name)
    return executor, source
```

- Shortcut to import the **default** executor class from config.

---

## 5. Usage Examples

### 5.1 Specifying a Core Executor via `airflow.cfg`

Suppose you want to use the LocalExecutor. In your `airflow.cfg`:
```ini
[core]
executor = LocalExecutor
```
When Airflow starts:
- `ExecutorLoader` sees `LocalExecutor` and recognizes it as a built-in alias.  
- Looks up the module path `airflow.executors.local_executor.LocalExecutor` and imports it.

### 5.2 Specifying a Custom Executor (Module Path)

If you have a custom executor:
```python
# mycompany/executors/made_up_executor.py
from airflow.executors.base_executor import BaseExecutor

class MadeUpExecutor(BaseExecutor):
    ...
```

In your `airflow.cfg`:
```ini
[core]
executor = mycompany.executors.made_up_executor.MadeUpExecutor
```
- `ExecutorLoader` sees it’s not a known alias, so it interprets the string as a module path, imports it, and instantiates it.

### 5.3 Using Executor Aliases

You can specify an alias+module path via a string like:
```
my_alias:mycompany.executors.made_up_executor.MadeUpExecutor
```
in `[core] executor = ...`, meaning:  
- **Alias**: `"my_alias"`  
- **Module path**: `"mycompany.executors.made_up_executor.MadeUpExecutor"`

**But** keep in mind, if you try to alias a *core* name like `"LocalExecutor"` with a different module, that might raise an error to avoid confusion.

---

## 6. Visual Diagram

Below is a schematic of how **ExecutorLoader** processes the configuration:

```mermaid
flowchart TB
    A[airflow.cfg] --> B{_get_executor_names()}
    B --> C((Parse [core].EXECUTOR)) 
    B --> D((Parse multi_team_config_files?))
    C --> E((Build ExecutorName objects))
    D --> E
    E --> F[_executor_names populated]
    F --> G[lookup_executor_name_by_str or load_executor used by rest of Airflow]

    G --> H{import_executor_cls}
    H --> I[import_string(<module_path>)]
    I --> J[Executor class loaded]
```

1. Airflow reads `[core].EXECUTOR` (and possibly multi-team config).
2. Builds internal structures mapping alias → module path → `ExecutorName`.
3. Future requests call `lookup_executor_name_by_str(...)` or `load_executor(...)` to load the proper executor.

---

# Final Takeaways

- `ExecutorLoader` is the unified place where Airflow looks up any executor.  
- It supports built-in executors by simple aliases (`LocalExecutor`, `CeleryExecutor`, etc.) or **custom** executors by a Python module path.  
- The logic for multi-team or multi-executor setups is included but partially gated by a dev mode check, as that feature is not fully released.  
- Once loaded, the chosen executor is instantiated, associated with the relevant team ID (if any), and used by the Airflow scheduler to run tasks.