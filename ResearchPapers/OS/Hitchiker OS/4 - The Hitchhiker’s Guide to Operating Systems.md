### 4. One Model Checker to Rule Them All

**Philosophy 3: Enumeration Demystifies Operating Systems**

The behavior of the executable model can be exhaustively explored by enumerating all possible non-deterministic choices. This section presents the MOSAIC model checker (Section 4.1), its application in operating system teaching (Section 4.2), and quantitative experiments (Section 4.3).

---

### 4.1 MOSAIC Model Checker Design and Implementation

Instead of executing a system call immediately, MOSAIC system calls return a dictionary mapping possible choices (labeled transitions) to lambda callbacks for performing the system call, even if there is only one choice.

**Example: System Call Definitions**
```python
def sys_sched(self):
    return {  # all possible choices
        f't{i+1}': (lambda i=i: self._switch_to(i))  # callback
        for i, th in enumerate(self._threads)
        if is_runnable(th.context)
    }

def sys_fork(self, *args):
    return {  # only one choice
        'fork': (lambda: self._do_fork())
    }
```

**Algorithm 1: State Space Explorer**
The algorithm performs a straightforward breadth-first search, memorizing traversed states.

```python
Q = {[]}  # queue of traces pending checking
S = set()  # set of checked states

while Q:
    tr = Q.pop()
    s, choices = replay(tr)
    if s not in S:
        S.add(s)  # add the unexplored state to S
        for c in choices:
            Q.append(tr + [c])  # extend tr with c and append to Q
```

**Interactive State Explorer Visualization**
```plaintext
         +-------------+
         |  Thread 1   |
         +-------------+
                 |
                 v
         +-------------+
         |  Thread 2   |
         +-------------+
```
The interactive explorer displays thread interleaving, highlighting program counters and expanding vertices for detailed examination.

---

### 4.2 Model Checking for Fun and Profits

The ability to exhaustively explore the state space makes model checkers suitable for explaining non-trivial cases in operating systems.

**Example: TOCTTOU Attack**
```python
def main():
    sys_bwrite('/etc/passwd', ('plain', 'secret...'))
    sys_bwrite('file', ('plain', 'data...'))
    pid = sys_fork()
    sys_sched()
    if pid == 0:  # attacker: symlink file -> /etc/passwd
        sys_bwrite('file', ('symlink', '/etc/passwd'))
    else:  # sendmail (root): write to plain file
        filetype, contents = sys_bread('file')  # for check
        if filetype == 'plain':
            sys_sched()  # TOCTTOU interval
            filetype, contents = sys_bread('file')  # for use
            match filetype:
                case 'symlink': filename = contents
                case 'plain': filename = 'file'
            sys_bwrite(filename, 'mail')
            sys_write(f'{filename} written')
        else:
            sys_write('rejected')
```
MOSAIC reveals that "etc/passwd written" is possible, demonstrating the TOCTTOU attack.

**Example: Concurrency with `tot++`**
```python
def Tsum():
    for _ in range(N):
        tmp = heap.tot  # load(tot)
        sys_sched()
        heap.tot = tmp + 1  # store(tot)
        sys_sched()

def main():
    heap.tot = 0
    for _ in range(T):
        sys_spawn(Tsum)
```
MOSAIC shows `tot` can be 2 regardless of N and T (for N, T ≥ 2), illustrating a potential thread interleaving issue.

**File System Consistency and Journaling**
```python
def main():
    # initially, file has a single block #1
    sys_bwrite('file.inode', 'i [#1]')
    sys_bwrite('used', '#1')
    sys_bwrite('#1', '#1 (old)')
    sys_sync()

    # append a block #2 to the file
    sys_bwrite('file.inode', 'i [#1 #2]')  # inode
    sys_bwrite('used', '#1 #2')  # bitmap
    sys_bwrite('#1', '#1 (new)')  # data block 1
    sys_bwrite('#2', '#2 (new)')  # data block 2
    sys_crash()  # system crash

    # display file system state at crash recovery
    inode = sys_bread('file.inode')
    used = sys_bread('used')
    sys_write(f'{inode:10}; used: {used:5} | ')
    for i in [1, 2]:
        if f'#{i}' in inode:
            b = sys_bread(f'#{i}')
            sys_write(f'{b} ')
```
MOSAIC checks potential file system inconsistencies upon crashes, revealing the necessity of `sync()` calls.

---

### 4.3 Experiments

The performance of MOSAIC is evaluated by checking six representative models. Despite state space explosion issues, the results demonstrate its utility for instructional purposes.

**Evaluation Results:**
```plaintext
Subject       | Parameters                | State | Memory  | Time
--------------|---------------------------|-------|---------|------
fork-buf      | n = 1 (p = 2)             | 15    | 17.0 MB | <0.1s
fork-buf      | n = 2 (p = 4)             | 557   | 19.8 MB | 3.3s
cond-var      | n = 1; tp = 1; tc = 1     | 33    | 17.3 MB | <0.1s
cond-var      | n = 1; tp = 1; tc = 2     | 306   | 19.7 MB | 0.1s
xv6-log       | n = 2                     | 55    | 17.3 MB | <0.1s
xv6-log       | n = 4                     | 265   | 19.2 MB | <0.1s
tocttou       | p = 2                     | 33    | 17.4 MB | <0.1s
tocttou       | p = 3                     | 97    | 17.8 MB | 0.2s
parallel-inc  | n = 1; ts = 2             | 40    | 17.2 MB | <0.1s
parallel-inc  | n = 2; ts = 2             | 164   | 18.0 MB | <0.1s
fs-crash      | n = 2                     | 90    | 17.5 MB | <0.1s
fs-crash      | n = 4                     | 332   | 19.4 MB | <0.1s
```

---

### Conclusion

The MOSAIC model checker provides a comprehensive tool for exhaustively exploring operating system behavior. It is particularly useful for teaching, allowing instructors to rigorously explain complex cases through detailed exploration and visualization.

---

This guide integrates state transition systems and exhaustive enumeration into OS education, offering a rigorous, interactive, and practical approach to understanding and teaching operating systems.