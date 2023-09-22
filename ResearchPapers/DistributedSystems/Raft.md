This is a protocol designed for the convenience of teaching and engineering implementation. It disassembles the protocol into several relatively independent modules-leader election, log replication, and security guarantee.

### Raft Timer
Raft primarily has two event loops: one for initiating elections (Follower, Candidate) when timeouts occur, and another for periodic heartbeats (Leader) with occasional log synchronization. The most straightforward approach is the loop+sleep mechanism that we mindlessly use for undergraduate projects. It involves an outer "while true" loop and an inner sleep with a slightly smaller time interval (but still at least an order of magnitude smaller than electionTimeout and heartbeatInterval) to periodically check for the arrival of time events (needElection, heartbeat).

#### Possible Issue:
with an error margin of at least the detection time interval, t. I started to worry about the potential errors caused by the inaccuracy when dealing with complex state changes involving multiple threads. Although using Go's timer seemed complicated, thinking about it always left me unclear, and it delayed me for a while.

#### Suggestion
Use the loop+sleep mechanism for periodic timeout detection. At that moment, everything became clear to me, and I realized the accuracy concern I had mentioned in the parentheses. As long as the detection interval is smaller than the timeout interval by one or two orders of magnitude, there shouldn't be many issues. This implementation has the advantages of simplicity, straightforwardness, directness, and control.

While implementing it, I suddenly found it interesting and created two slightly different versions, with the code provided below (simplified for clarity). One version has only one loop:

```go
go func() {
  for {
    now := time.Now()
    if now.Sub(last) < electionTimeout {
      time.Sleep(checkGapInMs)
    } else {
      // I initially had doubts about the order of these two statements, but then I realized
      // as long as startElection doesn't involve any blocking IO or operations, the order doesn't matter
      startElection()
      last = time.Now()
    }
  }
}()
```

The other version has two nested loops, where the inner loop is specifically used for waiting (perhaps inspired by busy-waiting in CPUs):

```go
for {
  // ping all the peers
  for s := 0; s < len(rf.peers); s++ {
     // append entry rpc
  }(s, args)
  
  // wait until time comes
  for {
    now := time.Now()
    if now.Sub(lastPingTime) < pingGapInMs {
      continue
    } else {
      lastPingTime = time.Now()
      break
    }
  }
}
```

Although the former is more concise, the latter has clearer logic. Sometimes concise code involves reusing different logics, which can lead to slightly unclear semantics. Considering that the primary purpose of code is for humans to read (what? Did you say it's for machines to read? Ahem, I believe this is a prerequisite for code to be code, otherwise the compiler won't let it pass), I think the latter is better.


### Raft Locking
The course also provided helpful hints, but as someone who has been "playing with locks" since the elevator project in my freshman year, I skipped right past them. Without any hesitation, I confidently dove into coding, bare-chested and head held high. However, memories can deceive, and lessons can be painfully learned. It turned out that I was facing deadlocks all along, but I had no idea where the problem was. Eventually, I had to obediently read the hints several times and realized how well they were written...

In summary, the key point is to add locks to all places where global variables are read or written, and then remove locks from places that have blocking operations (such as RPC calls).

Now, let me talk about the two pitfalls I encountered while jumping into the code:

1. Holding the lock outside of a function call and then reacquiring it inside the function:
   The reason for making this mistake was that, looking at the lengthy code, based on my experience, I thought I should wrap it with a lock. When calling the function later, I forgot to place it inside the critical section. In other words, I was trying to have the best of both worlds. Deadlock achievement unlocked! Without further ado, here's the code:

   a.
   ```go
   func (rf *Raft) startElection() {
     rf.mu.Lock()
     // balalala
   }
   ```

   b.
   ```go
   rf.mu.Lock()
   // balala
   rf.startElection()
   ```

2. Forgetting to release the lock when using break/return:
   The sneaky behavior of prematurely terminating branches with break, continue, or return is something we often do happily, but it doesn't align with our innate sense of symmetry, which sometimes leads to forgetting to handle it. What do I mean by asymmetry? Let's take a look at the code:

   ```go
   if !check(arg) {
     return
   }

   for condition {
     if need {
       break
     }

     // because here are so many balalala
     // then break can be used to reduce indent
   }
   ```

   vs.

   ```go
   if !check(arg) {
     return
   } else {
     for condition {
       if !need {
         // because here are so many balalala
         // then break can be used to reduce indent
       }
     }
   }
   ```

   In the latter case, with a quick glance at the aligned if-else statement, it's clear how many exits the function has. However, for the former case, if the branch statement is buried in a massive amount of code, it's easy to forget that there's an exit hiding in a corner. Naturally, the lock won't be released.

In my specific case, when the candidate was seeking votes, if it received a majority of votes, it would directly transition to being the leader. If there were still votes coming later, I would check if the current role was already the leader. If it was, I would simply return without caring about those votes. It was quite wicked... And as you might have guessed, I forgot to release the lock before returning. Here's the code, with some omissions:

```go
go func(server int, args RequestVoteArgs) {
  // use args to request vote
  reply := RequestVoteReply{}
  ok := rf.sendRequestVote(server, &args, &reply)

  isBecomeLeader := false
  rf.mu.Lock()
  if rf.state != Candidate || rf.currentTerm != args.Term {
    rf.mu.Unlock() // <--- This wicked place
    return
  }
 // check the votes 
 if reply.VotedGranted {
	 votes++ 
	 DPrintf( "%d get vote from %d, now votes are %d, total members are:%d" ,rf.me, server, votes, peersCount) 
	 
	 if votes > peersCount / 2 { 
		 if rf.state == Candidate {
			 isBecomeLeader = true         
			 DPrintf( "%d become leader" , rf.me)      
		}
		rf.state = Leader
	} else if reply.Term > rf.currentTerm {    
		rf.currentTerm = reply .Term     
		rf.state = Follower rf.votedFor = -1    
	}  
  
  rf.mu.Unlock()
  
  if isBecomeLeader {   
    rf.startHeartbeat()   
  }
}(s, args)
```

----
The author is discussing two different ways to implement a periodic event loop in the Raft consensus algorithm. Raft is a consensus algorithm that is designed to be easy to understand. It's equivalent to Paxos in fault-tolerance and performance, but its structure is different.

The event loops are used for two main purposes in Raft:

1. Initiating elections: If a follower or candidate node doesn't hear from a leader for a certain amount of time (the election timeout), it starts an election to choose a new leader.

2. Sending heartbeats: If a node is the leader, it periodically sends heartbeats to all the other nodes to let them know it's still alive.

The author is considering two ways to implement these event loops:

1. A single loop that checks the time on each iteration. If enough time has passed since the last event (election or heartbeat), it triggers the event and updates the time. Otherwise, it sleeps for a short time before checking again.

2. A loop with a nested loop inside it. The outer loop triggers the event, and the inner loop waits until enough time has passed since the last event. It does this by continuously checking the time in a loop (busy-waiting) until the event interval has passed.

The author notes that the first approach is more concise, but the second approach has clearer logic because it separates the event-triggering and waiting into separate loops. They suggest that the second approach might be easier for people to understand, even though it's a bit more verbose.

Here's a bit more detail about the code snippets:

The first snippet is a goroutine (a lightweight thread in Go) that runs an infinite loop. On each iteration, it checks if enough time (`electionTimeout`) has passed since the last event (`last`). If not, it sleeps for a short time (`checkGapInMs`) before checking again. If enough time has passed, it triggers an election by calling `startElection()` and updates `last` to the current time.

The second snippet is similar, but it separates the waiting into a nested loop. The outer loop triggers the event (in this case, pinging all the peers), and the inner loop waits until enough time (`pingGapInMs`) has passed since the last event (`lastPingTime`). It does this by continuously checking the time in a loop (busy-waiting) until the event interval has passed. Once enough time has passed, it updates `lastPingTime` to the current time and breaks out of the inner loop, causing the outer loop to trigger the event again.


### Log Synchronization (Log Replication)

After the Leader goes online, the log synchronization process is roughly like this:

1.  After the leader is selected, it is initialized, mainly for the initialization of the nextIndex array and the matchIndex array. There may be many ways of thinking, but this is what I do: all, `nextIndex = len(rf.log)`all`matchIndex = 0`
2.  Then start the heartbeat immediately, that is, AppendEntries. The paper says that it is enough to start sending a heartbeat without content, but in my implementation, I still send a heartbeat with a logEntry. It depends on my initialization method above and the parameter construction strategy below.
3.  Synchronization process, I divide it into **the trial and** **the transmission stage** after the match ; that is, firstly `prevLogIndex+prevLogTerm`match `logEntry`, and then the Leader sends the `logEntry`matched entries to the Follower at one time .
4.  For each parameter construction, In the context of the Raft consensus algorithm, the process of log replication involves sending chunks of the log (entries) from the leader to the followers. This process is controlled by a few key parameters: `prevLogIndex`, `prevLogTerm`, and `entries`.
	- `prevLogIndex` and `prevLogTerm` are used to ensure consistency between the leader's log and the follower's log. Before sending new entries, the leader tells the follower what it believes the follower's log looks like. Specifically, `prevLogIndex` is the index of the log entry immediately preceding the new ones, and `prevLogTerm` is the term of that entry.
	- `entries` are the new log entries to be appended to the log.

The process of log replication is divided into two phases: the probe phase and the transmission phase.

	- In the probe phase, the leader is trying to find the point at which the follower's log matches its own. To do this, it sets `prevLogIndex` to `nextIndex - 1`. This is done to minimize the amount of data sent over the network: the leader is essentially asking the follower, "Do you have an entry at this index?" without actually sending the entry itself.

	- In the transmission phase, the leader has found a point of agreement and is now sending new entries to the follower. It sets `prevLogIndex` to `matchIndex`, which is the index of the last entry that was known to be replicated on the follower. The `entries` sent are those from the leader's log in the range `[prevIndex+1, min(nextIndex, len(rf.log)-1)]`.
1.  In this way, no data is transmitted during each trial (that is, [], because `prevLogIndex`= `nextIndex`-1 at this time), and all logEntries on the Leader after matching can be transmitted at one time during the transmission phase. If there is no new log, several variables will remain in: `nextIndex = len(rf.log)；prevLogIndex = matchIndex = len(rf.log)-1`, `entries`as [];

Other points are some details attached to this process:

1.  Regularly detect most of the logEntry match Index to decide whether to commit, that is, move the Leader forward `commitIndex`, and synchronize it to each Follower in the `AppendEntries`subsequent RPC.
2.  At the same time, another thread needs to check `lastApplied`whether it has caught up `commitIndex`, and ensure that the submitted log is applied to the state machine in time. I think the separation of these two variables is mainly for logical decoupling—on the Leader, the Leader actively detects and `commitIndex`updates , and on the Follower, it passively accepts the Leader's message to update.
3.  After the Leader accepts the new cmd, it needs to update its corresponding position in `matchIndex`the array , because it is also the last vote for calculating the majority.
4.  After the Follower fails to connect, the Leader must promptly terminate the execution of the content behind the callback.

### [](https://www.qtmuniao.com/2018/08/29/raft-log-replication/#AppendEntries-%E5%BF%83%E8%B7%B3-amp-%E5%90%8C%E6%AD%A5 "AppendEntries Heartbeat & Synchronization")AppendEntries Heartbeat & Sync

It mainly corresponds to the two stages mentioned above, the trial stage and the transmission stage; other things to pay attention to are the locking and status self-check mentioned in the previous article.

1   
2   
3   
4   
5   
6   
7   
8   
9   
10   
11   
12   
13   
14   
15   
16   
17   
18  

// need handle the reply   
if reply.Term > rf.currentTerm {   
    rf.becomeFollower(reply.Term)   
} else { if !reply.Success { // roll back per term every time         nextIndex := rf.nextIndex[server] - 1 for nextIndex > 1 && rf.log[nextIndex].Term == args.PrevLogTerm {            nextIndex--         }         rf.nextIndex[server] = nextIndex  
      
          
  
          
  
  
  
  
    } else { // if match, sync all after         rf.matchIndex[server] = args.PrevLogIndex + len (args.Entries)rf.nextIndex[server] = len (rf.log)} }  
          
  
  
  
  

Then there is the parameter construction. `prevIndex` For , it is also a different construction in two stages. Then take the appropriate window `entries`.

1   
2   
3   
4   
5   
6   
7   
8   
9   
10   
11   
12   
13   
14   
15   
16   
17   
18   
19   
20  

func  (rf *Raft) constructAppendEntriesArg(idx int ) *AppendEntriesArgs {  
  prevLogIndex := 0 if rf.matchIndex[idx] == 0 {    prevLogIndex = rf.nextIndex[idx] - 1 // try to find a match   } else {    prevLogIndex = rf.matchIndex[idx] // after match to sync   }  prevLogTerm := rf.log[prevLogIndex].Term  
    
	  
  
  
  
  
  
  // the need to replica window [prevLogIndex+1, nextIndex) var entries []*LogEntrystart := prevLogIndex + 1   end := min(rf.nextIndex[idx], len (rf.log) -1 ) for i : = start; i <= end; i++ {entries = append (entries, rf.log[i])  }  
    
  
  
    
  
  
  
  return &AppendEntriesArgs{rf.currentTerm, rf.me, prevLogIndex,   
    prevLogTerm, entries, rf.commitIndex}   
}  

Finally, the AppendLogEntries callback, seeing the request of the previous term, directly rejects it. There is nothing to say, just take your own term back. In addition, regardless of args.Term > or = rf.currentTerm, the peer receiving AppendEntries must become a Follower and reset the timer. In addition, when matching logEntry, pay attention to truncation.

1   
2   
3 4   
5   
6   
7   
8   
9   
10 11 12 13 14 15 16 17 18   
19 20 21 22 23 24 25 26 27 28 29 30 31 32 33  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

func  (rf *Raft) AppendEntries(args *AppendEntriesArgs, reply *AppendEntriesReply) {  
  rf.mu.Lock() defer rf.mu.Unlock()  
    
  
  reply.Term = rf.currentTerm // reject the append entries if args.Term < rf.currentTerm {reply.Success = false return   }  
    
    
  
      
  
  
  rf.leaderId = args.LeaderId   
  rf.becomeFollower(args.Term)  
  
  if args.PrevLogIndex >= len (rf.log) {   
    reply.Success = false  
   } else  if rf.log[args.PrevLogIndex].Term != args.PrevLogTerm {   
    reply.Success = false  
     rf.log = rf.log[ :args.PrevLogIndex]   
  } else {   
    reply.Success = true // delete not match and append new ones if len (args.Entries) > 0 {rf.log = append (rf.log[:args.PrevLogIndex+ 1 ], args .Entries...)    }  
          
      
       
  
  
  
    // commit index if args.LeaderCommit > rf.commitIndex {rf.commitIndex = min(args.LeaderCommit, len (rf.log) -1 )    } } }  
      
  
  
  
  

### [](https://www.qtmuniao.com/2018/08/29/raft-log-replication/#CommitIndex-%E6%9B%B4%E6%96%B0 "CommitIndex update")CommitIndex update

It mainly relies on the matchIndex [] of the leader. The specific method is to sort them in ascending order, and then take the Index at the median position (I thought of it at the time, it feels amazing). Another point to note is that it is emphasized in the paper that only the logEntry in this Term can be submitted. This is for the repeated coverage caused by the Leader's singing and our appearance.

1   
2   
3   
4   
5   
6   
7   
8   
9   
10   
11   
12   
13   
14  

func  (rf *Raft) checkCommitIndex() {  
  peersCount := len (rf.peers)  
  matchIndexList := make ([] int , peersCount) copy (matchIndexList, rf.matchIndex)  sort.Ints(matchIndexList)  
    
  
  
  // match index before the "majority" are all matched by majority peers // before we inc commitIndex, we must check if its term match currentTerm   majority := peersCount / 2     peerMatchIndex := matchIndexList[majority] if peerMatchIndex > rf.commitIndex && rf.log[peerMatchIndex].Term == rf.currentTerm {        rf.commitIndex = peerMatchIndex } }  
    
  
  
      
  
  
  

### [](https://www.qtmuniao.com/2018/08/29/raft-log-replication/#%E5%93%8D%E5%BA%94%E6%8A%95%E7%A5%A8 "respond to vote")respond to vote

Each peer can cast one vote in each term at most, but after each term is updated, votedFor can be assigned a value of -1, that is, it can vote again. This case occurs when responding to a candidate's request to vote. If he has already voted, it seems that he cannot vote, but he finds that the term is not as large as the candidate, so he needs to become a Follower immediately and vote for the candidate.

1   
2   
3   
4   
5   
6   
7   
8   
9   
10   
11   
12   
13   
14   
15   
16   
17   
18   
19   
20 21   
22   
23   
24   
25   
26   
27   
28   
29   
30   
31  
  

func  (rf *Raft) RequestVote(args *RequestVoteArgs, reply *RequestVoteReply) {  
  rf.mu.Lock() defer rf.mu.Unlock()  reply.Term = rf.currentTerm  
    
      
  
  
  // once find a peer with higher term, follow if args.Term > rf.currentTerm {    rf.becomeFollower(args.Term)   }  
    
  
  
  
  // compare term and test if it voted if args.Term < rf.currentTerm || rf.votedFor != -1 {reply.VotedGranted = false return   }  
    
  
      
  
  
  // compare the last log entry  
   lastIndex := len (rf.log) - 1  
   lastLogTerm := rf.log[lastIndex].Term if args.LastLogTerm > lastLogTerm ||    args.LastLogTerm == lastLogTerm && args.LastLogIndex >= lastIndex { reply. VotedGranted = true  
    
  
  
  
    // convert to follower  
     rf.becomeFollower(args.Term)   
    rf.votedFor = args.CandidateId // do not forget  
   } else {   
    reply.VotedGranted = false  
   }   
}  

This involves the implementation of becomeFollower, that is, changing the state, then resetting the timer, and deciding whether to vote according to whether the term is updated.

1   
2   
3   
4   
5   
6   
7   
8   
9  

func  (rf *Raft) becomeFollower(term int ) {  
  DPrintf( "%d[%d] become follower" , rf.me, term)  
  rf.resetElectionTimer()  
  rf.state = Follower if term > rf.currentTerm {    rf. votedFor = -1   }  rf. currentTerm = term}