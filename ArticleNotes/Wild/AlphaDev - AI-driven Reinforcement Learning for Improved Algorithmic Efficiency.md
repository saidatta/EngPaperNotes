**Introduction**

Digital society's increasing computational and energy demand necessitates new strategies for enhancing computing power and sustainability. As we approach the physical limitations of microchips, we need to improve the code running on them, especially the algorithms executed trillions of times daily. 

---

**AlphaDev: AI Reinforcement Learning System**

AlphaDev is an AI system introduced in a paper published in Nature. It employs reinforcement learning to develop optimized computer science algorithms that surpass the efficiency of those devised over years by scientists and engineers.

Key Points:
- AlphaDev improved upon a widely used operation: sorting. 
- It discovered a faster algorithm for sorting that can be beneficial to countless applications ranging from online search ranking to data processing on personal devices. 
- The process of developing improved algorithms via AI has profound implications for programming and various aspects of the digital world.

---

**Applications of AlphaDev's Sorting Algorithm**

The sorting algorithm discovered by AlphaDev has been open-sourced in the main C++ library. Its incorporation signifies the first substantial change in the sorting library for over a decade and the first time an algorithm designed through reinforcement learning has been added.

Applications:
- Millions of developers worldwide can use this improved sorting algorithm in AI applications. 
- It impacts various industries including cloud computing, online shopping, and supply chain management.
- It represents a significant stepping stone in AI's role in optimizing the world's code, one algorithm at a time.

---

**Understanding Sorting**

Sorting is a process to arrange items in a specific order, like arranging letters alphabetically or numbers in ascending or descending order. This concept has evolved through history, from manual sorting in ancient libraries to the use of machines post the industrial revolution, and eventually to modern computer-based sorting algorithms. 

Example:
- Sorting the database of a large online shopping platform by price, customer ratings, etc.

---

**History of Sorting**

Historically, sorting has evolved over centuries, from manual sorting in libraries to the use of machines and modern computing methods.

Timeline:
- 2nd and 3rd Century: Scholars manually sorted thousands of books in the Great Library of Alexandria.
- Post Industrial Revolution: Invention of tabulation machines to sort information on punch cards, used in the 1890 U.S. census.
- 1950s: Development of the earliest computer science sorting algorithms with the advent of commercial computers.

---

**Assembly Instructions and Code Optimization**

AlphaDev optimizes algorithms by starting from scratch instead of refining existing ones. It looks for improvements in computer assembly instructions—a place often overlooked by human programmers.

- Assembly instructions are used to create binary code, the actionable format for computers. 
- Developers write in high-level languages like C++, which must be translated into 'low-level' assembly instructions for computers to understand.
- Potential improvements at this low level could have a larger impact on speed and energy usage.

---

**Translating High-Level Code to Assembly Instructions**

Code written in high-level programming languages like C++ is translated into assembly instructions using a compiler. The assembly instructions are then converted into executable machine code by an assembler, which can be run by the computer.

Example:
- C++ sorting algorithm (Figure A): High-level code written by a developer.
- Corresponding Assembly Code (Figure B): Translated low-level instructions that the computer can execute.

---

**Challenges and Opportunities**

Current sorting algorithms are products of decades of research, making them highly efficient. However, further refining these algorithms is challenging. Similar to finding a new energy-saving technique or more efficient mathematical approach, the development of enhanced algorithms is a significant step forward in the field of computer science. 

Opportunities:
- AlphaDev's approach of looking at low-level assembly instructions for improvements opens up new

---

**AlphaDev as an 'Assembly Game' Player**

To discover new algorithms, AlphaDev views the process of sorting as a single-player 'assembly game'. It observes the algorithm it has generated so far and the state of the CPU, then adds an instruction to the algorithm as its 'move'.

- The challenge lies in efficiently searching through a vast number of possible combinations of instructions to find an algorithm that sorts faster than the existing ones.
- The magnitude of possible combinations is compared to the number of particles in the universe or the number of possible moves in chess and Go games.
- A single incorrect move can invalidate the entire algorithm.

Example:
- Figure A: AlphaDev, the player, receives the state of the system as input and selects an assembly instruction to add to the algorithm.
- Figure B: After each move, the generated algorithm is tested with input sequences. The output is compared with expected results, and AlphaDev is rewarded based on correctness and latency.

---

**Verifying and Rewarding Algorithm Performance**

As AlphaDev builds the algorithm, it checks its correctness by comparing the algorithm's output to the expected results. In the context of sorting algorithms, unordered numbers go in, and correctly sorted numbers come out. AlphaDev is rewarded based on how correctly and quickly it sorts the numbers.

---

**Faster Sorting Algorithms Discovered by AlphaDev**

AlphaDev discovered new sorting algorithms, leading to improvements in the LLVM libc++ sorting library by up to 70% for shorter sequences and around 1.7% for sequences exceeding 250,000 elements.

Key Points:
- The focus was on improving sorting algorithms for shorter sequences of three to five elements, which are often called as part of larger sorting functions. 
- These improvements lead to an overall speedup for sorting any number of items.
- To make the new sorting algorithm more usable, it was reverse-engineered and translated into C++.

---

**Novel Approaches from AlphaDev**

AlphaDev discovered not just faster algorithms but also novel approaches. The new sorting algorithms contain new sequences of instructions that save a single instruction each time they're applied, making a substantial impact due to the trillions of daily uses.

Key Concepts:
- 'AlphaDev swap and copy moves': This novel approach is a shortcut that, although initially appearing like a mistake, leads to improved efficiency. 
- This innovation challenges conventional thinking about improving computer science algorithms, akin to the surprising move 37 by AlphaGo in its match against a Go world champion.

Example:
- The original sort3 implementation with min(A,B,C) vs. AlphaDev Swap Move with min(A,B).
- The original implementation with max(B, min(A, C, D)) in a larger sorting algorithm vs. AlphaDev's version using max(B, min(A, C)).

---

**Expanding to Hashing Algorithms**

After achieving faster sorting algorithms, AlphaDev was tested on a different algorithm: hashing. Hashing is a fundamental algorithm in computing used to retrieve, store, and compress data.

- Hashing takes a specific key, like a username, and converts it into a unique string of characters, which is used to quickly retrieve data associated with the key.
- When applied to the 9-16 bytes range of the hashing function, the algorithm AlphaDev discovered was 30% faster.
- AlphaDev’s new hashing algorithm is now available in the open-source Abseil library.

---

**Optimizing the World’s Code**

AlphaDev has demonstrated its ability to generalize and discover new algorithms

 with real-world impact by optimizing and launching improved sorting and hashing algorithms.

- Current exploration focuses on AlphaDev’s ability to optimize algorithms directly in high-level languages like C++, which would be more beneficial for developers.
- AlphaDev's discoveries, like the swap and copy moves, have the potential to inspire researchers and developers to further optimize fundamental algorithms for a more powerful and sustainable computing ecosystem.