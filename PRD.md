### Motivation
- Inspired by Systems 1 and 2 from Daniel Kahneman's _Thinking, Fast and Slow_
- Exploratory research project by an undergrad computer science student and aspiring AI engineer
- An agentic system that emulates how humans actually think

### Architecture
- An agent that uses two models
- "System 1"
    - Very small non-reasoning model (<4B parameters)
    - Will decide whether or not to invoke System 2
    - Will also answer easy questions like System 1 can do in the book
    - Tool calls
        - Check memory
        - Answer problem
        - Invoke system 2
- "System 2"
    - Larger, reasoning model
    - More expensive, slow
    - Will be used for problems that require more focused/intentional thinking

### Project
1. Implement this architecture and apply it to the problem of basic multiplication
2. Create a dataset of elementary-level multiplication problems
3. Apply system to dataset
    - The system should first give the problems to system 1
    - System 1 will either decide the problem is easy enough to complete itself or will hand the problem off to system 2
    - Every problem it has seen before will be kept in a memory file that it will check before trying to solve the problem using the LLM
4. Note results
    - How often was system 2 invoked
    - Compare speed, token usage, and accuracy between pure-system-1 and pure-system-2 implementations for the same task
    - Write everything up in a presentable, white-paper LaTeX format