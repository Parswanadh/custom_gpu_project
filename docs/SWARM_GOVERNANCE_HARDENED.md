# 🛡️ Swarm Governance: Failure-Isolation & Cross-Stream Verification

## 1. Failure-Isolation Protocol (`trap_handler.md`)
Every agent MUST maintain a local `trap_handler.md` file. 
- **Trigger:** Upon encountering any syntax error, compilation failure, or assertion violation.
- **Action:**
    1. Write the error trace to `trap_handler.md`.
    2. Halt execution immediately.
    3. Issue a "CRITICAL FAULT" signal to the Stream Orchestrator.
    4. The Orchestrator MUST freeze the entire stream, preventing downstream tasks from running on potentially corrupted design state.

## 2. Cross-Stream Verification Protocol
No agent in Stream B (Physical) can commit RTL unless the corresponding SVA properties (Stream A) have passed the formal solver.
- **Dependency Tracking:** Every task in `plan.md` now lists its stream dependency.
- **Lock-Step Execution:** The Validator Swarm (Stream C) performs a 'Git-Hash Cross-Reference' before initiating regression tests. If the hash of `rtl/` does not match the hash logged by Stream B's orchestrator, the Validator Swarm issues a mandatory stop.

## 3. Implementation of Hardened SOPs
I am injecting these rules directly into the `skill` files of every agent. This ensures they cannot 'forget' the protocol.
