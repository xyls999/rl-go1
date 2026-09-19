# Go1 hierarchical simulation workspace

Read `agent.md` before changing code or starting experiments. Its validated facts, interface version,
dependency checkpoints and unresolved issues are the project memory. Update it when those facts change.

Work only inside this workspace unless the user explicitly changes the scope. Use the existing Isaac Lab
and Python installation; do not edit the other desktop projects or silently switch their packages.

Implement real simulation and learning paths. Missing assets, sensors, dependencies, incompatible model
files, nonfinite inputs and training errors must remain visible. Never replace missing teachers with random
weights or camera frames with synthetic placeholders. A marked integration checkpoint is not a validated policy.

Preserve frozen model parameters and normalization; recurrent runtime state must still advance and reset
per environment. Actor deployment inputs must not read privileged simulation state. Test full export graphs,
not just action heads. Keep simulation-only goal sensing distinct from actual UWB hardware support.

User requirements: cover the requested data path, verify relevant changes, disclose failures, avoid unrelated
refactors and stop once the requested deliverables are complete. Parallel agents may work on independent
modules, with explicit ownership and integration by the primary agent. Keep comments focused on intent,
constraints and contracts rather than restating code or recording modification history.
