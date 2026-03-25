---
name: refactoring-plan
description: Structured phase-by-phase refactoring workflow for AI-friendly code architecture, domain modeling, and code quality. Use when the user asks to refactor, restructure, clean up, or improve code architecture, or mentions making code more maintainable or AI-readable.
---

# Refactoring Plan

A structured, phase-by-phase approach to refactoring code so that AI can reason about it accurately and consistently. **Behavior must never change as a result of refactoring.**

---

## Before Starting

- Save the current state as a Cursor checkpoint for rollback.
- Do NOT begin refactoring until explicitly approved for each phase.
- Work phase by phase — never refactor everything at once.
- After each phase, verify all existing behavior is preserved before proceeding.
- Track and review the Diff at each phase boundary.

---

## Phase 0 — Checkpoint

- Create a Cursor checkpoint at the current state.
- This checkpoint must be restorable at any point during the refactoring process.

---

## Phase 1 — Domain Model Identification (Read Only)

**Goal: Establish AI-reasonable concept-level domain boundaries.**

- Identify all core domains in the codebase.
  - Examples: `Flight`, `Gate`, `Runway`, `Schedule`, `Contact Stand`, `Remote Stand`, `Aircraft`, `Taxiway`, `Runway Taxiway`
- For each domain, document:
  - What data it holds
  - What functions operate on it
  - What it depends on
- Map dependencies between domains.
- **No code changes — analysis only.**

---

## Phase 2 — Static Config Extraction → `information.json`

**Goal: Single source of truth for all static values.**

- Identify all hardcoded values:
  - Numeric constants (timeouts, thresholds, capacities, intervals)
  - UI values (colors, sizes, spacing, fonts, layout dimensions)
  - Decorative or styling configuration
- Move all identified values into `information.json`.
- For `layout_design.py`: all design-related values must be read from `information.json`, not hardcoded.
- **Code logic must not change — values only.**

---

## Phase 3 — Dead Code & Debug Cleanup

**Goal: Remove noise so subsequent phases are simpler.**

- Remove all debug `print()` / `log()` / `console.log()` statements.
- Remove dead code: functions and variables never called or referenced.
- Remove commented-out code blocks no longer needed.
- **No logic changes — removal only.**
- Confirm behavior is preserved after cleanup.

---

## Phase 4 — Function Decomposition & Renaming

**Goal: Break down large functions, establish clear naming.**

### Decomposition Rules

- Any function over 200 lines must be broken down.
- Split by responsibility type:
  - `calculate_*` — pure computation, no side effects
  - `validate_*` — input/condition checking, returns bool or error
  - `transform_*` — data shape conversion
  - `render_*` — UI/display only
- Each function must do exactly one thing.

### Naming Rules

- Rename any function or variable whose name does not clearly describe its purpose.
- Use domain-based naming: prefer `get_flight_departure_delay()` over `get_delay()`.
- Conventions:
  - Functions/variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Classes/domains: `PascalCase`
- Confirm behavior is preserved after each function split.

---

## Phase 5 — Deduplication & Abstraction

**Goal: Eliminate repetition, make hardcoded logic universal.**

### Deduplication

- Find all blocks performing the same operation in multiple places.
- Extract into a shared, reusable function.
- The shared function must handle all existing call sites.

### Abstraction

- Identify logic hardcoded for specific conditions (e.g., specific flight types, gate IDs).
- Refactor into parameter-driven, universal functions.
- Example: `if type == "A320": value = 90` → `get_separation_interval(aircraft_type: str) -> int`

### Redundant Computation

- Find values recomputed on every call but never change between updates.
- Cache or pre-compute at the appropriate scope.
- Example: recalculating a static layout on every render → compute once, store result.
- Confirm behavior is preserved after each change.

---

## Phase 6 — Side Effect Removal & Error Handling

**Goal: Pure, testable functions with structured error handling.**

### Side Effect Removal

- Identify functions that modify external state, globals, or UI as a side effect.
- Separate pure computation from side effects.
- Functions should take inputs and return outputs — nothing more.

### Clean Input/Output

- Every function must have clearly typed inputs and outputs.
- No implicit dependencies on global state inside functions.
- Add type annotations to all function signatures.

### Error Handling

- Define explicit error types for each domain:
  - Example: `FlightNotFoundError`, `InvalidRunwayAssignmentError`, `ScheduleConflictError`
- Replace silent failures and bare `except` blocks with structured error handling.
- Always log errors with meaningful context — never swallow exceptions silently.
- Confirm each refactored function is independently testable.

---

## Phase 7 — AI-Friendly Pattern Unification

**Goal: Consistent, predictable code style across the entire codebase.**

- Unify code style so similar operations always look the same.
- Maintain repeating structural patterns — AI should predict the shape of any function.
- Ensure all functions follow the same input → process → output flow.
- Final Diff review: confirm zero behavior changes across all phases.

---

## Rules for Every Phase

- Never refactor the entire codebase at once — work in feature-level units within each phase.
- After every unit of work, verify existing behavior is intact.
- Do not modify logic unrelated to the current phase's scope.
- Do not introduce new features during refactoring.
- If a potential improvement is spotted outside scope, note it as a `// TODO:` and move on.
- All final code must be complete and functional — no placeholders.
