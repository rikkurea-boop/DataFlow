# P11.04 / P11.05 — Implementation handoff (pending runtime verification)

## Base and scope
- Base PR: OpenDCAI/DataFlow#548, SHA `e1ff8a377cf2a16409a8f22229e19e83e27653d2`.
- Branch: `bugfix/p11-04-05-registry-discovery-20261014`.
- Production file: `dataflow/utils/registry.py`.
- Regression file: `test/cpu_only/test_registry_recovery_determinism.py`.
- Upstream publication: not authorized.

## Failure modes identified
1. Discovery mutates `_obj_map` and `loader_map` during module import; exceptions can leave registrations and successfully imported sibling modules resident after failure.
2. `LazyLoader._load_class_from_file()` inserts its target module into `sys.modules` before `exec_module`, with no cleanup on failure.
3. Loader enumeration follows filesystem order, and registration silently uses the first colliding object encountered.

## Selected contracts
- Failed registry discovery/lookup rolls back registry object and loader maps and new import modules. Previously-existing modules and registrations are preserved.
- Direct LazyLoader execution failure removes its incomplete target module.
- Built-in registry registrations (under `dataflow.<registry_name>.*`) resolve collisions by lexicographically smallest source module/qualname. External explicit registrations preserve historical first-wins behavior.
- Full discovery remains lazy until called; mixed prompt module/package enumeration from #548 remains in place.
- Missing lookup remains `KeyError`; genuine import exceptions propagate.

## Verification status
- Regression tests are committed, but no executable RED/GREEN evidence is available yet.
- The repository's `.github/workflows/test.yml` runs on pushes to `main` and PRs targeting `main`; this branch alone does not trigger that workflow.
- GitHub Actions workflow runs for initial implementation commit: none.
- Local runtime environment cannot clone GitHub (DNS resolution of github.com unavailable), so focused tests and `pytest -m "cpu or api"` have **not** been run. Linux and Windows CI are **not verified**.
- Do **not** treat this implementation as tested/ready for upstream publication.

## Required next verification
1. Run the focused test under Python 3.10 and 3.12 (and other CI matrix versions).
2. Confirm it fails before the production fix (base #548), then passes afterward.
3. Run `test/cpu_only/test_prompt_registry_full_enumeration.py` and `pytest -m "cpu or api"`.
4. Revisit startup/lazy import compatibility and Windows behavior.
5. Obtain independent Manager review before any upstream PR.

## Known limitations
- Rollback cannot undo arbitrary external import side effects (I/O or mutations outside registries).
- Objects already returned to callers before later full enumeration can retain their original identity if a higher-precedence duplicate is subsequently discovered.
- Existing pre-imported module internals are not deep-copied by rollback.
