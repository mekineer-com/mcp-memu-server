# Historical database tools

Reference snapshots, not supported runtime commands. The launcher does not run
these tools, and their tests are outside the active `tests/` suite. Paths and
schema assumptions reflect the original location and may no longer apply.

Preserved for reference: category taxonomy migration, multimodal embedding
rebuild and embedding bakeoff, with their original tests.

The release migration runner and its tests are restored at the repository root
and in `tests/`. Run it only with services stopped and a database backup. Its
registered migration list is currently empty; restoring the command does not
repair old schema columns. Core updates invoke this command with the existing
database backup and rollback protection.
