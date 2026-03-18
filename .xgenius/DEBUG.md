# Debug Log

Errors and issues encountered during autonomous research.

## 2026-03-18 ~07:00 — Phase 2 output-dir /output bug (all h003-h009 jobs)

**Problem:** All 105 Phase 2 pilot jobs (h003-h009) were submitted with `--output-dir /output`, but the Singularity container only bind-mounts the output directory to `/runs` (configured in xgenius.toml as `output_dir_container = "/runs"`). Training completed successfully but CSV saving failed with `OSError: [Errno 30] Read-only file system: '/output'`.

**Root cause:** Session 3 batch submission script hardcoded `--output-dir /output` in the commands, but the scripts' default is `/runs` (correct). The explicit `--output-dir /output` flag overrode the correct default.

**Impact:** All 105 Phase 2 jobs ran to completion but lost results. ~105 GPU-hours wasted.

**Fix:** Cancelled all running Phase 2 jobs, resubmitted 105 jobs WITHOUT `--output-dir` flag (using script default of `/runs`).

**Lesson:** Always verify output paths match container bind mounts. The correct container output path is `/runs`, not `/output`.
