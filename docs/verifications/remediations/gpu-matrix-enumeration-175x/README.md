# Remediations: GPU Matrix Enumeration 175x Speedup

| Issue | Severity | Found By | Status |
|-------|----------|----------|--------|
| No profiler output, code hash, or baseline CPU spec for 175x claim | important | o3-pro | open |
| 8-GPU scaling data (strong-scaling) not provided | minor | o3-pro | open |
| Kernel fusion contribution not isolated (no ablation study) | minor | o3-pro | open |
| v4 baseline timings beyond 10^7 are vague ("~hours") | minor | o3-pro | open |
| Speedup is relative to own prior implementation, not external baseline | minor | Claude Opus 4.6 | acknowledged |
