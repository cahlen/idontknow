# Remediations: Zaremba Conjecture Proved (REVISE_AND_RESUBMIT)

| Issue | Severity | Found By | Status |
|-------|----------|----------|--------|
| rho_eta computed in FP64, not interval-certified (gap 1) | critical | Claude Opus 4.6, Grok, GPT-5.2 | open |
| MOW theorem matching not verified against specific theorem statements (gap 2) | critical | Claude Opus 4.6, Grok, GPT-5.2, o3-pro | open |
| C_eta extraction not rigorous (gap 3) | critical | Grok | open |
| Dolgopyat bound applies only to truncated model, not full operator | critical | o3-pro | open |
| Layer 4 (property tau tail) is non-effective and unverifiable | critical | o3-pro | open |
| Title says "proof" but 6 gaps remain; should say "proof framework" | important | Claude Opus 4.6, o3-pro | open |
| Multiple gaps remain, not "one specialist gap" as claimed | important | o3-pro | open |
| Tauberian step only sketched, needs explicit error constants | important | Grok | open |
| Transitivity proof had incorrect |H| bound (line 304) | important | GPT-5.2 | resolved |
| D0 LaTeX formatting ambiguous (886^3.57 misread) | minor | GPT-5.2 | resolved |
| c1 = 0.6046 is FP64 (6x margin tolerates ~20% error) | minor | GPT-5.2 | acknowledged |
| Brute-force verification to 2.1e11 with zero failures | none | All reviewers | resolved |
