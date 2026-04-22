# BitbyBit Expo Master Judge Playbook

Date: 2026-04-09  
Audience: 3-person team preparing for science Expo judging tomorrow  
Scope: Entire BitbyBit workspace (hardware + benchmark pipeline + showcase websites)

---

## 1) What This Document Is

This is your single source for:
1. What every major folder does.
2. What claims are safe and evidence-backed.
3. How judges usually think and score.
4. How to answer hard questions with confidence.
5. Exactly what not to claim.

If you memorize only one thing, memorize this sentence:

"We built a custom Verilog transformer-acceleration stack with fail-closed validation; our latest full matrix run (run_id 20260401-142256) shows 358 to 112 cycles (3.1964x) at 100 MHz simulation, with full evidence artifacts and 55-module regression at 323 PASS, 0 FAIL."

---

## 2) One-Minute Pitch Script

"We built BitbyBit as a full-stack hardware inference project: Verilog RTL modules, testbenches, benchmark orchestration, proof-pack validation, and a web evidence layer that reads live payloads from simulation artifacts.  
The key result is from run_id 20260401-142256: baseline full-model path is 358 cycles and imprint path is 112 cycles, for 3.1964x speedup at 100 MHz simulation.  
We enforce reliability with fail-closed automation using validate_full_chain.ps1, which runs full regression, production benchmark, WS1 parity gate, Vitest, and Playwright, then emits validation_manifest_latest.json.  
Our current regression anchor is 55 modules, 323 PASS, 0 FAIL, and WS1 parity gate is passing across dim16, dim32, dim64 with run_id 20260401-085547.  
We are transparent that this is simulation-measured, not post-route silicon timing, and we separate measured claims from exploratory metrics."

---

## 3) Fast Claim Card (Memorize)

Primary measured claim set:
1. Regression: 55 modules, 323 PASS, 0 FAIL.
2. Benchmark run_id: 20260401-142256.
3. Workload mode: matrix, 20 workloads, 20 measured runs, 400 measured samples.
4. Baseline cycles mean: 358.
5. Imprint cycles mean: 112.
6. Speedup mean: 3.1964x.
7. Baseline tokens/s mean: 279,329.
8. Imprint tokens/s mean: 892,857.
9. Baseline MEDUSA effective tokens/s mean: 837,988.
10. Imprint MEDUSA effective tokens/s mean: 2,678,571.

Reliability claim set:
1. Full-chain orchestrator exists and is fail-closed: scripts/validate_full_chain.ps1.
2. Stages tracked in validation_manifest_latest.json: full_regression, production_benchmark, ws1_parity_gate, website_vitest, website_playwright_suite.
3. Latest manifest status for all above stages: PASS.
4. WS1 parity run_id: 20260401-085547.
5. WS1 token parity pass: 24/24 for dim16, 24/24 for dim32, 24/24 for dim64.

Mandatory caveat sentence:
"All reported performance is simulation-measured at 100 MHz on RTL/testbench flow, not post-route silicon timing."

---

## 4) Entire Workspace Map (What Folder Does What)

## Root: d:/Projects/BitbyBit
1. auto-git-website/: Main Next.js evidence website used to present benchmark payload and validation data.
2. custom_gpu_project/: Core hardware + simulation + proof artifacts + scripts + docs.
3. react-ui-component-builder/: Internal skill/pattern docs for UI style and components.
4. check_site.js: Utility script related to site checks.
5. MCP_SOTA_FREE_STACK.md: MCP stack and tooling notes.

## auto-git-website (presentation app)
Purpose: Public-facing presentation layer that reads benchmark artifacts and renders metrics sections.

Key folders:
1. app/: Next.js routes and layout.
2. components/: Section-by-section UI blocks (Hero, Metrics, Comparison, ExecutionEvidence, etc.).
3. lib/: Data loader and type contracts.
4. tests/: Unit, integration, E2E validation.

Key files and responsibilities:
1. app/page.tsx: Top-level page; calls getWebsiteBenchmarkPayload() and passes benchmark prop to all major sections.
2. app/api/benchmark/route.ts: API endpoint for benchmark payload with Cache-Control no-store.
3. lib/benchmark-metrics.ts: Core payload resolver/parser/transformer with UTF-8 -> UTF-16 fallback and validation manifest summarization.
4. lib/benchmark-metrics-types.ts: Strong types for payload and chart points.
5. tests/e2e/homepage.spec.ts: Verifies API and UI metric consistency and caveat visibility.
6. tests/integration/benchmark-metrics.integration.test.ts: Validates transform shape and latency math.

Section folders in components/:
1. sections/: Actual homepage sections (ArchitectureSection, MetricsDashboardSection, ComparisonSection, etc.).
2. hero/: Hero visuals and stat cards.
3. metrics/: Charts and metric visualizations.
4. architecture/, pipeline/, quality/, error-memory/, debate/: Thematic deep-tech sections.
5. navigation/, shared/, ui/: Navigation, shared widgets, base UI.

## custom_gpu_project (core technical project)
Purpose: Hardware architecture, validation, benchmark generation, and research documentation.

Key folders:
1. rtl/: Verilog modules by domain.
2. tb/: Testbenches by domain.
3. scripts/: Automation (regression, benchmark, parity, payload validation, weight export).
4. sim/: Generated artifacts and logs (large evidence repository).
5. docs/: Human-readable strategy, architecture, expo prep.
6. website_next/: Secondary Next.js website implementation.
7. website/: Static/legacy website implementation.
8. weights/: Generated and cached model/imprint artifacts.
9. model/: Currently empty placeholder folder in this workspace snapshot.

rtl/ domain map:
1. primitives/: Foundational arithmetic/control building blocks.
2. compute/: MAC, softmax, GELU, sparse/ternary/mixed-precision modules.
3. transformer/: Layer norm, attention, FFN, RoPE, GQA modules.
4. memory/: AXI memory, KV cache quantizer, memory controllers.
5. control/: Command/config/performance control blocks.
6. integration/: Multi-module integration layers (e.g., optimized transformer layer).
7. gpt2/: GPT2-focused engine path modules.
8. top/: Top-level system wrappers.

tb/ domain map:
1. Mirrors rtl domains for verification.
2. Includes demo and cocotb areas.
3. Enforces pass/fail patterns parsed by regression scripts.

scripts/ important files:
1. run_all_tests.ps1: Full Verilog regression harness and parser.
2. run_production_demo.ps1: Canonical production benchmark flow.
3. run_demo.ps1: Demo wrapper called by production flow.
4. validate_full_chain.ps1: Master fail-closed orchestrator.
5. run_ws1_scale_proof.py: WS1 dimension sweep/parity gate with enforce option.
6. build_phase3_benchmark_proof_pack.py: Proof-pack build.
7. validate_benchmark_payload.py: Contract/schema validation.
8. export_gemma3_imprint.py and extract_gpt2_weights.py: Model/weight preparation.

sim/ purpose:
1. Source of truth artifacts used by websites and judge claims.
2. Contains compare_summary_latest.json and validation_manifest_latest.json.
3. Contains regression logs, proof packs, WS1 parity reports, and run archives.
4. Contains many generated logs and binaries; treat as evidence warehouse.

docs/ purpose:
1. EXPO_WIN_PLAYBOOK_31_03.md: Judge-facing claims and reproducibility framing.
2. Judge_QA.md: Long-form Q and A prep.
3. SOTA_Benchmark_Showcase_31_03.md: Benchmark claim framing and artifact references.
4. architecture.md/system_architecture.md: Technical architecture references.
5. progress.md: Historical implementation trail.

website_next/ purpose:
1. Another Next.js app for presenting project narrative.
2. Source-only footprint is moderate (exclude node_modules/.next).

website/ purpose:
1. Static site stack (assets/css/js/index.html).
2. Useful as fallback/demo variant.

---

## 5) End-to-End Evidence Flow (What Connects to What)

Flow:
1. Hardware tests and benchmark runs execute from custom_gpu_project/scripts.
2. Outputs land in custom_gpu_project/sim/.
3. compare_summary_latest.json becomes primary benchmark payload.
4. validation_manifest_latest.json captures full-chain stage outcomes.
5. auto-git-website/lib/benchmark-metrics.ts resolves and parses payload.
6. auto-git-website/app/page.tsx and /api/benchmark consume transformed payload.
7. UI sections show cycle/speedup/caveat/validation data directly from payload.

Important design protections:
1. If payload file is missing, benchmark loader throws explicit error.
2. Loader supports UTF-8 and UTF-16LE JSON decoding fallback.
3. API route disables cache with no-store.
4. E2E test compares API payload vs UI raw hooks for consistency.

---

## 6) Judge Psychology: How They Actually Score

Based on rubric research from multiple science-fair sources (ISEF ecosystem pages + public judge rubrics from BNL, Science Fair Foundation BC, LA Science Fair, Maine State Science Fair), judges repeatedly converge on the same dimensions.

Common scoring dimensions that recur:
1. Originality/Creativity.
2. Scientific thought or engineering thought.
3. Method quality (controls, variables, repeatability, proper testing).
4. Data collection quality and analysis depth.
5. Conclusion validity and limitations awareness.
6. Clarity of presentation and interview quality.
7. Degree of student independence.
8. Real-world relevance/impact.
9. Future work quality.
10. Team collaboration quality (for team projects).

Observed weighting patterns from extracted rubrics:
1. Creativity can be high weight (often around 20-30%).
2. Scientific/engineering rigor is heavily weighted.
3. Interview performance is frequently high weight (example rubric shows 25/100 for interview).
4. Display/poster quality matters, but usually less than methodology + reasoning.

Behavior pattern of strong judges:
1. They test understanding depth, not memorized speeches.
2. They probe limitations and error sources quickly.
3. They often ask what the student did personally vs assisted work.
4. They value honesty about scope limits more than inflated claims.
5. They reward reproducibility and clear evidence trails.

Interview style clues from judge guides:
1. Judges ask for a summary first.
2. Then they ask controls/variables/method/data/conclusion questions.
3. Then they ask extensions: future work, real-world use, mistakes, what changed.
4. They expect equal participation from all team members.

How this maps to your project:
1. Your strongest edge is reproducibility + validation discipline.
2. Your biggest risk is over-claiming simulation results as silicon results.
3. Your second biggest risk is not separating baseline vs imprint context.
4. Your interview score rises if you lead with caveat transparency proactively.

---

## 7) 3-Person Team Role Strategy

Assign fixed roles for every judge interaction.

Role A: System Architect (Person 1)
1. Owns high-level narrative and architecture map.
2. Explains rtl/tb/scripts/sim/docs relationships.
3. Handles novelty, tradeoffs, and design rationale.

Role B: Verification and Metrics Lead (Person 2)
1. Owns all numbers, run_id references, and artifact paths.
2. Explains regression results, validation chain, proof-pack, parity gate.
3. Handles questions on confidence, reproducibility, and stats.

Role C: Product and Presentation Lead (Person 3)
1. Owns website data flow and UI evidence explanation.
2. Explains benchmark loader, API behavior, no-store, E2E consistency checks.
3. Handles communication, impact framing, and user-facing clarity.

Interview handoff protocol:
1. Person 1 opens in 20-30 seconds.
2. Person 2 takes metrics and evidence in 30-40 seconds.
3. Person 3 closes with reproducibility and caveat in 20 seconds.
4. For each question, one primary answerer plus one short supporting add-on.

---

## 8) High-Probability Judge Questions and Best Answers

## A) Novelty and Design

Q1. What is your true innovation?
A. End-to-end co-design: custom Verilog inference modules plus fail-closed validation and payload-driven evidence website. Not only a model demo, but architecture + verification + transparent reporting pipeline.

Q2. What does imprint path mean?
A. It is the hardwired-weights path in this project flow. In our benchmark artifacts, it is the faster measured path (112 cycles mean) versus baseline (358 cycles mean) for the same benchmark setup.

Q3. Is this a full GPU replacement?
A. No. It is a specialized research architecture showing transformer-focused acceleration techniques and verification discipline.

Q4. Why build from scratch instead of using existing frameworks?
A. To expose and optimize architecture-level choices directly in RTL and to produce transparent, reproducible evidence from gate-level simulation workflows.

## B) Correctness and Verification

Q5. How do you know results are correct?
A. Full regression and fail-closed orchestration. Latest regression summary is 55 modules, 323 PASS, 0 FAIL. Full-chain manifest stages are PASS.

Q6. How do you prevent silent failures?
A. Scripts enforce non-zero exits and stage failure propagation. validate_full_chain.ps1 records stage status and fails overall on any stage FAIL.

Q7. How do you verify across dimensions?
A. WS1 parity gate run_id 20260401-085547 reports parity pass for dim16, dim32, dim64 with 24/24 token pass each.

Q8. What if a payload file is malformed or encoded differently?
A. benchmark-metrics.ts parses UTF-8 first, then UTF-16LE fallback, and throws if parsing fails.

## C) Metrics Integrity

Q9. Are these numbers hardcoded in the website?
A. No. auto-git-website/lib/benchmark-metrics.ts resolves compare_summary_latest.json and transforms live payload values.

Q10. How do you ensure UI and API show same numbers?
A. Playwright test homepage.spec.ts compares API payload values with UI raw test hooks.

Q11. What exact run anchors your headline claim?
A. run_id 20260401-142256 in sim/compare_summary_latest.json and benchmark proof-pack artifacts.

Q12. Why should we trust that run?
A. It includes seeded matrix workload, sample counts, stats fields, and separate manifest/proof artifacts under fail-closed validation.

Q13. Did you measure on real silicon?
A. No. Simulation-measured at 100 MHz assumption. We explicitly state this caveat in docs and payload labels.

## D) Scientific Method and Judge Rubric Alignment

Q14. Where are controls/variables in your benchmark design?
A. Controlled workload seed, fixed benchmark frequency assumption, fixed run counts and workload counts; compared baseline vs imprint paths under matched benchmark procedure.

Q15. How did you address repeatability?
A. Matrix mode with repeated measured runs and fixed seed, plus reproducibility commands documented.

Q16. What are your main error sources or uncertainty points?
A. Simulation-to-silicon timing gap, model-scope constraints, and possible external environment differences when reproducing toolchain.

Q17. What did you personally build versus tooling assistance?
A. Project consists of hand-authored RTL, testbench and orchestration scripts, and custom website payload integration/testing; external libraries are used for ecosystem tooling (Next.js, Vitest, Playwright).

Q18. What would be your next experiment?
A. Post-route timing and hardware implementation validation to convert cycle-level evidence into physically measured latency/power/area.

## E) Limitations and Honesty

Q19. What is your biggest limitation today?
A. Current performance is simulation-measured, not on fabricated silicon or FPGA timing closure.

Q20. Second biggest limitation?
A. Scope specialization: benchmark and architecture path are focused and should not be presented as universal accelerator supremacy.

Q21. Are MEDUSA-effective numbers primary claims?
A. No. They are exploratory context; primary judge claim remains baseline vs imprint measured cycle and throughput metrics.

Q22. Can this generalize to all LLM workloads right now?
A. Not yet proven broadly. We present the evaluated setup and artifacted workload matrix only.

Q23. Is every doc in repo equally up to date?
A. No. We treat latest anchored artifacts and latest validated run IDs as source of truth over historical notes.

## F) Website and Communication

Q24. Why include a website at all?
A. To make verification evidence transparent and inspectable for non-RTL audiences without changing underlying source artifacts.

Q25. What if /api/benchmark fails during demo?
A. Route returns explicit error envelope and sections handle unavailable payload gracefully; we still keep raw artifact files and command logs as primary evidence.

Q26. Why no-store on API?
A. Prevent stale benchmark payload caching and ensure current artifact-backed values.

Q27. How do you avoid flashy UI overpowering substance?
A. Every key displayed metric is tied to payload values and tested; caveat text is shown with metrics.

## G) Team and Process

Q28. How was team work divided?
A. Architecture/RTL focus, verification/benchmark focus, and presentation/web integration focus with common artifact checkpoints.

Q29. How did you resolve disagreements?
A. By using fail-closed evidence runs and run_id-anchored artifacts as decision criteria.

Q30. Why should this win?
A. Strong integration of hardware architecture, rigorous validation, reproducibility discipline, and transparent communication of both strengths and limitations.

## H) Advanced Technical Pushback

Q31. Is 3.1964x speedup just because of a special path?
A. It is measured baseline-vs-imprint under the defined setup; we clearly distinguish path type and avoid claiming universal speedup.

Q32. What proves your regression parser is robust?
A. run_all_tests.ps1 has explicit parser order handling (TB_RESULT, summary patterns, then markers) and hard failure modes.

Q33. How do you guard against run orchestration drift?
A. validate_full_chain.ps1 centralizes canonical stage execution and records stage results in manifest JSON.

Q34. Can judges replicate in one command?
A. Yes, validate_full_chain.ps1 orchestrates full sequence; run_production_demo.ps1 provides benchmark-focused path.

Q35. What is the core scientific contribution beyond implementation effort?
A. A measurable, reproducible architecture-validation workflow demonstrating how custom hardware design claims can be tied to fail-closed evidence and transparent reporting.

---

## 9) Do-Not-Claim List (Critical)

Never say:
1. "This is already proven on real silicon."
2. "This beats all commercial GPUs."
3. "These results generalize to all models and workloads."
4. "Power and area are fully characterized."
5. "MEDUSA-effective throughput is our primary production metric."

Safe replacements:
1. "Simulation-measured at 100 MHz in this validated setup."
2. "Measured speedup in our baseline-vs-imprint path for run_id 20260401-142256."
3. "Reproducible within the documented pipeline and artifacts."
4. "Future work includes post-route timing and hardware implementation validation."

---

## 10) Judge-Trap Recovery Scripts

If asked something you do not know exactly:
1. "Great question. I do not want to guess. Here is what we have verified in artifacts right now: [state anchored fact]."
2. "Our current validated claim is [fact]. The extension you asked is planned as future work in [doc/roadmap]."
3. "We can show the exact file/run right now."

If challenged on fairness:
1. "Agreed. That is why we present baseline and imprint side by side with the same benchmark setup and explicit caveats."

If challenged on rigor:
1. "We enforce fail-closed stages and track stage outcomes in validation_manifest_latest.json."

---

## 11) Reproducibility Command Pack

From custom_gpu_project root:

```powershell
# Full regression
powershell -ExecutionPolicy Bypass -File .\scripts\run_all_tests.ps1

# Production benchmark + proof-pack flow
powershell -ExecutionPolicy Bypass -File .\scripts\run_production_demo.ps1 -WorkloadMode matrix -WarmupRuns 5 -MeasuredRuns 20 -WorkloadCount 20 -WorkloadSeed 20260331

# Full-chain fail-closed validation (includes website tests)
powershell -ExecutionPolicy Bypass -File .\scripts\validate_full_chain.ps1

# WS1 parity gate (explicit)
python .\scripts\run_ws1_scale_proof.py --dims 16,32,64 --workload-count 24 --workload-seed 20260331 --token-space 16 --position-space 8 --seq-len 32 --enforce-gate
```

Primary evidence files:
1. sim/full_regression_20260401.log
2. sim/compare_summary_latest.json
3. sim/validation_manifest_latest.json
4. sim/phase3_benchmark_proof_pack.json
5. sim/parity_report.json
6. sim/dim_sweep_report.json

Website evidence files:
1. auto-git-website/lib/benchmark-metrics.ts
2. auto-git-website/app/page.tsx
3. auto-git-website/app/api/benchmark/route.ts
4. auto-git-website/tests/e2e/homepage.spec.ts
5. auto-git-website/tests/integration/benchmark-metrics.integration.test.ts

---

## 12) 12-Minute Team Presentation Structure

Minute 0-1:
1. Problem framing and why custom acceleration architecture matters.

Minute 1-3:
1. Folder-level architecture map (rtl, tb, scripts, sim, docs, website).

Minute 3-6:
1. Validation pipeline and evidence flow.
2. Explain fail-closed orchestrator and artifact outputs.

Minute 6-8:
1. Headline metrics with run_id and caveat.
2. Baseline vs imprint explanation.

Minute 8-10:
1. Website evidence demo and API/UI consistency.

Minute 10-11:
1. Limitations and honesty section (simulation scope, future hardware validation).

Minute 11-12:
1. Closing value proposition and reproducibility confidence statement.

---

## 13) Tonight Rehearsal Plan (High Priority)

Round 1 (45 min):
1. Each person memorizes 10 key facts and 10 Q/A.
2. Practice the 1-minute pitch until smooth.

Round 2 (45 min):
1. Mock hostile judge session.
2. Force each member to answer at least 10 hard questions.

Round 3 (30 min):
1. Artifact drill: each member must locate key files instantly.
2. Practice saying run_id and caveat without hesitation.

Final 20 min:
1. Decide primary speaker order and handoff phrases.
2. Finalize no-claim boundaries.

---

## 14) Expo Day Checklist

Before first judge arrives:
1. Open compare_summary_latest.json, validation_manifest_latest.json, and full_regression_20260401.log.
2. Open auto-git-website page and /api/benchmark endpoint.
3. Keep run_id and caveat card visible.
4. Confirm who answers which category.

During Q&A:
1. Start precise, short, evidence-backed.
2. Mention caveat proactively.
3. Never guess numbers.
4. If uncertain, anchor to file and run_id.

After each judge:
1. Capture the hardest question asked.
2. Update team script immediately.

---

## 15) External Judge-Rubric Insights Used

Official ecosystem references:
1. ISEF International Rules: https://www.societyforscience.org/isef/international-rules/
2. ISEF Rules and Guidelines (HTML): https://www.societyforscience.org/isef/international-rules/rules-and-guidelines/
3. ISEF Forms: https://www.societyforscience.org/isef/forms/
4. ISEF Rules FAQ: https://www.societyforscience.org/isef/international-rules/faq/

Rubric and judging behavior references extracted for pattern analysis:
1. Brookhaven judging rubric (scientific method/engineering process dimensions): https://www.bnl.gov/sciencefair/files/pdf/judging-criteria.pdf
2. Science Fair Foundation BC rubric and judging workflow notes: https://sciencefairs.ca/wp-content/uploads/2025/10/Session-3-Judging-Rubrics-and-Scoresheets-Slides-V2.pdf
3. LA Science Fair judging guidelines and weighted criteria notes: http://www.lasciencefair.org/Forms/JudgingCriteria.PDF
4. Maine State Science Fair rubric (100-point structure with interview emphasis): https://www.maine-state-science-fair.com/wp-content/uploads/2021/09/rubrics.pdf
5. Mississippi State rubric copy (modified lower-fair criteria structure): https://www.sciencefair.msstate.edu/wp-content/uploads/LF_rubric_2023_24.pdf

Judge-pattern conclusion from these sources:
1. You win by combining originality + rigorous method + data clarity + interview confidence + honest limits.
2. Your project is strongest when you lead with evidence and caveat before the judge asks.

---

## 16) Final Confidence Statement for Judges

"Our team focused on measurable engineering truth: we built the architecture, validated it through fail-closed automation, and present only claims tied to reproducible artifacts. We are proud of the speedup and equally clear about its simulation scope."
