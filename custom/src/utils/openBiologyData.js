export const overviewMetrics = [
  {
    id: 1,
    completed_number: 5,
    total_number: 6,
    title: "Pipelines live",
    progress: "82%",
    progress_info: "5/6 actively orchestrated",
    icon: "feather-cpu",
  },
  {
    id: 2,
    completed_number: 38,
    total_number: 52,
    title: "Queued jobs",
    progress: "73%",
    progress_info: "balanced across sites",
    icon: "feather-activity",
  },
  {
    id: 3,
    completed_number: 18,
    total_number: 24,
    title: "Data assets tracked",
    progress: "75%",
    progress_info: "matrices + images + omics",
    icon: "feather-database",
  },
  {
    id: 4,
    completed_number: 92,
    total_number: 100,
    title: "QC pass rate",
    progress: "92%",
    progress_info: "rolling 7d",
    icon: "feather-shield",
  },
];

export const openBiologyModules = [
  {
    key: "biomarker-analysis",
    name: "Biomarker Analysis Agent",
    focus: "LLM agent that triages biomarkers, generates hypotheses, and writes reports.",
    status: "Operational",
    progress: 82,
    owner: "AI/ML",
    dependencies: ["Biomarker Matrices", "Genomic Sequencing"],
    health: "Stable",
    activeRuns: 7,
    lastRun: "14m ago",
  },
  {
    key: "biomarker-matrices",
    name: "Biomarker Matrices",
    focus: "Normalization matrices, cohort stratification, and panel scoring utilities.",
    status: "Ingesting",
    progress: 68,
    owner: "Data Engineering",
    dependencies: ["Genomic Sequencing"],
    health: "Warm",
    activeRuns: 5,
    lastRun: "49m ago",
  },
  {
    key: "bioprocess-automation",
    name: "Bioprocess Automation",
    focus: "Closed-loop control for upstream runs, QC gate automation, and audit trails.",
    status: "Running",
    progress: 74,
    owner: "Bioprocess Ops",
    dependencies: [],
    health: "Stable",
    activeRuns: 9,
    lastRun: "5m ago",
  },
  {
    key: "genomic-sequencing",
    name: "Genomic Sequencing",
    focus: "Sequencer integration, demux, coverage QC, and variant packaging.",
    status: "QC gating",
    progress: 63,
    owner: "Genomics",
    dependencies: [],
    health: "Attention",
    activeRuns: 4,
    lastRun: "1h ago",
  },
  {
    key: "imaging-agent",
    name: "Imaging Agent",
    focus: "Vision agent for microscopy tasks, annotations, and runbook generation.",
    status: "Operational",
    progress: 70,
    owner: "Imaging",
    dependencies: ["Biomarker Matrices"],
    health: "Stable",
    activeRuns: 6,
    lastRun: "32m ago",
  },
];

export const moduleResources = {
  "biomarker-analysis": {
    repo: "/Users/o11/Documents/OpenSC/OpenBiology/OpenBiology-BiomarkerAnalysisAgent",
    docs: "docs/ or README in BiomarkerAnalysisAgent",
    pages: ["Agent loops", "Prompt guardrails", "Report publishing"],
  },
  "biomarker-matrices": {
    repo: "/Users/o11/Documents/OpenSC/OpenBiology/OpenBiology-BiomarkerMatrices",
    docs: "README + data dictionaries",
    pages: ["Matrix builds", "Panel scoring", "Cohort stratification"],
  },
  "bioprocess-automation": {
    repo: "/Users/o11/Documents/OpenSC/OpenBiology/OpenBiology-BioprocessAutomation",
    docs: "Runbooks + automation scripts",
    pages: ["Upstream control", "QC interlocks", "Audit streaming"],
  },
  "genomic-sequencing": {
    repo: "/Users/o11/Documents/OpenSC/OpenBiology/OpenBiology-GenomicSequencing",
    docs: "Sequencer SOPs + QC notebooks",
    pages: ["Demux + QC", "Coverage dashboards", "Variant packaging"],
  },
  "imaging-agent": {
    repo: "/Users/o11/Documents/OpenSC/OpenBiology/OpenBiology-ImagingAgent",
    docs: "Vision agent README + overlays",
    pages: ["Microscopy ingest", "Annotation pipelines", "Runbook drafting"],
  },
};

export const pipelinePerformance = {
  categories: ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
  series: [
    {
      name: "Biomarker Analysis",
      type: "bar",
      data: [22, 28, 31, 35, 38, 40],
    },
    {
      name: "Matrices",
      type: "bar",
      data: [15, 18, 20, 22, 25, 29],
    },
    {
      name: "Bioprocess",
      type: "line",
      data: [30, 33, 36, 39, 42, 45],
    },
    {
      name: "Sequencing",
      type: "line",
      data: [18, 20, 23, 24, 26, 28],
    },
    {
      name: "Imaging",
      type: "area",
      data: [12, 15, 18, 20, 22, 24],
    },
  ],
};

export const runActivity = [
  {
    id: 1,
    moduleKey: "bioprocess-automation",
    module: "Bioprocess Automation",
    item: "Fed-batch 24L control loop",
    stage: "Run-time",
    status: "Running",
    owner: "Ops",
    eta: "32m",
  },
  {
    id: 2,
    moduleKey: "biomarker-matrices",
    module: "Biomarker Matrices",
    item: "Cohort stratification v2.4",
    stage: "Feature build",
    status: "Ingesting",
    owner: "Data",
    eta: "18m",
  },
  {
    id: 3,
    moduleKey: "biomarker-analysis",
    module: "Biomarker Analysis Agent",
    item: "Responder prediction draft",
    stage: "Agent loop",
    status: "Review",
    owner: "Science",
    eta: "needs approval",
  },
  {
    id: 4,
    moduleKey: "genomic-sequencing",
    module: "Genomic Sequencing",
    item: "WGS_2487 coverage QC",
    stage: "QC",
    status: "Attention",
    owner: "Genomics",
    eta: "blocked on controls",
  },
  {
    id: 5,
    moduleKey: "imaging-agent",
    module: "Imaging Agent",
    item: "Organoid plate 11 annotations",
    stage: "Inference",
    status: "Running",
    owner: "Imaging",
    eta: "41m",
  },
  {
    id: 6,
    moduleKey: "bioprocess-automation",
    module: "Bioprocess Automation",
    item: "Clean-in-place validation",
    stage: "Validation",
    status: "Done",
    owner: "QA",
    eta: "closed",
  },
];

export const upcomingActions = [
  {
    id: 1,
    title: "Sync biomarker matrices with latest sequencing batch",
    owner: "Data",
    moduleKey: "biomarker-matrices",
    color: "primary",
    date: { day: "07", month: "FEB", time: "09:00 AM" },
    participants: [
      { id: 1, user_name: "Data Eng", user_img: "/images/avatar/2.png" },
      { id: 2, user_name: "Genomics", user_img: "/images/avatar/3.png" },
    ],
  },
  {
    id: 2,
    title: "Review AI agent outputs for the oncology program",
    owner: "Science",
    moduleKey: "biomarker-analysis",
    color: "success",
    date: { day: "07", month: "FEB", time: "01:30 PM" },
    participants: [
      { id: 1, user_name: "Science Lead", user_img: "/images/avatar/4.png" },
      { id: 2, user_name: "ML", user_img: "/images/avatar/5.png" },
    ],
  },
  {
    id: 3,
    title: "Imaging agent deploy to lab robots",
    owner: "Imaging",
    moduleKey: "imaging-agent",
    color: "warning",
    date: { day: "08", month: "FEB", time: "10:15 AM" },
    participants: [
      { id: 1, user_name: "Robotics", user_img: "/images/avatar/6.png" },
      { id: 2, user_name: "Imaging", user_img: "/images/avatar/7.png" },
    ],
  },
  {
    id: 4,
    title: "Recalibrate bioprocess sensors before night run",
    owner: "Ops",
    moduleKey: "bioprocess-automation",
    color: "danger",
    date: { day: "08", month: "FEB", time: "06:30 PM" },
    participants: [
      { id: 1, user_name: "Ops Lead", user_img: "/images/avatar/8.png" },
      { id: 2, user_name: "QA", user_img: "/images/avatar/9.png" },
    ],
  },
  {
    id: 5,
    title: "Release coverage report for sequencing batch 2487",
    owner: "Genomics",
    moduleKey: "genomic-sequencing",
    color: "info",
    date: { day: "09", month: "FEB", time: "10:45 AM" },
    participants: [
      { id: 1, user_name: "Genomics", user_img: "/images/avatar/3.png" },
      { id: 2, user_name: "Data Eng", user_img: "/images/avatar/2.png" },
    ],
  },
];

export const moduleKpis = {
  "biomarker-analysis": {
    summary:
      "LLM-driven agent for biomarker hypothesis generation, report drafting, and triage of responders/non-responders.",
    stats: [
      { label: "Pipelines linked", value: "3 / 4" },
      { label: "Drafts pending review", value: "4" },
      { label: "Median turnaround", value: "22m" },
      { label: "Human-in-loop approvals", value: "2 open" },
    ],
    backlog: [
      "Wire new matrix features into the prompt context windows.",
      "Auto-publish responder narratives to the clinical wiki.",
      "Gate sequencing QC status before generating reports.",
    ],
    checkpoints: [
      "Context windows capped at 120k tokens to keep latency predictable.",
      "Guardrail templates validated against HIPAA redaction rules.",
      "Streaming mode enabled for long-form reports.",
    ],
  },
  "biomarker-matrices": {
    summary:
      "Manages cohort stratification, normalization matrices, and feature panels feeding downstream agents.",
    stats: [
      { label: "Feature sets", value: "18 active" },
      { label: "Panels refreshed", value: "6 today" },
      { label: "Data freshness", value: "< 45m" },
      { label: "QC gates", value: "92% pass" },
    ],
    backlog: [
      "Add sequencing batch 2487 to longitudinal cohort.",
      "Recompute immunology panel with updated ELISA curves.",
      "Publish matrix diffs to the Imaging Agent for overlays.",
    ],
    checkpoints: [
      "Backfills run with drift alerts enabled.",
      "S3 lineage manifests emitted for every panel refresh.",
      "Sampling rules pin cohort size to 500 for quick iteration.",
    ],
  },
  "bioprocess-automation": {
    summary:
      "Closed-loop automation for upstream and downstream processes with auditability and QC interlocks.",
    stats: [
      { label: "Runs active", value: "9" },
      { label: "Interlocks", value: "12 enforced" },
      { label: "Runtime alerts", value: "1 (temp drift)" },
      { label: "MTBF", value: "11.8d" },
    ],
    backlog: [
      "Tune PID loop for fed-batch nutrient feed.",
      "Add smoke test for CIP alarms before night shift.",
      "Expose batch metadata to Sequencing handoff.",
    ],
    checkpoints: [
      "All automation scripts versioned and signed.",
      "Audit log streaming to compliance bucket every 5m.",
      "Fallback recipes validated against last successful run.",
    ],
  },
  "genomic-sequencing": {
    summary:
      "Sequencer control, demultiplexing, coverage QC, and variant packaging for downstream analytics.",
    stats: [
      { label: "Lanes running", value: "3 / 4" },
      { label: "Coverage P90", value: "36x" },
      { label: "QC holds", value: "2" },
      { label: "VCF packages", value: "11 today" },
    ],
    backlog: [
      "Release new variant annotation build with ClinVar Jan data.",
      "Auto-pause low coverage lanes after 2 alerts.",
      "Ship coverage deltas into Matrices for cohort drift checks.",
    ],
    checkpoints: [
      "FASTQ checksummed and mirrored to cold storage.",
      "Adapters trimmed with validated settings per platform.",
      "Sample sheet validation pinned to release v0.12.",
    ],
  },
  "imaging-agent": {
    summary:
      "Vision agent for microscopy capture, annotation, and runbook drafting for lab automation.",
    stats: [
      { label: "Plates in queue", value: "6" },
      { label: "Avg. latency", value: "480ms/frame" },
      { label: "AI overrides", value: "1 today" },
      { label: "Overlay exports", value: "14" },
    ],
    backlog: [
      "Enable z-stack stitching for the new camera rig.",
      "Port latest Matrices to color maps for overlays.",
      "Push nightly benchmark against held-out organoid set.",
    ],
    checkpoints: [
      "Calibration checked every 12h with NIST slide.",
      "GPU memory pinned to avoid eviction during bursts.",
      "Runbook templates linted before publishing to robots.",
    ],
  },
};
