/**
 * Client-Side Application Logic for KV-Cache Compression Interface.
 */

let benchmarkData = null;
let presetsList = [];
let currentPreset = null;

document.addEventListener("DOMContentLoaded", () => {
  loadPresets();
  loadBenchmarks();
  updateParamBadges();
});

function switchMode(mode) {
  document.getElementById("interactive-mode").classList.remove("active");
  document.getElementById("dashboard-mode").classList.remove("active");
  document.getElementById("tab-interactive").classList.remove("active");
  document.getElementById("tab-dashboard").classList.remove("active");

  if (mode === "interactive") {
    document.getElementById("interactive-mode").classList.add("active");
    document.getElementById("tab-interactive").classList.add("active");
  } else {
    document.getElementById("dashboard-mode").classList.add("active");
    document.getElementById("tab-dashboard").classList.add("active");
    renderDashboardTable("phase8");
  }
}

function onMethodChange() {
  const method = document.getElementById("method-select").value;
  const sinksGroup = document.getElementById("group-sinks");
  const windowGroup = document.getElementById("group-window");
  const chunkGroup = document.getElementById("group-chunk");
  const budgetInput = document.getElementById("budget-input");

  if (method === "full_baseline") {
    sinksGroup.style.display = "none";
    windowGroup.style.display = "none";
    chunkGroup.style.display = "none";
    budgetInput.disabled = true;
    budgetInput.value = "Full (B=L)";
    document.getElementById("budget-mib").textContent = "Unconstrained";
  } else if (method === "sliding_window") {
    sinksGroup.style.display = "none";
    windowGroup.style.display = "block";
    chunkGroup.style.display = "none";
    budgetInput.disabled = false;
    budgetInput.value = "260";
    document.getElementById("window-input").value = "260";
    updateParamBadges();
  } else if (method === "h2o") {
    sinksGroup.style.display = "none";
    windowGroup.style.display = "block";
    chunkGroup.style.display = "none";
    budgetInput.disabled = false;
    budgetInput.value = "260";
    document.getElementById("window-input").value = "196";
    updateParamBadges();
  } else if (method === "ogc_cache") {
    sinksGroup.style.display = "block";
    windowGroup.style.display = "block";
    chunkGroup.style.display = "block";
    budgetInput.disabled = false;
    budgetInput.value = "260";
    document.getElementById("sinks-input").value = "4";
    document.getElementById("window-input").value = "192";
    document.getElementById("chunk-input").value = "64";
    updateParamBadges();
  } else {
    // StreamingLLM
    sinksGroup.style.display = "block";
    windowGroup.style.display = "block";
    chunkGroup.style.display = "none";
    budgetInput.disabled = false;
    budgetInput.value = "260";
    document.getElementById("sinks-input").value = "4";
    document.getElementById("window-input").value = "256";
    updateParamBadges();
  }
}

function updateParamBadges() {
  const budget = parseInt(document.getElementById("budget-input").value) || 260;
  // bytes per token = 2 * 28 * 8 * 128 * 2 = 114688 bytes = 0.02734 MiB
  const mib = (budget * 114688) / (1024 * 1024);
  document.getElementById("budget-mib").textContent = `${mib.toFixed(2)} MiB`;
}

async function loadPresets() {
  try {
    const res = await fetch("/api/presets");
    if (res.ok) {
      presetsList = await res.json();
      const select = document.getElementById("preset-select");
      select.innerHTML = '<option value="" disabled selected>Select long-context preset...</option>';
      presetsList.forEach((p, idx) => {
        const opt = document.createElement("option");
        opt.value = idx;
        opt.textContent = `[${p.target_context_length} tok | ${p.depth_category}] ${p.item_id}`;
        select.appendChild(opt);
      });
    }
  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

function loadSelectedPreset() {
  const select = document.getElementById("preset-select");
  const idx = select.value;
  if (idx === "" || !presetsList[idx]) return;

  currentPreset = presetsList[idx];
  document.getElementById("document-input").value = currentPreset.document;
  document.getElementById("question-input").value = currentPreset.question;
  document.getElementById("doc-token-badge").textContent = `Tokens: ~${currentPreset.actual_token_length}`;

  const gtBox = document.getElementById("ground-truth-box");
  gtBox.classList.remove("hidden");
  document.getElementById("gt-value").textContent = currentPreset.ground_truth_answers.join(", ");
  document.getElementById("match-badge").textContent = "Preset Loaded";
  document.getElementById("match-badge").className = "badge badge-info";
}

function clearInputs() {
  document.getElementById("document-input").value = "";
  document.getElementById("question-input").value = "";
  document.getElementById("doc-token-badge").textContent = "Tokens: 0";
  document.getElementById("output-box").innerHTML = '<div class="placeholder-text">Awaiting inference run. Output will appear here...</div>';
  document.getElementById("ground-truth-box").classList.add("hidden");
  currentPreset = null;
}

async function executeInference() {
  const doc = document.getElementById("document-input").value.trim();
  const question = document.getElementById("question-input").value.trim();
  const method = document.getElementById("method-select").value;
  const budget = parseInt(document.getElementById("budget-input").value) || 260;
  const sinks = parseInt(document.getElementById("sinks-input").value) || 4;
  const windowTokens = parseInt(document.getElementById("window-input").value) || 192;
  const chunkSize = parseInt(document.getElementById("chunk-input").value) || 64;

  if (!doc) {
    alert("Please enter a context document or load a preset first.");
    return;
  }

  const runBtn = document.getElementById("run-btn");
  const statusBadge = document.getElementById("exec-status");
  const outputBox = document.getElementById("output-box");

  runBtn.disabled = true;
  runBtn.innerHTML = '<span class="btn-icon">⏳</span> Computing Inference...';
  statusBadge.textContent = "Running MPS Inference";
  statusBadge.className = "badge badge-warning";
  outputBox.innerHTML = '<div class="placeholder-text">Executing prompt prefill and KV-cache compression on Apple Silicon MPS...</div>';

  const payload = {
    document: doc,
    question: question,
    method: method,
    budget: budget,
    sinks: sinks,
    window: windowTokens,
    chunk_size: chunkSize,
  };

  try {
    const startTime = performance.now();
    const res = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const elapsedTotal = performance.now() - startTime;

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.error || "Inference failed on server.");
    }

    const data = await res.json();
    outputBox.textContent = data.generated_text || "[Empty response]";

    // Update Ground Truth Verification if available
    if (currentPreset && currentPreset.ground_truth_answers) {
      const gts = currentPreset.ground_truth_answers;
      const isCorrect = gts.every(gt => data.generated_text.toLowerCase().includes(gt.toLowerCase()));
      const gtBadge = document.getElementById("match-badge");
      gtBadge.textContent = isCorrect ? "PASSED (Matches GT)" : "FAILED (Factual Mismatch)";
      gtBadge.className = isCorrect ? "badge badge-success" : "badge badge-warning";
    }

    // Update KPIs
    const kvMemMib = (data.compressed_kv_bytes / (1024 * 1024)).toFixed(2);
    const uncompMib = (data.uncompressed_kv_bytes / (1024 * 1024)).toFixed(2);
    const savingsPct = data.kv_cache_savings_pct.toFixed(1);

    document.getElementById("kpi-kv-mem").textContent = `${kvMemMib} MiB`;
    document.getElementById("kpi-kv-sub").textContent = `Full Cache: ${uncompMib} MiB`;

    document.getElementById("kpi-savings").textContent = `${savingsPct}%`;
    document.getElementById("kpi-savings-sub").textContent = `${(data.uncompressed_kv_bytes - data.compressed_kv_bytes) / 1024} KiB freed`;

    document.getElementById("kpi-prefill").textContent = `${data.prefill_time_ms.toFixed(1)} ms`;
    document.getElementById("kpi-prefill-sub").textContent = `${(data.input_tokens / (data.prefill_time_ms / 1000)).toFixed(0)} prompt tok/s`;

    document.getElementById("kpi-decode").textContent = `${data.decode_tokens_per_sec.toFixed(1)} tok/s`;
    document.getElementById("kpi-decode-sub").textContent = `${data.output_tokens_generated} tokens generated`;

    // Update Progress Bar
    const retainedRatio = data.uncompressed_kv_bytes > 0 ? (data.compressed_kv_bytes / data.uncompressed_kv_bytes) * 100 : 100;
    const clampedRatio = Math.max(3, Math.min(100, retainedRatio));
    document.getElementById("memory-progress-bar").style.width = `${clampedRatio}%`;
    document.getElementById("bar-percentage").textContent = `${retainedRatio.toFixed(1)}% of baseline`;

    // Update Telemetry Table
    document.getElementById("t-input-tokens").textContent = data.input_tokens.toLocaleString();
    document.getElementById("t-retained-tokens").textContent = data.retained_tokens.toLocaleString();
    document.getElementById("t-actual-bytes").textContent = `${data.compressed_kv_bytes.toLocaleString()} bytes (${kvMemMib} MiB)`;
    document.getElementById("t-formula-bytes").textContent = `${data.theoretical_kv_bytes.toLocaleString()} bytes`;
    document.getElementById("t-peak-alloc").textContent = `${(data.peak_device_allocated_mib || 0).toFixed(1)} MiB`;
    document.getElementById("t-host-rss").textContent = `${(data.host_rss_mib || 0).toFixed(1)} MiB`;

    statusBadge.textContent = "Completed Successfully";
    statusBadge.className = "badge badge-success";
  } catch (err) {
    outputBox.innerHTML = `<span style="color: var(--accent-red)">Error: ${err.message}</span>`;
    statusBadge.textContent = "Inference Error";
    statusBadge.className = "badge badge-warning";
  } finally {
    runBtn.disabled = false;
    runBtn.innerHTML = '<span class="btn-icon">⚡</span> Run Compressed Inference';
  }
}

async function loadBenchmarks() {
  try {
    const res = await fetch("/api/benchmarks");
    if (res.ok) {
      benchmarkData = await res.json();
      renderDashboardTable("phase8");
    }
  } catch (err) {
    console.error("Failed to load benchmark results:", err);
  }
}

function switchDashboardTab(phase) {
  document.querySelectorAll(".dashboard-filter-tabs .filter-btn").forEach(btn => btn.classList.remove("active"));
  event.target.classList.add("active");
  renderDashboardTable(phase);
}

function renderDashboardTable(phase) {
  if (!benchmarkData) return;
  const tbody = document.getElementById("benchmark-tbody");
  tbody.innerHTML = "";

  const dataSrc = benchmarkData[phase] || benchmarkData["phase8"];
  if (!dataSrc) return;

  const baselineAcc = dataSrc["Full Baseline"] ? dataSrc["Full Baseline"].overall_accuracy_pct : 100.0;

  for (const [name, info] of Object.entries(dataSrc)) {
    const tr = document.createElement("tr");

    let mechanism = "Baseline";
    if (name.includes("StreamingLLM")) mechanism = "Attention Sinks + Window";
    else if (name.includes("Sliding Window")) mechanism = "Pure Positional Eviction";
    else if (name.includes("H2O")) mechanism = "Global Causal Attention Oracle";
    else if (name.includes("OGC-Cache")) mechanism = "Query Observation + 1D Chunk";

    const acc = info.overall_accuracy_pct.toFixed(2);
    const diff = (info.overall_accuracy_pct - baselineAcc).toFixed(2);
    const diffStr = name === "Full Baseline" ? "Reference" : `${diff > 0 ? "+" : ""}${diff} pp`;
    const savings = info.overall_kv_savings_pct.toFixed(2);

    let kv4096 = "112.9 MiB";
    if (name !== "Full Baseline") kv4096 = "7.11 MiB";

    const prefill = info.avg_prefill_ms.toFixed(1);
    const decode = info.avg_decode_tok_s.toFixed(1);

    tr.innerHTML = `
      <td><strong>${name}</strong></td>
      <td>${mechanism}</td>
      <td style="color: ${info.overall_accuracy_pct >= 40 ? 'var(--accent-green)' : 'var(--text-primary)'}">${acc}%</td>
      <td style="color: ${diffStr === 'Reference' ? 'var(--accent-blue)' : (diff >= -60 ? 'var(--accent-green)' : 'var(--accent-orange)')}">${diffStr}</td>
      <td style="color: var(--accent-green)">${savings}%</td>
      <td>${kv4096}</td>
      <td>${prefill} ms</td>
      <td>${decode} tok/s</td>
    `;
    tbody.appendChild(tr);
  }
}
