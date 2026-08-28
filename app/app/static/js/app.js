const analyzeBtn = document.getElementById("analyze-btn");
const reportBtn = document.getElementById("report-btn");
const statusEl = document.getElementById("status");
const textEl = document.getElementById("news-text");

let lastResult = null;

// ── Theme toggle (dark / light, persisted) ───────────────────────────────
function currentTheme() {
  const stored = localStorage.getItem("sfnd-theme");
  if (stored === "dark" || stored === "light") return stored;
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function updateThemeIcon() {
  const icon = document.getElementById("theme-icon");
  if (icon) icon.textContent = currentTheme() === "dark" ? "dark_mode" : "light_mode";
}

const themeToggle = document.getElementById("theme-toggle");
if (themeToggle) {
  themeToggle.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    localStorage.setItem("sfnd-theme", next);
    document.documentElement.dataset.theme = next;
    updateThemeIcon();
  });
}
updateThemeIcon();

analyzeBtn.addEventListener("click", analyze);
reportBtn.addEventListener("click", downloadReport);

function showStatus(msg, kind) {
  statusEl.textContent = msg || "";
  statusEl.className = "status" + (kind ? " " + kind : "");
}

async function analyze() {
  const text = textEl.value.trim();
  if (!text) {
    showStatus("Please paste some text first.", "error");
    return;
  }

  analyzeBtn.disabled = true;
  reportBtn.disabled = true;
  hideResult();
  showStatus("Analyzing…", "info");

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    lastResult = data;
    renderResult(data);
    showStatus("", "");
  } catch (err) {
    showStatus(err.message, "error");
  } finally {
    analyzeBtn.disabled = false;
  }
}

function hideResult() {
  document.getElementById("result").classList.add("hidden");
}

function renderResult(data) {
  const resultEl = document.getElementById("result");
  resultEl.classList.remove("hidden");

  const badge = document.getElementById("verdict-badge");
  badge.className = "badge " + data.label;
  badge.innerHTML = "";
  const badgeIcon = document.createElement("span");
  badgeIcon.className = "material-symbols-outlined";
  badgeIcon.textContent = data.label === "REAL" ? "verified" : "error";
  badge.appendChild(badgeIcon);
  badge.appendChild(document.createTextNode(" " + data.label));

  const pct = data.confidence * 100;
  const fill = document.getElementById("confidence-fill");
  fill.style.width = pct + "%";
  fill.className = "fill " + data.label;
  document.getElementById("confidence-text").textContent =
    `${pct.toFixed(1)}%`;

  document.getElementById("prob-real").textContent = (data.probability_real * 100).toFixed(2) + "%";
  document.getElementById("prob-fake").textContent = (data.probability_fake * 100).toFixed(2) + "%";
  document.getElementById("article-meta").textContent =
    `ID ${data.article_id} · ${data.timestamp}`;

  // Penalties (FAKE) vs. real note
  const penaltiesEl = document.getElementById("penalties");
  const realNote = document.getElementById("real-note");
  if (data.label === "FAKE") {
    const primaryBody = document.getElementById("penalties-primary");
    const moreBody = document.getElementById("penalties-more-body");
    const moreDetails = document.getElementById("penalties-more");
    const matchNote = document.getElementById("match-note");
    primaryBody.innerHTML = "";
    moreBody.innerHTML = "";

    const primary = data.penalties.find(p => p.primary);
    const rest = data.penalties.filter(p => !p.primary);

    if (primary) {
      primaryBody.appendChild(buildPenaltyRow(primary, true));
      for (const p of rest) moreBody.appendChild(buildPenaltyRow(p, false));
      moreDetails.classList.remove("hidden");
    } else {
      for (const p of data.penalties) primaryBody.appendChild(buildPenaltyRow(p, false));
      moreDetails.classList.add("hidden");
    }

    const matched = [...new Set(data.penalties.flatMap(p => p.matched_keywords || []))];
    matchNote.textContent = matched.length
      ? "Matched keywords: " + matched.join(" · ")
      : "No specific section keywords detected — all sections listed for reference.";

    penaltiesEl.classList.remove("hidden");
    realNote.classList.add("hidden");
  } else {
    penaltiesEl.classList.add("hidden");
    realNote.classList.remove("hidden");
  }

  reportBtn.disabled = false;
}

function buildPenaltyRow(p, isPrimary) {
  const tr = document.createElement("tr");
  if (isPrimary) tr.className = "primary";
  const tdSec = document.createElement("td");
  tdSec.className = "sec";
  tdSec.textContent = p.section;
  if (isPrimary) {
    const badge = document.createElement("span");
    badge.className = "badge-small";
    badge.textContent = "PRIMARY";
    tdSec.appendChild(badge);
  }
  const tdTitle = document.createElement("td");
  tdTitle.textContent = p.title;
  const tdPen = document.createElement("td");
  tdPen.textContent = p.penalty;
  tr.append(tdSec, tdTitle, tdPen);
  return tr;
}

async function downloadReport() {
  const text = textEl.value.trim();
  if (!text || !lastResult) return;

  reportBtn.disabled = true;
  showStatus("Generating PDF report…", "info");
  try {
    const res = await fetch("/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, article_id: lastResult.article_id }),
    });
    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.error || "Report generation failed");
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `penalty_notice_${lastResult.article_id}.pdf`;
    a.click();
    URL.revokeObjectURL(url);
    showStatus("PDF report downloaded.", "");
  } catch (err) {
    showStatus(err.message, "error");
  } finally {
    reportBtn.disabled = false;
  }
}
