// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

const modalities = new Map([
  ["all", "All"], ["cloth", "Cloth"], ["cable", "Cable"], ["sand", "Sand"], ["coupled", "Coupled"],
]);
const el = (id) => document.getElementById(id);
const dateFormat = new Intl.DateTimeFormat("en", { dateStyle: "medium", timeZone: "UTC" });
const numberFormat = new Intl.NumberFormat("en", { maximumFractionDigits: 6 });
const filterButtons = [...document.querySelectorAll("button[data-modality]")];
const slug = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const imageFile = /^media\/(?:[\w-]+\/)*[\w.-]+\.(?:png|jpe?g|webp|gif)$/i;
const videoFile = /^media\/(?:[\w-]+\/)*[\w.-]+\.(?:mp4|webm)$/i;
const evidenceFile = /^evidence\/(?:[\w-]+\/)*[\w.-]+\.(?:json|csv|pdf|txt)$/i;
const fallbackRun = "elastic-six-second-v3";
const specialistCloth = "elastic-accuracy-speed-v3";
const specialistCable = "elastic-refined-cable-v4";

let modality = urlModality();
let manifest = null;
let payload = null;
let loading = false;
let featured = null;
let primaryComparisons = [];
const expanded = new Set();

function urlModality() {
  const value = new URLSearchParams(location.search).get("modality");
  return modalities.has(value) ? value : "all";
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (typeof text === "string") node.textContent = text;
  if (className) node.className = className;
  return node;
}

function dateElement(date, className) {
  if (typeof date !== "string" || !Number.isFinite(Date.parse(date))) return null;
  const stamp = element("time", dateFormat.format(new Date(date)), className);
  stamp.dateTime = date;
  return stamp;
}

function anchor(label, href) {
  const link = element("a", label);
  link.href = href;
  return link;
}

function sourceAnchor(source) {
  const url = typeof source === "string" ? source : source?.url;
  if (typeof url !== "string") return null;
  const label = source?.label || "Source";
  if (evidenceFile.test(url)) return anchor(label, `./${url}`);
  try {
    const target = new URL(url);
    if (target.protocol !== "https:" || target.username || target.password) return null;
    const link = anchor(label, target.href);
    link.rel = "noreferrer";
    return link;
  } catch {
    return null;
  }
}

function appendSources(container, sources) {
  const list = element("ul", null, "source-links");
  for (const source of sources) {
    const link = sourceAnchor(source);
    if (link) {
      const item = element("li");
      item.append(link);
      list.append(item);
    }
  }
  if (list.childElementCount) container.append(list);
}

function formatValue(value) {
  if (typeof value === "number") {
    return value !== 0 && Math.abs(value) < 1e-5 ? value.toExponential(2) : numberFormat.format(value);
  }
  return typeof value === "string" ? value : "Pending";
}

function metricText(metric) {
  const value = formatValue(metric.value);
  return metric.unit ? `${value}${metric.unit === "%" || metric.unit === "×" ? "" : " "}${metric.unit}` : value;
}

function appendMetrics(container, metrics) {
  const list = element("dl", null, "metrics");
  for (const metric of metrics) {
    if (typeof metric?.label !== "string") continue;
    const item = element("div");
    const definition = element("dd", metricText(metric));
    if (metric.detail) definition.append(element("span", metric.detail, "metric-context"));
    item.append(element("dt", metric.label), definition);
    list.append(item);
  }
  if (list.childElementCount) container.append(list);
}

function mediaItem(record) {
  if (typeof record?.src !== "string" || typeof record.label !== "string") return null;
  const figure = element("figure", null, "media-item");
  if (record.type === "video" && videoFile.test(record.src)) {
    const video = element("video");
    video.controls = true;
    video.playsInline = true;
    video.preload = "none";
    if (typeof record.poster === "string" && imageFile.test(record.poster)) video.poster = `./${record.poster}`;
    video.setAttribute("aria-label", record.label);
    const movie = element("source");
    movie.src = `./${record.src}`;
    movie.type = record.src.toLowerCase().endsWith(".webm") ? "video/webm" : "video/mp4";
    video.append(movie, anchor("Download rollout video", movie.src));
    figure.append(video);
  } else if (record.type === "image" && imageFile.test(record.src)) {
    const image = element("img");
    image.loading = "lazy";
    image.src = `./${record.src}`;
    image.alt = record.label;
    figure.append(image);
  } else {
    return null;
  }
  figure.append(element("figcaption", record.label));
  return figure;
}

function assetsFor(run) {
  const assets = element("div", null, "assets");
  for (const record of run.media ?? []) {
    const figure = mediaItem(record);
    if (figure) assets.append(figure);
  }
  return assets;
}

function evidenceFor(run) {
  const content = element("div", null, "run-content");
  content.append(element("p", run.scope, "scope"));
  if (run.summary) content.append(element("p", run.summary, "run-summary"));
  appendMetrics(content, run.metrics ?? []);
  appendSources(content, run.sources ?? []);
  return content;
}

function referenceCard(comparison) {
  const card = element("article", null, "reference-card");
  if (slug.test(comparison.id)) card.id = comparison.id;
  card.append(element("h4", comparison.label), element("p", comparison.track, "meta"));
  card.append(element("p", comparison.scope, "scope"));
  appendMetrics(card, comparison.values ?? []);
  appendSources(card, comparison.source ? [comparison.source] : []);
  return card;
}

function comparisonTable(comparisons) {
  const labels = comparisons[0]?.values.map((value) => value.label);
  if (!labels?.length || !comparisons.every((entry) => entry.values.length === labels.length
    && entry.values.every((value, index) => value.label === labels[index]))) {
    const cards = element("div", null, "reference-grid");
    cards.append(...comparisons.map(referenceCard));
    return cards;
  }
  const wrap = element("div", null, "table-wrap");
  const table = element("table", null, "comparison-table");
  table.append(element("caption", "Matched evaluation results", "sr-only"));
  const header = element("thead");
  const headings = element("tr");
  for (const label of ["Benchmark", ...labels]) {
    const column = element("th", label);
    column.scope = "col";
    headings.append(column);
  }
  header.append(headings);
  table.append(header);
  const body = element("tbody");
  for (const comparison of comparisons) {
    const row = element("tr");
    if (slug.test(comparison.id)) row.id = comparison.id;
    const cell = element("th", comparison.label);
    cell.scope = "row";
    cell.append(element("span", comparison.track, "meta"));
    row.append(cell);
    for (const metric of comparison.values) row.append(element("td", metricText(metric)));
    body.append(row);
  }
  table.append(body);
  wrap.append(table);
  return wrap;
}

function specialistResult(id, metricLabel, label, context) {
  const run = manifest.runs.find((entry) => entry.id === id);
  const metric = run?.metrics?.find((entry) => entry.label === metricLabel);
  if (!metric) return null;
  const item = element("li");
  const link = anchor(label, `#${id}`);
  item.append(link, element("strong", metricText(metric)), element("span", context, "meta"));
  return item;
}

function renderCurrent() {
  featured = manifest.runs.find((run) => run.id.startsWith("elastic-shared-"))
    ?? manifest.runs.find((run) => run.id === fallbackRun)
    ?? manifest.runs[0] ?? null;
  primaryComparisons = featured?.id === fallbackRun
    ? manifest.comparisons.filter((entry) => /^elastic-six-second-(cloth|cable)-v3$/.test(entry.id))
    : manifest.comparisons.filter((entry) => entry.id.startsWith(`${featured?.id}-`));

  const container = el("runs");
  container.replaceChildren();
  if (!featured) {
    container.append(element("p", "Shared benchmarks will appear here when available.", "empty"));
    return;
  }

  const comparison = element("article", null, "current-result");
  comparison.id = featured.id;
  comparison.append(element("h3", featured.title));
  const evidence = evidenceFor(featured);
  if (primaryComparisons.length) {
    const results = element("section", null, "matched-results");
    results.id = "comparisons";
    results.append(element("h4", "Matched comparison", "sr-only"));
    if (primaryComparisons.every((entry) => /position error|stability/i.test(entry.label)
      && entry.values.every((value) => value.unit === "%"))) {
      results.append(element("p", "Mean position error as a percent of rest length", "comparison-unit"));
    }
    results.append(comparisonTable(primaryComparisons));
    const detail = primaryComparisons.map((entry) => entry.scope);
    if (detail.every((scope) => scope === detail[0])) results.append(element("p", detail[0], "scope"));
    comparison.append(results);
  } else {
    evidence.id = "comparisons";
  }
  comparison.append(evidence);
  const assets = assetsFor(featured);
  if (assets.childElementCount) comparison.append(assets);
  container.append(comparison);

  const specialist = element("aside", null, "specialist-reference");
  const results = element("ul", null, "specialist-scores");
  const benchmarks = [
    specialistResult(specialistCloth, "Cloth mean position error", "Cloth specialist", "Pinned Newton reference"),
    specialistResult(specialistCable, "Mean normalized position error", "Cable specialist", "Finer-step Newton reference"),
  ];
  for (const result of benchmarks) if (result) results.append(result);
  if (results.childElementCount) {
    specialist.append(element("h3", "Prescribed-load benchmarks"), results,
      element("p", "Separate 1.8-second experiments with full schedules and their own reference settings.", "scope"));
    container.append(specialist);
  }

  const rolloutPoster = (run) => run?.media?.find((item) => typeof item?.poster === "string" && imageFile.test(item.poster));
  const previewRun = rolloutPoster(featured) ? featured : manifest.runs.find((run) => run.id === fallbackRun);
  const preview = rolloutPoster(previewRun);
  if (preview) {
    el("preview-link").href = `#${previewRun.id}`;
    el("preview-title").textContent = previewRun.title;
    el("preview-image").src = `./${preview.poster}`;
    el("preview-image").alt = preview.label;
  }
}

function archiveEntry(run) {
  const entry = element("details", null, "archive-entry");
  entry.id = run.id;
  const summary = element("summary");
  summary.append(element("span", run.title, "archive-run-title"));
  const date = dateElement(run.date, "run-date");
  if (date) summary.append(date);
  entry.append(summary);
  const expand = () => {
    if (entry.open) {
      expanded.add(run.id);
      if (entry.childElementCount === 1) {
        const content = evidenceFor(run);
        const labels = run.modalities.map((name) => modalities.get(name)).filter(Boolean);
        const info = element("p", [run.status, ...labels].join(" · "), "meta");
        content.prepend(info);
        entry.append(content);
        const assets = assetsFor(run);
        if (assets.childElementCount) entry.append(assets);
      }
    } else {
      expanded.delete(run.id);
    }
  };
  entry.addEventListener("toggle", expand);
  if (expanded.has(run.id)) {
    entry.open = true;
    expand();
  }
  return entry;
}

function renderArchive() {
  for (const button of filterButtons) button.setAttribute("aria-pressed", String(button.dataset.modality === modality));
  if (!manifest) return;
  const all = manifest.runs.filter((run) => run !== featured);
  const needle = el("archive-search").value.trim().toLowerCase();
  const runs = all.filter((run) => (modality === "all" || run.modalities.includes(modality))
    && [run.id, run.title, run.scope, run.status].some((value) => String(value).toLowerCase().includes(needle)));
  el("archive-count").textContent = `${all.length} runs`;
  const container = el("archive-runs");
  container.replaceChildren(...runs.map(archiveEntry));
  if (!runs.length) container.append(element("p", "No runs match these filters.", "empty"));
  el("filter-status").textContent = `${runs.length} archived ${runs.length === 1 ? "run" : "runs"} shown.`;
}

function renderReferences() {
  const older = manifest?.comparisons.filter((comparison) => !primaryComparisons.includes(comparison)) ?? [];
  el("reference-count").textContent = `${older.length} comparisons`;
  if (el("archive-comparisons").open) {
    el("reference-grid").replaceChildren(...older.map(referenceCard));
  }
}

function setModality(next) {
  modality = next;
  const url = new URL(location.href);
  if (modality === "all") url.searchParams.delete("modality");
  else url.searchParams.set("modality", modality);
  history.replaceState(null, "", url);
  renderArchive();
}

function revealResult(id = location.hash.slice(1)) {
  if (!slug.test(id) || !manifest) return;
  const archive = el("archive-panel");
  if (id === "archive") {
    archive.open = true;
    return;
  }
  const run = manifest.runs.find((entry) => entry.id === id);
  const comparison = manifest.comparisons.find((entry) => entry.id === id);
  if (run && run !== featured) {
    archive.open = true;
    if (!el(id)) {
      el("archive-search").value = "";
      if (modality === "all" || run.modalities.includes(modality)) renderArchive();
      else setModality("all");
    }
    const entry = el(id);
    if (entry) entry.open = true;
  } else if (id === "archive-comparisons" || (comparison && !primaryComparisons.includes(comparison))) {
    archive.open = true;
    el("archive-comparisons").open = true;
    renderReferences();
  }
  if (run || id === "archive-comparisons" || comparison) {
    requestAnimationFrame(() => el(id)?.scrollIntoView());
  }
}

function validate(data) {
  if (data?.schema !== 1 || typeof data.headline !== "string" || !Array.isArray(data.runs) || !Array.isArray(data.comparisons)) {
    throw new Error("Unsupported benchmark manifest");
  }
  for (const run of data.runs) {
    if (!run || typeof run.id !== "string" || !slug.test(run.id) || typeof run.title !== "string" || !Array.isArray(run.modalities)) {
      throw new Error("Invalid rollout entry");
    }
    for (const key of ["metrics", "media", "sources"]) {
      if (run[key] !== undefined && !Array.isArray(run[key])) throw new Error(`Invalid ${key}`);
    }
  }
  for (const comparison of data.comparisons) {
    if (!comparison || typeof comparison.label !== "string" || !Array.isArray(comparison.values)) {
      throw new Error("Invalid reference entry");
    }
  }
  return data;
}

async function refreshReport(manual = false) {
  if (loading || (!manual && document.hidden)) return;
  loading = true;
  el("refresh").disabled = true;
  try {
    const response = await fetch("./report.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Report request failed (${response.status})`);
    const body = await response.text();
    if (body !== payload) {
      const first = manifest === null;
      manifest = validate(JSON.parse(body));
      payload = body;
      const stamp = dateElement(manifest.updated);
      el("updated").hidden = stamp === null;
      if (stamp) {
        el("updated").dateTime = stamp.dateTime;
        el("updated").textContent = `Updated ${stamp.textContent}`;
      }
      renderCurrent();
      renderArchive();
      renderReferences();
      if (first) revealResult();
      el("network-status").textContent = "Latest benchmark results loaded.";
    } else if (manual) {
      el("network-status").textContent = "The benchmark report is up to date.";
    }
  } catch {
    const description = "Could not load the benchmark report. Use Refresh to retry.";
    if (!manifest) el("runs").replaceChildren(element("p", description, "empty"));
    el("network-status").textContent = description;
  } finally {
    el("refresh").disabled = false;
    el("runs").setAttribute("aria-busy", "false");
    loading = false;
  }
}

for (const button of filterButtons) button.addEventListener("click", () => setModality(button.dataset.modality));
el("archive-search").addEventListener("input", renderArchive);
el("archive-comparisons").addEventListener("toggle", () => {
  if (el("archive-comparisons").open) renderReferences();
});
el("refresh").addEventListener("click", () => refreshReport(true));
document.addEventListener("click", (event) => {
  const link = event.target.closest("a[href^='#']");
  if (link?.hash === location.hash) revealResult();
});
window.addEventListener("hashchange", () => revealResult());
window.addEventListener("popstate", () => {
  modality = urlModality();
  renderArchive();
  revealResult();
});
document.addEventListener("visibilitychange", () => refreshReport());
renderArchive();
void refreshReport(true);
window.setInterval(refreshReport, 60_000);
