// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

const modalities = new Map([
  ["all", "All"], ["sand", "Sand"], ["cloth", "Cloth"], ["cable", "Cable"], ["coupled", "Coupled"],
]);
const el = (id) => document.getElementById(id);
const dateFormat = new Intl.DateTimeFormat("en", { dateStyle: "medium", timeZone: "UTC" });
const numberFormat = new Intl.NumberFormat("en", { maximumFractionDigits: 6 });
const filterButtons = [...document.querySelectorAll("button[data-modality]")];
const slug = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const imageFile = /^media\/(?:[\w-]+\/)*[\w.-]+\.(?:png|jpe?g|webp|gif)$/i;
const videoFile = /^media\/(?:[\w-]+\/)*[\w.-]+\.(?:mp4|webm)$/i;
const evidenceFile = /^evidence\/(?:[\w-]+\/)*[\w.-]+\.(?:json|csv|pdf|txt)$/i;
const selected = new URLSearchParams(location.search).get("modality");
let modality = modalities.has(selected) ? selected : "all";
let manifest = null;
let payload = null;
let loading = false;

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

function sourceAnchor(source) {
  const url = typeof source === "string" ? source : source?.url;
  if (typeof url !== "string") return null;
  const anchor = element("a", source?.label || "Source");
  if (evidenceFile.test(url)) {
    anchor.href = `./${url}`;
    return anchor;
  }
  try {
    const target = new URL(url);
    if (target.protocol !== "https:" || target.username || target.password) return null;
    anchor.href = target.href;
    anchor.rel = "noreferrer";
    return anchor;
  } catch {
    return null;
  }
}

function appendSources(container, sources) {
  const list = element("ul", null, "source-links");
  for (const source of sources) {
    const anchor = sourceAnchor(source);
    if (anchor) {
      const item = element("li");
      item.append(anchor);
      list.append(item);
    }
  }
  if (list.childElementCount) container.append(list);
}

function appendMetrics(container, metrics) {
  const list = element("dl", null, "metrics");
  for (const metric of metrics) {
    if (typeof metric?.label !== "string") continue;
    const item = element("div");
    let value = metric.value;
    if (typeof value === "number") {
      value = value !== 0 && Math.abs(value) < 1e-5 ? value.toExponential(2) : numberFormat.format(value);
    }
    const definition = element("dd", typeof value === "string" ? value : "Pending");
    if (metric.unit) definition.append(element("span", metric.unit, "metric-unit"));
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
    const download = element("a", "Download rollout video");
    download.href = movie.src;
    video.append(movie, download);
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

function runCard(run) {
  const article = element("article", null, "run-card");
  if (slug.test(run.id)) article.id = run.id;
  const heading = element("div", null, "run-content");
  const labels = element("div", null, "label-row");
  labels.append(element("span", run.status, "tag status-tag"));
  for (const type of run.modalities) {
    if (modalities.has(type) && type !== "all") labels.append(element("span", modalities.get(type), "tag"));
  }
  const date = dateElement(run.date, "run-date");
  if (date) labels.append(date);
  heading.append(labels, element("h3", run.title), element("p", run.scope, "scope"));
  article.append(heading);

  const assets = element("div", null, "assets");
  for (const record of run.media ?? []) {
    const figure = mediaItem(record);
    if (figure) assets.append(figure);
  }
  if (assets.childElementCount) article.append(assets);

  const details = element("div", null, "run-content");
  if (run.summary) details.append(element("p", run.summary, "run-summary"));
  appendMetrics(details, run.metrics ?? []);
  appendSources(details, run.sources ?? []);
  if (details.childElementCount) article.append(details);
  return article;
}

function renderRuns() {
  for (const button of filterButtons) button.setAttribute("aria-pressed", String(button.dataset.modality === modality));
  if (!manifest) return;

  const runs = manifest.runs.filter((run) => modality === "all" || run.modalities.includes(modality));
  const container = el("runs");
  container.replaceChildren(...runs.map(runCard));
  if (runs.length === 0) {
    const area = modality === "all" ? "this benchmark" : modalities.get(modality).toLowerCase();
    container.append(element("p", `Results for ${area} will appear here when a run is added.`, "empty"));
  }
  el("filter-status").textContent = `${runs.length} rollout ${runs.length === 1 ? "result" : "results"} shown.`;
}

function renderReferences() {
  const cards = manifest.comparisons.map((comparison) => {
    const card = element("article", null, "reference-card");
    card.append(element("p", comparison.track, "kicker"), element("h3", comparison.label));
    card.append(element("p", comparison.scope, "scope"));
    appendMetrics(card, comparison.values ?? []);
    appendSources(card, comparison.source ? [comparison.source] : []);
    return card;
  });
  el("reference-grid").replaceChildren(...cards);
}

function setModality(next) {
  modality = next;
  const url = new URL(location.href);
  if (modality === "all") url.searchParams.delete("modality");
  else url.searchParams.set("modality", modality);
  history.replaceState(null, "", url);
  renderRuns();
}

function revealRun() {
  const id = location.hash.slice(1);
  if (!slug.test(id) || !manifest?.runs.some((run) => run.id === id)) return;
  if (!el(id)) setModality("all");
  el(id)?.scrollIntoView();
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
      el("headline").textContent = manifest.headline;
      const stamp = dateElement(manifest.updated);
      el("updated").hidden = stamp === null;
      if (stamp) {
        el("updated").dateTime = stamp.dateTime;
        el("updated").textContent = stamp.textContent;
      }
      renderRuns();
      renderReferences();
      if (first) revealRun();
      el("network-status").textContent = "Latest benchmark results loaded.";
    } else if (manual) {
      el("network-status").textContent = "The benchmark report is up to date.";
    }
  } catch {
    const description = "Could not load the benchmark report. Use Refresh to retry.";
    if (!manifest) {
      el("headline").textContent = description;
      el("runs").replaceChildren(element("p", description, "empty"));
    }
    el("network-status").textContent = description;
  } finally {
    el("refresh").disabled = false;
    el("runs").setAttribute("aria-busy", "false");
    loading = false;
  }
}

for (const button of filterButtons) button.addEventListener("click", () => setModality(button.dataset.modality));
el("refresh").addEventListener("click", () => refreshReport(true));
document.querySelector(".hero-figure figcaption a").addEventListener("click", () => {
  if (manifest?.runs.some((run) => run.id === "sphere-support") && !el("sphere-support")) setModality("all");
});
window.addEventListener("hashchange", revealRun);
window.addEventListener("popstate", () => {
  const next = new URLSearchParams(location.search).get("modality");
  modality = modalities.has(next) ? next : "all";
  renderRuns();
  revealRun();
});
document.addEventListener("visibilitychange", () => refreshReport());
renderRuns();
void refreshReport(true);
window.setInterval(refreshReport, 60_000);
