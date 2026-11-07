// Helpers shared by the pages (game, replay, admin). Every text coming from the
// server (pseudos, names) is inserted with textContent, never as HTML.
"use strict";

// Options of the pages: those of the Flask server by default; the browser demo
// (web_demo/) sets window.SKYJO before loading this file.
const SKYJO = {
  replayUrl: (id) => `/replay/${encodeURIComponent(id)}`,
  pseudoOptional: false, // true: no pseudo needed, the record covers every game of this browser
  transport: null, // (method, url, body) -> data, instead of HTTP (the demo: a Web Worker)
  ...(window.SKYJO || {}),
};

const $ = (sel) => document.querySelector(sel);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

/** The browser's storage, or null when it is refused (blocked site data: even
 * reading window.localStorage then throws). */
function browserStorage(name) {
  try {
    const storage = window[name];
    const probe = "skyjo-probe";
    storage.setItem(probe, "1");
    storage.removeItem(probe);
    return storage;
  } catch {
    return null;
  }
}
const LOCAL = browserStorage("localStorage");
const SESSION = browserStorage("sessionStorage");

/** Read (value undefined), write or remove (null) a key of LOCAL or SESSION; null storage: nothing kept. */
function store(storage, key, value) {
  if (!storage) return null;
  try {
    if (value === undefined) return storage.getItem(key);
    if (value === null) storage.removeItem(key);
    else storage.setItem(key, value);
  } catch {
    return null;
  }
  return null;
}

const fmt = (x, digits = 1) => x.toLocaleString("fr-FR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const signed = (x) => (x > 0 ? "+" : "") + fmt(x);
const pct = (x) => `${Math.round(x * 100)} %`;
const dateText = (iso) => new Date(iso).toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "short" });

async function api(method, url, body) {
  if (SKYJO.transport) return SKYJO.transport(method, url, body);
  const options = { method };
  if (body !== undefined) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(body);
  }
  const response = await fetch(url, options);
  let data;
  try {
    data = await response.json();
  } catch {
    data = { error: `Réponse invalide du serveur (${response.status}).` };
  }
  if (!response.ok) throw new Error(data.error || `Erreur ${response.status}.`);
  return data;
}

function valueClass(v) {
  if (v < 0) return "v-neg";
  if (v === 0) return "v-zero";
  if (v <= 4) return "v-low";
  if (v <= 8) return "v-mid";
  return "v-high";
}

/** A card: back if hidden (no value), empty slot if removed. ``revealed``: hidden card shown with its value (replay). */
function cardNode(cell, asButton = false, revealed = false) {
  const node = el(asButton ? "button" : "div", "card");
  if (asButton) node.type = "button";
  if (cell.state === "removed") {
    node.classList.add("removed");
    node.setAttribute("aria-label", "colonne retirée");
  } else if (cell.state === "hidden" && !(revealed && cell.value !== null)) {
    node.classList.add("back");
    node.setAttribute("aria-label", "carte cachée");
  } else {
    node.classList.add(valueClass(cell.value));
    node.textContent = cell.value;
    if (cell.state === "hidden") {
      node.classList.add("revealed");
      node.title = "carte encore cachée à ce moment";
    }
  }
  return node;
}

function levelBadge(level) {
  return el("span", `level level-${level.normalize("NFD").replace(/[^a-zA-Z]/g, "").toLowerCase()}`, level);
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.hidden = true; }, 6000);
}

/** Record by style and number of players (same figures as play_vs_ai.py --bilan). */
function recordTable(groups) {
  const table = el("table", "result");
  const head = el("tr");
  for (const label of ["Style des IA", "Joueurs", "Parties", "Victoires", "Hasard", "Ton score", "Score IA", "Écart", "Tes col.", "Col. IA"]) {
    head.append(el("th", "", label));
  }
  table.append(head);
  for (const g of groups) {
    const tr = el("tr");
    tr.append(el("td", "", g.mode), el("td", "", String(g.n_players)), el("td", "", String(g.games)),
      el("td", "", `${g.wins} (${pct(g.wins / g.games)})`), el("td", "", pct(g.chance)),
      el("td", "", fmt(g.my_score)), el("td", "", fmt(g.ai_score)), el("td", "", signed(g.gap)),
      el("td", "", fmt(g.my_columns)), el("td", "", fmt(g.ai_columns)));
    table.append(tr);
  }
  return table;
}

/** Recent games, with a link to their replay when there is one. */
function gamesTable(games) {
  const table = el("table", "result games");
  const head = el("tr");
  for (const label of ["Date", "Joueurs", "Adversaires", "Ton score", "Scores IA", "Résultat", "Manches", ""]) {
    head.append(el("th", "", label));
  }
  table.append(head);
  for (const g of games) {
    const tr = el("tr", g.human_won ? "win" : "");
    const link = el("td", "");
    if (g.replayable) {
      const a = el("a", "", "Revoir");
      a.href = SKYJO.replayUrl(g.id);
      link.append(a);
    } else {
      link.append(el("span", "note", "pas de relecture"));
    }
    tr.append(el("td", "", dateText(g.date)), el("td", "", String(g.n_players)), el("td", "left", g.styles.join(", ")),
      el("td", "", String(g.scores[0])), el("td", "", g.scores.slice(1).join(" · ")),
      el("td", "", g.human_won ? "gagnée" : "perdue"), el("td", "", String(g.rounds)), link);
    table.append(tr);
  }
  return table;
}
