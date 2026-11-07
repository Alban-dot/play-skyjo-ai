// Browser demo: loaded before common.js. Starts the game's Web Worker (worker.js)
// as soon as the page opens, with a progress bar, and answers the page's requests
// through it instead of the Flask server (window.SKYJO.transport, used by api()).
// Kept in this browser only: the finished games (localStorage, at most 50, see
// skyjo/ui/browser.py) and the game in progress (sessionStorage: settings, seed and
// the player's moves, replayed after a reload).
"use strict";

(() => {
  const GAMES_KEY = "skyjo-demo-games";
  const CURRENT_KEY = "skyjo-demo-current";
  const NO_STORAGE = "Ton navigateur refuse le stockage local (navigation privée stricte ou données des sites "
    + "bloquées) : tu peux jouer, mais ton bilan et la relecture des parties ne sont pas disponibles, et une "
    + "partie en cours est perdue si tu recharges la page.";
  const UNSUPPORTED = "Ton navigateur ne peut pas faire tourner le jeu : il lui manque WebAssembly ou les Web "
    + "Workers (navigateur trop ancien, ou fonctions avancées désactivées). Essaie une version récente de "
    + "Chrome, Firefox, Safari ou Edge.";

  // the storage, or null when refused (blocked site data: even reading window.localStorage throws)
  const storage = (name) => {
    try {
      const s = window[name];
      s.setItem("skyjo-probe", "1");
      s.removeItem("skyjo-probe");
      return s;
    } catch {
      return null;
    }
  };
  const LOCAL = storage("localStorage");
  const SESSION = storage("sessionStorage");
  const storageOk = Boolean(LOCAL && SESSION);

  const read = (place, key, fallback) => {
    try {
      const value = place ? place.getItem(key) : null;
      return value === null ? fallback : JSON.parse(value);
    } catch {
      return fallback;
    }
  };
  const write = (place, key, value) => {
    if (!place) return;
    try {
      if (value === null) place.removeItem(key);
      else place.setItem(key, JSON.stringify(value));
    } catch {
      // storage full: the game goes on, it is just not kept
    }
  };

  const supported = typeof WebAssembly === "object" && typeof WebAssembly.instantiate === "function"
    && typeof Worker === "function";
  let markReady;
  let markFailed;
  const ready = new Promise((resolve, reject) => { markReady = resolve; markFailed = reject; });
  ready.catch(() => {}); // reported by the loader and by each request

  // --- loader --------------------------------------------------------------------------

  const node = (id) => document.getElementById(id);
  const submit = () => document.querySelector("#new-game button[type=submit]");

  function showLoader(pct, label, state) {
    const loader = node("loader");
    if (!loader) return;
    loader.dataset.state = state;
    node("loader-fill").style.width = `${pct}%`;
    node("loader-bar").setAttribute("aria-valuenow", String(Math.round(pct)));
    node("loader-text").textContent = label;
    const button = submit();
    if (button) {
      button.disabled = state !== "ready";
      button.textContent = state === "ready" ? "Nouvelle partie" : state === "failed" ? "Jeu indisponible" : "Chargement du jeu…";
    }
  }

  // --- worker --------------------------------------------------------------------------

  let worker = null;
  let next = 0;
  const pending = new Map();

  function start() {
    if (!supported) {
      showLoader(0, UNSUPPORTED, "failed");
      markFailed(new Error(UNSUPPORTED));
      return;
    }
    if (!storageOk && !document.getElementById("new-game")) { // a replay without storage: no game to show
      document.querySelectorAll("[data-hide-when-ready]").forEach((n) => { n.hidden = true; });
      markFailed(new Error(NO_STORAGE));
      return;
    }
    showLoader(1, "Chargement du jeu…", "loading");
    if (!storageOk) {
      const note = document.getElementById("storage-note");
      if (note) {
        note.textContent = NO_STORAGE;
        note.hidden = false;
      }
    }
    worker = new Worker("worker.js");
    worker.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === "progress") showLoader(msg.pct, msg.label, "loading");
      else if (msg.type === "ready") {
        const seconds = msg.seconds.toLocaleString("fr-FR", { maximumFractionDigits: 1 });
        showLoader(100, `Jeu prêt (chargé en ${seconds} s).`, "ready");
        document.querySelectorAll("[data-hide-when-ready]").forEach((n) => { n.hidden = true; });
        markReady();
      } else if (msg.type === "error") {
        showLoader(100, `Le jeu n'a pas pu se charger : ${msg.message}`, "failed");
        markFailed(new Error(msg.message));
      } else if (msg.type === "reply") {
        pending.get(msg.id)(msg.answer);
        pending.delete(msg.id);
      }
    };
    worker.onerror = (event) => {
      showLoader(100, `Le jeu n'a pas pu se charger : ${event.message || "erreur du Worker"}`, "failed");
      markFailed(new Error(event.message || "erreur du Worker"));
    };
    worker.postMessage({ type: "init", games: read(LOCAL, GAMES_KEY, []) });
  }

  async function call(message) {
    await ready;
    const id = next++;
    const answer = await new Promise((resolve) => {
      pending.set(id, resolve);
      worker.postMessage({ type: "request", id, message: JSON.stringify(message) });
    });
    const parsed = JSON.parse(answer);
    if (parsed.games) write(LOCAL, GAMES_KEY, parsed.games);
    return parsed;
  }

  function randomSeed() {
    const value = new Uint32Array(1);
    crypto.getRandomValues(value);
    return value[0] % 1000000; // the range of the server's random seeds
  }

  /** The page's requests (same URLs and answers as the Flask server). */
  async function transport(method, url, body) {
    if (method === "GET" && url === "/api/styles") { // from the build: shown before Pyodide is loaded
      const response = await fetch("styles.json");
      if (!response.ok) throw new Error("styles.json introuvable : lance scripts/build_web_demo.py.");
      return response.json();
    }
    if (!supported) throw new Error(UNSUPPORTED);
    // without storage, the record would only cover this page and a replay (another page) has no game
    if (!storageOk && (url.startsWith("/api/record") || url.endsWith("/history"))) throw new Error(NO_STORAGE);
    let sent = body;
    if (method === "POST" && url === "/api/games") sent = { ...body, seed: body.seed ?? randomSeed() };
    const reload = method === "GET" && url.match(/^\/api\/games\/([^/?]+)\?log=1$/);
    if (reload) { // a reloaded page: rebuild the game in progress in the new worker first
      const current = read(SESSION, CURRENT_KEY, null);
      if (current && current.id === decodeURIComponent(reload[1])) {
        const resumed = await call({ resume: current });
        if (resumed.status !== 200) write(SESSION, CURRENT_KEY, null);
      }
    }
    const answer = await call({ method, url, body: sent });
    if (answer.status !== 200) throw new Error(answer.data.error || `Erreur ${answer.status}.`);
    const data = answer.data;
    if (method === "POST" && url === "/api/games") {
      write(SESSION, CURRENT_KEY, { id: data.state.id, settings: sent, requests: [] });
    } else if (method === "POST" && url.endsWith("/action")) {
      const current = read(SESSION, CURRENT_KEY, null);
      if (data.state.game_over) write(SESSION, CURRENT_KEY, null);
      else if (current && current.id === data.state.id) {
        current.requests.push(body);
        write(SESSION, CURRENT_KEY, current);
      }
    }
    return data;
  }

  window.SKYJO = {
    replayUrl: (id) => `replay.html?id=${encodeURIComponent(id)}`,
    pseudoOptional: true,
    transport,
  };
  start();
})();
