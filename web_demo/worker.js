// The game of the browser demo, in a Web Worker so that the page never freezes:
// Pyodide (served by the site, pyodide/), NumPy, the skyjo modules (skyjo.zip) and
// the network's weights (policy.bin), then skyjo.ui.browser.BrowserBackend answers
// the same requests as the Flask server.
//
// Messages from the page: {type: "init", games}, {type: "request", id, message}
// (message: JSON of {method, url, body} or {resume}). To the page: {type: "progress",
// pct, label}, {type: "ready", seconds}, {type: "error", message}, {type: "reply", id, answer}.
"use strict";

let handle = null; // BrowserBackend.handle, once ready
let ready = null; // promise of the boot

const progress = (pct, label) => postMessage({ type: "progress", pct, label });

async function bytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} : ${response.status}`);
  return response.arrayBuffer();
}

const BOOT = `
import json
from skyjo.agents.numpy_policy import NumpyPolicy
from skyjo.ui.browser import BrowserBackend
from skyjo.ui.styles import load_styles

backend = BrowserBackend(NumpyPolicy.load("/home/pyodide/policy.bin"), load_styles(),
                         json.loads(saved_games), model_name=model_name)
handle = backend.handle
`;

async function boot(games) {
  const start = performance.now();
  progress(2, "Lecture de la configuration…");
  const info = await (await fetch("build_info.json")).json();
  const base = new URL("pyodide/", self.location.href).href;
  progress(5, "Téléchargement de Python (Pyodide)…");
  importScripts(base + "pyodide.js");
  progress(15, "Démarrage de Python…");
  const pyodide = await loadPyodide({ indexURL: base });
  if (pyodide.version !== info.pyodide) {
    throw new Error(`Pyodide ${pyodide.version} trouvé, ${info.pyodide} attendu : relance scripts/fetch_pyodide.py.`);
  }
  progress(60, "Chargement de NumPy, du jeu et de l'IA…");
  const [, zip, weights] = await Promise.all([pyodide.loadPackage("numpy"), bytes("skyjo.zip"), bytes("policy.bin")]);
  progress(85, "Préparation de la partie…");
  pyodide.unpackArchive(zip, "zip", { extractDir: "/home/pyodide" });
  pyodide.FS.writeFile("/home/pyodide/policy.bin", new Uint8Array(weights));
  pyodide.globals.set("saved_games", JSON.stringify(games || []));
  pyodide.globals.set("model_name", info.model);
  pyodide.runPython(BOOT);
  handle = pyodide.globals.get("handle");
  progress(100, "Prêt");
  postMessage({ type: "ready", seconds: (performance.now() - start) / 1000 });
}

self.onmessage = async (event) => {
  const msg = event.data;
  if (msg.type === "init") {
    ready = boot(msg.games).catch((err) => {
      postMessage({ type: "error", message: String(err && err.message ? err.message : err) });
      throw err;
    });
    return;
  }
  if (msg.type === "request") {
    let answer;
    try {
      await ready;
      answer = handle(msg.message);
    } catch (err) {
      answer = JSON.stringify({ status: 500, data: { error: `Erreur du jeu : ${err && err.message ? err.message : err}` } });
    }
    postMessage({ type: "reply", id: msg.id, answer });
  }
};
