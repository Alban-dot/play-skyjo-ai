// Skyjo against the AIs: the page only knows the public state sent by the
// server (hidden cards have no value). Only the legal cells are active buttons.
"use strict";

const GAME_KEY = "skyjo-game";
const SPEED_KEY = "skyjo-speed";
const PSEUDO_KEY = "skyjo-pseudo";
const STYLES_KEY = "skyjo-styles";
const SHOW_LOG_KEY = "skyjo-show-log";
const DEFAULT_STYLE = "expert";
const PSEUDO_RULE = /^[\p{L}\p{N} -]{2,20}$/u;

const ui = {
  state: null,      // last public state shown
  settings: null,   // {pseudo, styles} of the current game
  styles: [],       // styles offered by the server
  busy: false,      // waiting for the server or replaying the AIs' moves
  picks: [],        // setup: cells chosen so far
  discarding: false, // drawn card: the player chose to discard it, a hidden card is to be flipped
  flash: null,      // {player, pos} of the last move
  lastMoves: {},    // seat -> last turn event of that AI in the round
  gameEnd: null,    // game_end event, shown once the replay is over
};

const me = () => ui.state.you;
const label = (seat) => (seat === me() ? "Toi" : ui.state.names[seat]);

// --- server ----------------------------------------------------------------------

function delay() {
  return Number($("#speed").value);
}

function interactive() {
  const s = ui.state;
  return Boolean(s) && !ui.busy && s.your_turn && !s.game_over && $("#modal").hidden;
}

async function newGame(settings) {
  if (ui.busy) return;
  ui.busy = true;
  try {
    const data = await api("POST", "/api/games", settings);
    Object.assign(ui, { state: data.state, picks: [], discarding: false, flash: null, gameEnd: null, lastMoves: {} });
    ui.settings = { pseudo: data.state.names[data.state.you], styles: settings.styles };
    store(SESSION, GAME_KEY, data.state.id);
    $("#log").replaceChildren();
    const seed = settings.seed === null ? "graine au hasard" : `graine ${settings.seed}`;
    addLog({ kind: "info", text: `Nouvelle partie : ${data.state.n_players} joueurs, ${seed}.` });
    showScreen("game");
    await playEvents(data.events);
  } catch (err) {
    toast(err.message);
  } finally {
    ui.busy = false;
    render();
  }
}

async function send(action) {
  if (!interactive()) return;
  const id = ui.state.id;
  Object.assign(ui, { busy: true, picks: [], discarding: false, flash: null });
  render();
  try {
    const data = await api("POST", `/api/games/${id}/action`, action);
    await playEvents(data.events);
    ui.state = data.state;
  } catch (err) {
    toast(err.message);
    try {
      ui.state = (await api("GET", `/api/games/${id}`)).state;
    } catch {
      // keep the last state shown
    }
  } finally {
    ui.busy = false;
    render();
  }
  if (ui.gameEnd) {
    const event = ui.gameEnd;
    ui.gameEnd = null;
    await showGameEnd(event);
  }
}

/** Replay the events one by one, each with the state right after it; pause before each AI move. */
async function playEvents(events) {
  for (const event of events) {
    const aiMove = event.player !== undefined && event.player !== me() && (event.kind === "turn" || event.kind === "setup");
    if (aiMove && delay() > 0) await sleep(delay());
    ui.state = event.state;
    ui.flash = event.kind === "turn" ? { player: event.player, pos: event.pos } : null;
    if (event.kind === "turn" && event.player !== me()) ui.lastMoves[event.player] = event;
    if (event.kind === "round_end") ui.lastMoves = {};
    addLog(event);
    render();
    if (event.kind === "round_end") await showRoundEnd(event);
    if (event.kind === "game_end") ui.gameEnd = event;
  }
}

// --- table -------------------------------------------------------------------------

/** What a click on my cell ``pos`` does now, or null if it is not clickable. */
function cellAction(pos) {
  if (!interactive()) return null;
  const { phase, legal } = ui.state;
  if (phase === "setup") return legal.setup.includes(pos) ? { pick: pos } : null;
  if (phase === "place_drawn" && ui.discarding) return legal.flip.includes(pos) ? { type: "flip", pos } : null;
  if (phase === "place_drawn" || phase === "place_discard") return legal.place.includes(pos) ? { type: "place", pos } : null;
  return null;
}

function clickCell(pos) {
  const action = cellAction(pos);
  if (!action) return;
  if (action.pick === undefined) {
    send(action);
    return;
  }
  ui.picks = ui.picks.includes(pos) ? ui.picks.filter((p) => p !== pos) : [...ui.picks, pos];
  if (ui.picks.length === 2) send({ type: "setup", positions: ui.picks });
  else render();
}

function lastMoveText(event) {
  if (event.source === "discard") return `a pris le ${event.card} à la défausse`;
  if (event.action === "flip") return `a pioché et jeté un ${event.card}, puis retourné une carte`;
  return `a pioché et posé un ${event.card}`;
}

function playerPanel(seat) {
  const s = ui.state;
  const player = s.players[seat];
  const mine = seat === me();
  const panel = el("div", "player");
  if (!s.game_over && s.current_player === seat) panel.classList.add("current");
  const head = el("div", "player-head");
  head.append(el("span", "player-name", mine ? `Toi · ${player.name}` : player.name));
  if (player.style) head.append(levelBadge(player.style.level));
  head.append(el("span", "player-stats", `total ${s.total_scores[seat]} · visible ${player.visible_score}`));
  if (s.finisher === seat) head.append(el("span", "badge", mine ? "tu as tout retourné" : "a tout retourné"));
  if (!s.game_over && s.current_player === seat && !mine) head.append(el("span", "badge turn", "joue"));
  panel.append(head);
  const last = ui.lastMoves[seat];
  if (!mine) {
    panel.append(el("p", "last-move-text", last ? `Dernier coup : ${lastMoveText(last)}` : " "));
  }
  const grid = el("div", "grid");
  player.grid.forEach((cell, pos) => {
    const node = cardNode(cell, mine);
    if (mine) {
      const clickable = cellAction(pos) !== null;
      node.disabled = !clickable;
      node.classList.toggle("clickable", clickable);
      node.classList.toggle("selected", ui.picks.includes(pos));
      node.addEventListener("click", () => clickCell(pos));
      if (cell.state === "hidden") node.setAttribute("aria-label", `ta carte cachée, case ${pos + 1}`);
    }
    if (last && last.pos === pos) {
      node.classList.add("last-move");
      if (last.source === "discard") node.classList.add("from-discard");
    }
    if (ui.flash && ui.flash.player === seat && ui.flash.pos === pos) node.classList.add("flash");
    grid.append(node);
  });
  panel.append(grid);
  return panel;
}

function promptText() {
  const s = ui.state;
  const { phase, legal } = s;
  if (s.game_over) return "Partie terminée.";
  if (!s.your_turn) return `${label(s.current_player)} joue…`;
  if (ui.busy) return "…";
  if (phase === "setup") return ui.picks.length ? "Encore une carte à retourner." : "Début de manche : retourne deux de tes cartes.";
  if (phase === "choose_source") {
    return legal.take_discard ? `Pioche une carte, ou prends le ${s.top_discard} de la défausse.` : "Pioche une carte.";
  }
  if (phase === "place_discard") return `Place le ${s.hand} sur une de tes cases.`;
  if (ui.discarding) return `Tu jettes le ${s.hand} : retourne une de tes cartes cachées (clique sur ta carte pour annuler).`;
  return legal.flip.length
    ? `Place ton ${s.hand} sur une case, ou clique sur la défausse pour le jeter.`
    : `Place ton ${s.hand} sur une case.`;
}

function render() {
  const s = ui.state;
  if (!s) return;
  $("#info").textContent = `Manche ${s.round} · ${s.n_players} joueurs`;

  const opponents = $("#opponents");
  opponents.classList.toggle("many", s.n_players === 4);
  const others = s.players.map((_, seat) => seat).filter((seat) => seat !== me());
  opponents.replaceChildren(...others.map(playerPanel));
  $("#me").replaceChildren(playerPanel(me()));

  const ok = interactive();
  const draw = $("#draw");
  const canDraw = ok && s.phase === "choose_source" && s.legal.draw;
  draw.disabled = !canDraw;
  draw.classList.toggle("clickable", canDraw);
  $("#draw-count").textContent = `(${s.draw_pile_size})`;

  const discard = $("#discard");
  discard.className = "card big";
  discard.textContent = "";
  if (s.top_discard === null) {
    discard.classList.add("empty");
  } else {
    discard.classList.add(valueClass(s.top_discard));
    discard.textContent = s.top_discard;
  }
  const canDiscard = ok && ((s.phase === "choose_source" && s.legal.take_discard)
    || (s.phase === "place_drawn" && s.legal.flip.length > 0));
  discard.disabled = !canDiscard;
  discard.classList.toggle("clickable", canDiscard);
  discard.classList.toggle("selected", ui.discarding);
  discard.setAttribute("aria-label", s.top_discard === null ? "défausse vide" : `défausse : ${s.top_discard}`);

  const holding = s.hand !== null && s.hand !== undefined;
  $("#hand-pile").hidden = !holding;
  if (holding) {
    const hand = $("#hand");
    hand.className = `card big ${valueClass(s.hand)}`;
    hand.textContent = s.hand;
  }

  const banner = $("#banner");
  banner.hidden = s.finisher === null || s.game_over;
  if (!banner.hidden) {
    banner.textContent = s.finisher === me()
      ? "Dernier tour ! Tu as retourné toutes tes cartes."
      : `Dernier tour ! ${s.names[s.finisher]} a retourné toutes ses cartes.`;
  }
  $("#prompt").textContent = promptText();

  renderScores();
  const end = $("#end-actions");
  end.hidden = !s.game_over || ui.busy;
  if (!end.hidden && !end.childElementCount) {
    end.append(...endButtons(s.seed).map(([text, value, cls]) => {
      const button = el("button", cls, text);
      button.type = "button";
      button.addEventListener("click", () => afterGame(value, s.seed));
      return button;
    }));
  }
  if (end.hidden) end.replaceChildren();
}

function renderScores() {
  const s = ui.state;
  const last = s.last_round;
  const table = $("#scores");
  const head = el("thead");
  const row = el("tr");
  row.append(el("th", "", "Joueur"), el("th", "", "Manche préc."), el("th", "", "Total"));
  head.append(row);
  const body = el("tbody");
  s.names.forEach((_, seat) => {
    const tr = el("tr", seat === me() ? "you" : "");
    let previous = "–";
    if (last) {
      previous = String(last.scores[seat]);
      if (last.scores[seat] !== last.raw_scores[seat]) previous += " (×2)";
      if (last.finisher === seat) previous += " ✱";
    }
    tr.append(el("td", "", label(seat)), el("td", "", previous), el("td", "", String(s.total_scores[seat])));
    body.append(tr);
  });
  table.replaceChildren(head, body);
  if (last && last.finisher !== null) table.append(el("caption", "note", "✱ a terminé la manche précédente"));
}

function addLog(event, fresh = true) {
  const item = el("li", fresh ? "new" : "", event.text);
  if (event.player !== undefined && ui.state && event.player === me()) item.classList.add("mine");
  if (["column", "last_turn", "first_player", "game_end"].includes(event.kind)) item.classList.add("important");
  if (event.kind === "round_end") item.classList.add("round");
  const log = $("#log");
  log.prepend(item);
  while (log.childElementCount > 400) log.lastElementChild.remove();
}

/** Wide screen: the log column is shown or not (remembered in the browser). */
function setLogShown(shown) {
  $("#show-log").checked = shown;
  $("#game").classList.toggle("log-hidden", !shown);
  store(LOCAL, SHOW_LOG_KEY, shown ? "1" : "0");
}

/** Narrow screen: the log under the table, folded until opened. */
function toggleLogFold() {
  const journal = $("#journal");
  const open = journal.classList.toggle("open");
  const button = $("#btn-log");
  button.setAttribute("aria-expanded", String(open));
  button.textContent = open ? "Masquer" : "Afficher";
}

// --- modals --------------------------------------------------------------------------

function openModal(content, buttons) {
  return new Promise((resolve) => {
    const box = $("#modal-box");
    box.replaceChildren(...content);
    const bar = el("div", "modal-actions");
    for (const [text, value, cls] of buttons) {
      const button = el("button", cls || "secondary", text);
      button.type = "button";
      button.addEventListener("click", () => {
        $("#modal").hidden = true;
        render();
        resolve(value);
      });
      bar.append(button);
    }
    box.append(bar);
    $("#modal").hidden = false;
    render();
    bar.lastElementChild.focus();
  });
}

function revealedGrids(names, grids, scores) {
  const wrap = el("div", "reveal");
  grids.forEach((grid, seat) => {
    const panel = el("div", "player");
    const head = el("div", "player-head");
    head.append(el("span", "player-name", names[seat]), el("span", "player-stats", `${scores[seat]} pts`));
    const cells = el("div", "grid");
    cells.append(...grid.map((cell) => cardNode(cell)));
    panel.append(head, cells);
    wrap.append(panel);
  });
  return wrap;
}

function doublingText(event, names) {
  const f = event.finisher;
  if (f === null) return "Manche arrêtée sans joueur ayant tout retourné : pas de doublement.";
  const raw = event.raw_scores[f];
  const best = Math.min(...event.raw_scores.filter((_, i) => i !== f));
  const who = f === me() ? "Tu as" : `${names[f]} a`;
  if (event.scores[f] !== raw) {
    return `${who} terminé avec ${raw} points sans avoir strictement le plus petit score `
      + `(meilleur adversaire : ${best}) : ${raw} × 2 = ${event.scores[f]}.`;
  }
  if (raw <= 0) return `${who} terminé avec ${raw} point(s) : score nul ou négatif, pas de doublement.`;
  return `${who} terminé avec strictement le plus petit score (${raw} contre ${best}) : pas de doublement.`;
}

async function showRoundEnd(event) {
  const s = event.state;
  const names = s.names.map((n, seat) => (seat === me() ? "Toi" : n));
  const title = el("h2", "", `Fin de la manche ${event.round}`);
  const sub = el("p", "sub", event.finisher === null ? "Manche arrêtée."
    : event.finisher === me() ? "Tu as terminé la manche. Toutes les cartes sont révélées."
      : `${names[event.finisher]} a terminé la manche. Toutes les cartes sont révélées.`);
  const table = el("table", "result");
  const head = el("tr");
  head.append(el("th", "", "Joueur"), el("th", "", "Somme des cartes"), el("th", "", "Points de la manche"), el("th", "", "Total"));
  table.append(head);
  names.forEach((name, seat) => {
    const tr = el("tr", "");
    const doubled = event.scores[seat] !== event.raw_scores[seat];
    const points = el("td", doubled ? "doubled" : "", doubled
      ? `${event.raw_scores[seat]} × 2 = ${event.scores[seat]}` : String(event.scores[seat]));
    tr.append(el("td", "", name + (event.finisher === seat ? " ✱" : "")), el("td", "", String(event.raw_scores[seat])),
      points, el("td", "", String(event.total_scores[seat])));
    table.append(tr);
  });
  const detail = el("p", "note", `✱ ${doublingText(event, names)}`);
  const next = s.game_over ? "Voir le classement" : "Manche suivante";
  await openModal([title, sub, revealedGrids(names, event.grids, event.scores), table, detail],
    [[next, "next", "primary"]]);
}

function ranking(totals) {
  const order = totals.map((t, seat) => [t, seat]).sort((a, b) => a[0] - b[0]);
  return order.map(([total, seat]) => ({ seat, total, rank: 1 + totals.filter((t) => t < total).length }));
}

function endButtons(seed) {
  return [["Rejouer", "replay", "primary"], [`Mêmes donnes (graine ${seed})`, "same", "secondary"],
    ["Revoir la partie", "review", "secondary"], ["Mon bilan", "record", "secondary"], ["Accueil", "home", "secondary"]];
}

async function showGameEnd(event) {
  const winners = event.winners;
  const title = el("h2", "", winners.length === 1
    ? (winners[0] === me() ? "Tu as gagné !" : `${label(winners[0])} a gagné.`)
    : `Égalité pour la première place : ${winners.map(label).join(", ")}.`);
  const sub = el("p", "sub", `Partie terminée. Graine ${event.seed} : « Mêmes donnes » rejoue les mêmes cartes. `
    + "La partie est enregistrée dans ton bilan ; « Revoir la partie » la rejoue pas à pas, cartes révélées.");
  const table = el("table", "result");
  const head = el("tr");
  head.append(el("th", "", "Rang"), el("th", "left", "Joueur"), el("th", "", "Score total"));
  table.append(head);
  for (const { seat, total, rank } of ranking(event.total_scores)) {
    const tr = el("tr", winners.includes(seat) ? "win" : "");
    tr.append(el("td", "", rank === 1 ? "1er" : `${rank}e`), el("td", "left", label(seat)), el("td", "", String(total)));
    table.append(tr);
  }
  const choice = await openModal([title, sub, table], [...endButtons(event.seed), ["Voir la table", "close", "ghost"]]);
  await afterGame(choice, event.seed);
}

async function afterGame(choice, seed) {
  if (choice === "replay") await newGame({ ...ui.settings, seed: null });
  else if (choice === "same") await newGame({ ...ui.settings, seed });
  else if (choice === "review") window.open(SKYJO.replayUrl(ui.state.id), "_blank", "noopener");
  else if (choice === "record") await showRecord();
  else if (choice === "home") goHome(false);
}

async function showRecord() {
  const pseudo = ui.settings ? ui.settings.pseudo : $("#pseudo").value.trim();
  const everyGame = SKYJO.pseudoOptional; // the record of this browser, whatever the pseudo
  if (!everyGame && !PSEUDO_RULE.test(pseudo)) {
    toast("Indique d'abord ton pseudo pour voir ton bilan.");
    return;
  }
  let data;
  try {
    data = await api("GET", everyGame ? "/api/record" : `/api/record?pseudo=${encodeURIComponent(pseudo)}`);
  } catch (err) {
    toast(err.message);
    return;
  }
  const content = [el("h2", "", data.pseudo ? `Bilan de ${data.pseudo}` : "Ton bilan dans ce navigateur")];
  const total = data.groups.reduce((n, g) => n + g.games, 0);
  if (!total) {
    content.push(el("p", "sub", data.pseudo ? "Aucune partie terminée sous ce pseudo pour l'instant."
      : "Aucune partie terminée dans ce navigateur pour l'instant."));
  } else {
    content.push(el("p", "sub", `${total} partie${total > 1 ? "s" : ""} terminée${total > 1 ? "s" : ""}, par style d'adversaires.`));
    content.push(recordTable(data.groups), el("p", "note", data.note), el("h3", "", "Dernières parties"), gamesTable(data.games));
  }
  await openModal(content, [["Fermer", "close", "primary"]]);
}

// --- home ----------------------------------------------------------------------------

function showScreen(name) {
  $("#home").hidden = name !== "home";
  $("#game").hidden = name !== "game";
  $("#btn-home").hidden = name !== "game";
  if (name === "home") $("#info").textContent = "";
}

function goHome(confirmLeave = true) {
  const s = ui.state;
  if (confirmLeave && s && !s.game_over && !window.confirm("Abandonner la partie en cours ? Elle ne sera pas enregistrée.")) {
    return;
  }
  ui.state = null;
  ui.settings = null;
  store(SESSION, GAME_KEY, null);
  showScreen("home");
}

function savedStyles() {
  try {
    const saved = JSON.parse(store(LOCAL, STYLES_KEY) || "[]");
    return Array.isArray(saved) ? saved : [];
  } catch {
    return [];
  }
}

/** One style selector per AI seat, with the description of the chosen style. */
function renderSeats() {
  const n = Number($("#new-game").players.value);
  const saved = savedStyles();
  const current = [...document.querySelectorAll("#seats select")].map((s) => s.value);
  const seats = $("#seats");
  seats.replaceChildren();
  for (let i = 0; i < n - 1; i += 1) {
    const wanted = current[i] || saved[i] || DEFAULT_STYLE;
    const row = el("div", "seat");
    const labelNode = el("label", "seat-label", `Adversaire ${i + 1}`);
    const select = el("select");
    select.id = `seat-${i}`;
    labelNode.htmlFor = select.id;
    for (const style of ui.styles) {
      const option = el("option", "", `${style.name} — ${style.level}`);
      option.value = style.id;
      select.append(option);
    }
    select.value = ui.styles.some((s) => s.id === wanted) ? wanted : ui.styles[0].id;
    const description = el("p", "seat-description");
    const describe = () => {
      const style = ui.styles.find((s) => s.id === select.value);
      description.replaceChildren(levelBadge(style.level), el("span", "", ` ${style.description}`));
    };
    select.addEventListener("change", describe);
    describe();
    row.append(labelNode, select, description);
    seats.append(row);
  }
}

async function buildHome() {
  $("#pseudo").value = store(LOCAL, PSEUDO_KEY) || "";
  try {
    ui.styles = (await api("GET", "/api/styles")).styles;
  } catch (err) {
    toast(err.message);
    return;
  }
  renderSeats();
  for (const input of document.querySelectorAll("input[name=players]")) input.addEventListener("change", renderSeats);
  $("#new-game").addEventListener("submit", (e) => {
    e.preventDefault();
    const pseudo = $("#pseudo").value.trim().replace(/\s+/g, " ");
    if (!PSEUDO_RULE.test(pseudo) && !(SKYJO.pseudoOptional && pseudo === "")) {
      toast("Pseudo : 2 à 20 caractères, lettres, chiffres, espaces ou tirets.");
      $("#pseudo").focus();
      return;
    }
    const raw = $("#seed").value.trim();
    const seed = raw === "" ? null : Number(raw);
    if (seed !== null && (!Number.isInteger(seed) || seed < 0)) {
      toast("La graine doit être un entier positif ou nul.");
      return;
    }
    const styles = [...document.querySelectorAll("#seats select")].map((s) => s.value);
    store(LOCAL, PSEUDO_KEY, pseudo);
    store(LOCAL, STYLES_KEY, JSON.stringify(styles));
    newGame({ pseudo: pseudo || null, styles, seed });
  });
}

async function resume() {
  const id = store(SESSION, GAME_KEY);
  if (!id) return;
  try {
    const data = await api("GET", `/api/games/${id}?log=1`);
    if (data.state.game_over) {
      store(SESSION, GAME_KEY, null);
      return;
    }
    ui.state = data.state;
    ui.settings = { pseudo: data.state.names[data.state.you],
      styles: data.state.players.filter((p) => p.style).map((p) => p.style.id) };
    // the log and the last move of each AI, rebuilt from the events of the game
    for (const event of data.log || []) {
      addLog(event, false);
      if (event.kind === "turn" && event.player !== me()) ui.lastMoves[event.player] = event;
      if (event.kind === "round_end") ui.lastMoves = {};
    }
    addLog({ kind: "info", text: "Partie reprise." });
    showScreen("game");
    render();
  } catch {
    store(SESSION, GAME_KEY, null);
  }
}

function init() {
  buildHome();
  const speed = $("#speed");
  const saved = store(LOCAL, SPEED_KEY);
  if (saved !== null && [...speed.options].some((o) => o.value === saved)) speed.value = saved;
  speed.addEventListener("change", () => store(LOCAL, SPEED_KEY, speed.value));
  setLogShown(store(LOCAL, SHOW_LOG_KEY) !== "0");
  $("#show-log").addEventListener("change", (e) => setLogShown(e.target.checked));
  $("#btn-log").addEventListener("click", toggleLogFold);
  $("#draw").addEventListener("click", () => {
    if (interactive() && ui.state.phase === "choose_source" && ui.state.legal.draw) send({ type: "draw" });
  });
  $("#discard").addEventListener("click", () => {
    if (!interactive()) return;
    const s = ui.state;
    if (s.phase === "choose_source" && s.legal.take_discard) send({ type: "take_discard" });
    else if (s.phase === "place_drawn" && s.legal.flip.length) {
      ui.discarding = !ui.discarding;
      render();
    }
  });
  $("#hand").addEventListener("click", () => {
    if (ui.discarding) {
      ui.discarding = false;
      render();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && (ui.discarding || ui.picks.length) && $("#modal").hidden) {
      Object.assign(ui, { discarding: false, picks: [] });
      render();
    }
  });
  $("#btn-home").addEventListener("click", () => goHome());
  $("#btn-record").addEventListener("click", () => showRecord());
  resume();
}

init();
