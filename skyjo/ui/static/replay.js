// Replay of a finished game, step by step, every card revealed (the server only
// serves the history of finished games).
"use strict";

const replay = { game: null, steps: [], index: 0, timer: null };
const PLAY_DELAY = 900;

function roundStarts() {
  return replay.steps.map((s, i) => [s, i]).filter(([s, i]) => i === 0 || s.state.round !== replay.steps[i - 1].state.round
    || s.round_end).map(([, i]) => i);
}

function renderSummary() {
  const g = replay.game;
  const box = $("#summary");
  const won = g.human_won ? "gagnée" : "perdue";
  box.replaceChildren(
    el("h2", "", `Partie de ${g.pseudo} — ${won}`),
    el("p", "sub", `${dateText(g.date)} · ${g.n_players} joueurs · ${g.rounds} manche${g.rounds > 1 ? "s" : ""}`
      + ` · graine ${g.seed} · adversaires : ${g.styles.join(", ")}`),
  );
  const table = el("table", "result");
  const head = el("tr");
  head.append(el("th", "left", "Joueur"), el("th", "", "Score final"));
  table.append(head);
  g.names.forEach((name, seat) => {
    const tr = el("tr", g.winners.includes(seat) ? "win" : "");
    tr.append(el("td", "left", name + (g.winners.includes(seat) ? " 🏆" : "")), el("td", "", String(g.scores[seat])));
    table.append(tr);
  });
  box.append(table);
}

function playerBox(name, grid, extra, highlight, current) {
  const panel = el("div", "player");
  if (current) panel.classList.add("current");
  const head = el("div", "player-head");
  head.append(el("span", "player-name", name));
  if (extra) head.append(el("span", "player-stats", extra));
  const cells = el("div", "grid");
  grid.forEach((cell, pos) => {
    const node = cardNode(cell, false, true);
    if (highlight === pos) node.classList.add("last-move");
    cells.append(node);
  });
  panel.append(head, cells);
  return panel;
}

function renderRoundEnd(step) {
  const box = $("#round-end");
  const end = step.round_end;
  box.hidden = !end;
  if (!end) return;
  const names = replay.game.names;
  const finisher = end.finisher === null ? "Manche arrêtée." : `${names[end.finisher]} a terminé la manche.`;
  const table = el("table", "result");
  const head = el("tr");
  head.append(el("th", "left", "Joueur"), el("th", "", "Somme des cartes"), el("th", "", "Points de la manche"),
    el("th", "", "Total"));
  table.append(head);
  names.forEach((name, seat) => {
    const doubled = end.scores[seat] !== end.raw_scores[seat];
    const tr = el("tr");
    tr.append(el("td", "left", name + (end.finisher === seat ? " ✱" : "")), el("td", "", String(end.raw_scores[seat])),
      el("td", doubled ? "doubled" : "", doubled ? `${end.raw_scores[seat]} × 2 = ${end.scores[seat]}`
        : String(end.scores[seat])), el("td", "", String(end.total_scores[seat])));
    table.append(tr);
  });
  box.replaceChildren(el("h3", "", `Fin de la manche ${end.round}`), el("p", "sub", finisher), table);
}

function render() {
  const step = replay.steps[replay.index];
  const s = step.state;
  const names = replay.game.names;
  $("#info").textContent = `Manche ${step.round_end ? step.round_end.round : s.round}`;
  $("#counter").textContent = `Étape ${replay.index} / ${replay.steps.length - 1}`;
  $("#step").value = String(replay.index);
  const texts = $("#texts");
  texts.replaceChildren(...step.texts.map((t) => el("p", "", t)));

  renderRoundEnd(step);
  const players = $("#players");
  players.classList.toggle("many", names.length >= 3);
  if (step.round_end) {
    // end of a round: the grids after the final reveal (the next deal comes at the next step)
    players.replaceChildren(...step.round_end.grids.map((grid, seat) => playerBox(names[seat], grid,
      `${step.round_end.scores[seat]} pts`, null, false)));
  } else {
    players.replaceChildren(...s.players.map((p, seat) => playerBox(names[seat], p.grid,
      `total ${s.total_scores[seat]}`, step.seat === seat ? step.pos : null, s.current_player === seat
        && s.phase !== "game_over")));
  }

  $("#draw-count").textContent = `(${s.draw_pile_size})`;
  const discard = $("#discard");
  discard.className = "card big";
  discard.textContent = "";
  if (s.top_discard === null) discard.classList.add("empty");
  else {
    discard.classList.add(valueClass(s.top_discard));
    discard.textContent = s.top_discard;
  }
  const holding = s.hand !== null && !step.round_end;
  $("#hand-pile").hidden = !holding;
  if (holding) {
    $("#hand").className = `card big ${valueClass(s.hand)}`;
    $("#hand").textContent = s.hand;
    $("#hand-label").textContent = `En main : ${names[s.current_player]}`;
  }

  $("#first").disabled = $("#prev").disabled = $("#prev-round").disabled = replay.index === 0;
  const atEnd = replay.index === replay.steps.length - 1;
  $("#last").disabled = $("#next").disabled = $("#next-round").disabled = atEnd;
  if (atEnd) stop();
}

function go(index) {
  replay.index = Math.max(0, Math.min(replay.steps.length - 1, index));
  render();
}

function goRound(direction) {
  const starts = roundStarts();
  const target = direction > 0 ? starts.find((i) => i > replay.index)
    : [...starts].reverse().find((i) => i < replay.index);
  go(target === undefined ? (direction > 0 ? replay.steps.length - 1 : 0) : target);
}

function stop() {
  clearInterval(replay.timer);
  replay.timer = null;
  $("#play").textContent = "▶ Lecture";
}

function togglePlay() {
  if (replay.timer) {
    stop();
    return;
  }
  if (replay.index === replay.steps.length - 1) go(0);
  replay.timer = setInterval(() => go(replay.index + 1), PLAY_DELAY);
  $("#play").textContent = "⏸ Pause";
}

async function init() {
  // /replay/<id> on the server, replay.html?id=<id> in the browser demo
  const id = new URLSearchParams(window.location.search).get("id")
    || decodeURIComponent(window.location.pathname.split("/").pop());
  let data;
  try {
    data = await api("GET", `/api/games/${encodeURIComponent(id)}/history`);
  } catch (err) {
    $("#summary").replaceChildren(el("h2", "", "Relecture impossible"), el("p", "sub", err.message));
    document.querySelectorAll(".controls, .replay-board").forEach((n) => { n.hidden = true; });
    return;
  }
  replay.game = data.game;
  replay.steps = data.steps;
  $("#step").max = String(replay.steps.length - 1);
  renderSummary();
  $("#first").addEventListener("click", () => go(0));
  $("#last").addEventListener("click", () => go(replay.steps.length - 1));
  $("#prev").addEventListener("click", () => go(replay.index - 1));
  $("#next").addEventListener("click", () => go(replay.index + 1));
  $("#prev-round").addEventListener("click", () => goRound(-1));
  $("#next-round").addEventListener("click", () => goRound(1));
  $("#play").addEventListener("click", togglePlay);
  $("#step").addEventListener("input", (e) => go(Number(e.target.value)));
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.key === "ArrowRight") go(replay.index + 1);
    else if (e.key === "ArrowLeft") go(replay.index - 1);
    else if (e.key === "Home") go(0);
    else if (e.key === "End") go(replay.steps.length - 1);
    else if (e.key === " ") {
      e.preventDefault();
      togglePlay();
    } else return;
  });
  render();
}

init();
