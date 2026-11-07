// Administration: pseudos, their record by style and their games (with replays).
// The browser sends the credentials it was asked for when the page was opened.
"use strict";

async function showPlayer(pseudo) {
  let data;
  try {
    data = await api("GET", `/api/admin/players/${encodeURIComponent(pseudo)}`);
  } catch (err) {
    toast(err.message);
    return;
  }
  const box = $("#detail");
  box.hidden = false;
  box.replaceChildren(el("h2", "", `Bilan de ${data.pseudo}`));
  if (!data.groups.length) {
    box.append(el("p", "sub", "Aucune partie."));
  } else {
    box.append(recordTable(data.groups), el("p", "note", data.note),
      el("h3", "", `Dernières parties (${data.games.length} au plus récent)`), gamesTable(data.games));
  }
  box.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function init() {
  let data;
  try {
    data = await api("GET", "/api/admin/players");
  } catch (err) {
    toast(err.message);
    return;
  }
  const wrap = $("#players");
  if (!data.players.length) {
    wrap.replaceChildren(el("p", "sub", "Aucune partie enregistrée."));
    return;
  }
  const table = el("table", "result");
  const head = el("tr");
  for (const text of ["Pseudo", "Parties", "Victoires", "Dernière partie", ""]) head.append(el("th", "", text));
  table.append(head);
  for (const p of data.players) {
    const tr = el("tr");
    const open = el("button", "secondary small", "Voir");
    open.type = "button";
    open.addEventListener("click", () => showPlayer(p.pseudo));
    const cell = el("td");
    cell.append(open);
    tr.append(el("td", "left", p.pseudo), el("td", "", String(p.games)),
      el("td", "", `${p.wins} (${pct(p.wins / p.games)})`), el("td", "", dateText(p.last)), cell);
    table.append(tr);
  }
  wrap.replaceChildren(table);
}

init();
