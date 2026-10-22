// AporiaRAG's page: ask, watch the route light up step by step, read the cited answer.
// The backend streams one JSON line per finished step, then the outcome (see ui/app.py).

"use strict";

// The stations of the route, in the order they can run. Main stations sit on the line;
// detours sit below it and run only when nothing found answers, or the answer needs fixing.
const STATIONS = [
  { id: "ask", label: ["Ask each", "author"] },
  { id: "search", label: ["Search"], main: true },
  { id: "grade", label: ["Grade"], main: true },
  { id: "rewrite", label: ["Rewrite", "question"] },
  { id: "search-again", label: ["Search", "again"] },
  { id: "grade-again", label: ["Grade", "again"] },
  { id: "web", label: ["Search", "the web"] },
  { id: "grade-pages", label: ["Grade", "pages"] },
  { id: "write", label: ["Write"], main: true },
  { id: "check", label: ["Check"], main: true },
  { id: "revise", label: ["Write", "again"] },
  { id: "check-again", label: ["Check", "again"] },
  { id: "others", label: ["Other", "authors"] },
];
const MAIN_Y = 44;
const DETOUR_Y = 98;
const STAGES = [
  "rewriting", "retrieval", "grading", "web search", "generation", "revision", "other authors",
];
const SVG = "http://www.w3.org/2000/svg";

const $ = (selector) => document.querySelector(selector);
const form = $("#ask");
const question = $("#question");
const submit = $("#submit");
const map = $("#map");
const steps = $("#steps");

const x = (index) => 40 + index * (880 / (STATIONS.length - 1));
const y = (station) => (station.main ? MAIN_Y : DETOUR_Y);

function svg(name, attributes = {}, parent = map) {
  const element = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  parent.append(element);
  return element;
}

function drawMap() {
  map.replaceChildren();
  const mains = STATIONS.map((station, i) => [station, i]).filter(([station]) => station.main);
  svg("path", {
    class: "track",
    d: `M ${x(mains[0][1])} ${MAIN_Y} H ${x(mains.at(-1)[1])}`,
  });
  svg("path", { class: "track track-detour", d: `M ${x(0)} ${DETOUR_Y} H ${x(STATIONS.length - 1)}` });
  svg("polyline", { id: "taken", class: "taken", points: "" });
  STATIONS.forEach((station, i) => {
    const group = svg("g", {
      class: `station${station.main ? " main" : ""}`,
      "data-station": station.id,
    });
    svg("circle", { cx: x(i), cy: y(station), r: station.main ? 8 : 6 }, group);
    const text = svg("text", { x: x(i), y: station.main ? MAIN_Y - 18 : DETOUR_Y + 24 }, group);
    station.label.forEach((line, n) => {
      const span = svg("tspan", { x: x(i), dy: n === 0 ? 0 : 15 }, text);
      span.textContent = line;
    });
  });
}

// -- One run ---------------------------------------------------------------

let run = null; // { seen: Set, path: [station ids], started: ms, timer }

function stationFor(node, seen) {
  switch (node) {
    case "ask_each_author":
      return "ask";
    case "retrieve":
      return seen.has("rewrite") ? "search-again" : "search";
    case "grade":
      if (seen.has("web")) return "grade-pages";
      return seen.has("search-again") ? "grade-again" : "grade";
    case "rewrite":
      return "rewrite";
    case "web_search":
      return "web";
    case "generate":
      return "write";
    case "check":
      return seen.has("revise") ? "check-again" : "check";
    case "revise":
      return "revise";
    case "other_authors":
      return "others";
    default:
      return null;
  }
}

function markStation(id, state) {
  map.querySelector(`[data-station="${id}"]`)?.classList.add(state);
}

function clearMap() {
  map.querySelectorAll(".station").forEach((g) => g.classList.remove("visited", "waiting"));
  $("#taken").setAttribute("points", "");
}

function drawTaken() {
  const points = run.path.map((id) => {
    const i = STATIONS.findIndex((station) => station.id === id);
    return `${x(i)},${y(STATIONS[i])}`;
  });
  $("#taken").setAttribute("points", points.join(" "));
}

function wait(id) {
  map.querySelectorAll(".waiting").forEach((g) => g.classList.remove("waiting"));
  if (id) markStation(id, "waiting");
}

function working() {
  let line = steps.querySelector(".working");
  if (!line) {
    line = document.createElement("li");
    line.className = "working";
    line.innerHTML = '<span>Working on the next step</span><span class="seconds"></span>';
    steps.append(line);
  }
  const elapsed = (performance.now() - run.started) / 1000;
  line.querySelector(".seconds").textContent = `${elapsed.toFixed(1)} s`;
}

function onStep(step) {
  const id = stationFor(step.node, run.seen);
  if (id) {
    run.seen.add(id);
    run.path.push(id);
    markStation(id, "visited");
    drawTaken();
    wait(id);
  }
  steps.querySelector(".working")?.remove();
  const line = document.createElement("li");
  line.innerHTML = `<span></span><span class="seconds">${step.seconds.toFixed(1)} s</span>`;
  line.firstChild.textContent = step.summary;
  steps.append(line);
  working();
}

function finish() {
  clearInterval(run.timer);
  steps.querySelector(".working")?.remove();
  wait(null);
  submit.disabled = false;
  submit.textContent = "Ask";
}

// -- The outcome -----------------------------------------------------------

const escape = (text) =>
  text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const CITATION = /\[(\d+(?:\s*[,\-–]\s*\d+)*)\]/g;

function numbersIn(group) {
  return group.split(",").flatMap((part) => {
    const range = part.trim().match(/^(\d+)\s*[-–]\s*(\d+)$/);
    if (!range) return [Number(part.trim())];
    const numbers = [];
    for (let n = Number(range[1]); n <= Number(range[2]); n++) numbers.push(n);
    return numbers;
  });
}

function linkCitations(html, count) {
  return html.replace(CITATION, (_, group) => {
    const links = numbersIn(group).map((n) =>
      n >= 1 && n <= count
        ? `<a class="cite" href="#source-${n}" data-source="${n}">${n}</a>`
        : `<span class="cite cite-missing" title="No passage has this number">${n}</span>`
    );
    return `[${links.join(", ")}]`;
  });
}

function inline(text, count) {
  const html = escape(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*?)\*/g, "$1<em>$2</em>");
  return linkCitations(html, count);
}

function renderAnswer(text, outcome) {
  const count = outcome.sources.length;
  const blocks = text.trim().split(/\n\s*\n/);
  return blocks
    .map((block) => {
      const lines = block.split("\n");
      const heading = outcome.survey && lines[0].match(/^(.+?) \[(\d+)\]$/);
      if (heading) {
        const body = lines.slice(1).join(" ");
        return `<h3>${escape(heading[1])} ${linkCitations(`[${heading[2]}]`, count)}</h3>` +
          `<p>${inline(body, count)}</p>`;
      }
      if (lines.every((line) => /^\s*[-*] /.test(line))) {
        const items = lines.map((line) => `<li>${inline(line.replace(/^\s*[-*] /, ""), count)}</li>`);
        return `<ul>${items.join("")}</ul>`;
      }
      return `<p>${lines.map((line) => inline(line, count)).join("<br>")}</p>`;
    })
    .join("");
}

function plural(count, one, many = `${one}s`) {
  return `${count} ${count === 1 ? one : many}`;
}

function routeNote(outcome) {
  const parts = [];
  if (outcome.route === "normal") parts.push("Answered from the corpus on the first search.");
  if (outcome.route === "loop") {
    parts.push(`Nothing relevant on the first search, so the question was searched again as “${outcome.query}”.`);
  }
  if (outcome.route === "web") {
    parts.push(
      outcome.outside_author
        ? `The rewriter named ${outcome.outside_author}, who is not in the corpus, so the web was searched.`
        : "Nothing in the corpus answered, so the web was searched."
    );
    if (outcome.web_error) parts.push(`The web search failed: ${outcome.web_error}.`);
  }
  return parts.join(" ");
}

function renderChecks(outcome) {
  const box = $("#checks");
  const remaining = outcome.problems;
  box.hidden = !outcome.draft_problems && !remaining.length;
  box.classList.toggle("clean", !remaining.length);
  box.replaceChildren();
  if (box.hidden) return;
  const first = document.createElement("p");
  if (outcome.draft_problems) {
    first.textContent =
      `The first answer had ${plural(outcome.draft_problems, "problem")} with its passages, so it was written again.` +
      (remaining.length ? "" : " The second answer passes the checks.");
    box.append(first);
  }
  if (remaining.length) {
    const intro = document.createElement("p");
    intro.textContent = "Still not borne out by its passages:";
    const list = document.createElement("ul");
    for (const problem of remaining) {
      const item = document.createElement("li");
      const detail = problem.kind === "quote" ? `“${problem.detail}”` : problem.detail;
      item.textContent = `${problem.label[0].toUpperCase()}${problem.label.slice(1)}: ${detail}`;
      list.append(item);
    }
    box.append(intro, list);
  }
}

function sourceItem(source) {
  const item = document.createElement("li");
  item.className = "source";
  item.id = `source-${source.number}`;
  item.innerHTML =
    `<span class="siglum">[${source.number}]</span><span class="citation"></span><blockquote></blockquote>`;
  const citation = item.querySelector(".citation");
  if (source.url) {
    const link = document.createElement("a");
    link.href = source.url;
    link.rel = "noopener noreferrer";
    link.target = "_blank";
    link.textContent = source.citation;
    citation.append(link);
  } else {
    citation.textContent = source.citation;
  }
  const quote = item.querySelector("blockquote");
  quote.textContent = source.text.replace(/\s+/g, " ").trim();
  if (source.text.length > 420) {
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "toggle";
    toggle.textContent = "Show the whole passage";
    toggle.addEventListener("click", () => {
      const open = item.classList.toggle("open");
      toggle.textContent = open ? "Show less" : "Show the whole passage";
    });
    item.append(toggle);
  }
  return item;
}

function fillList(boxId, passages) {
  const box = $(boxId);
  box.hidden = !passages.length;
  box.querySelector(".count").textContent = `(${passages.length})`;
  const list = box.querySelector(".passage-list");
  list.replaceChildren(
    ...passages.map((passage) => {
      const item = document.createElement("li");
      item.textContent = passage.citation;
      item.title = passage.text.replace(/\s+/g, " ").slice(0, 300);
      return item;
    })
  );
}

function scopeNote(outcome) {
  if (outcome.author_questions.length) {
    return `Asked of each author on their own: ${outcome.author_questions.join(" ")}`;
  }
  if (outcome.named.length) {
    return `Searched only the authors named in the question: ${outcome.named.join(", ")}.`;
  }
  return "";
}

function renderOutcome(outcome) {
  $("#result").hidden = false;
  const note = scopeNote(outcome);
  $("#scope-note").hidden = !note;
  $("#scope-note").textContent = note;
  const text = $("#answer-text");
  if (outcome.answer === null) {
    const graded = outcome.left_out.length;
    text.innerHTML = `<p>${
      graded === 1
        ? "The one passage found does not answer this question."
        : `None of the ${graded} passages found answers this question.`
    }</p>`;
  } else {
    text.innerHTML = renderAnswer(outcome.answer, outcome);
  }
  renderChecks(outcome);
  $("#route-note").textContent = routeNote(outcome);
  const stages = STAGES.filter((stage) => stage in outcome.seconds).map(
    (stage) => `${stage} ${outcome.seconds[stage].toFixed(1)} s`
  );
  const total = (performance.now() - run.started) / 1000;
  $("#times").textContent = `Answered in ${total.toFixed(1)} s: ${stages.join(", ")}.`;
  $("#sources-title").textContent = outcome.web ? "Web sources" : "Passages";
  $("#sources").replaceChildren(...outcome.sources.map(sourceItem));
  fillList("#others-box", outcome.others);
  fillList("#left-out-box", outcome.left_out);
  fillList("#skipped-box", outcome.skipped);
}

// -- Asking ----------------------------------------------------------------

function showError(message) {
  const box = $("#error");
  box.hidden = !message;
  box.textContent = message || "";
}

function refusal(status, body) {
  if (status === 422) {
    const field = body?.detail?.[0]?.loc?.at(-1);
    if (field === "question") return "Write a question first.";
    if (field === "k") return "Ask for at least one passage.";
  }
  return typeof body?.detail === "string" ? body.detail : `The server answered ${status}.`;
}

async function ask(event) {
  event.preventDefault();
  const text = question.value.trim();
  if (!text) {
    showError("Write a question first.");
    question.focus();
    return;
  }
  const author = $("#author").value;
  const request = {
    question: text,
    author: author && author !== "*" ? author : null,
    all_authors: author === "*",
    k: Number($("#k").value) || 5,
    others: $("#others").checked,
  };
  showError("");
  $("#result").hidden = true;
  steps.replaceChildren();
  clearMap();
  run = { seen: new Set(), path: [], started: performance.now(), timer: null };
  submit.disabled = true;
  submit.textContent = "Answering…";
  wait("search");
  working();
  run.timer = setInterval(working, 200);
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      showError(refusal(response.status, body));
      return;
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let end;
      while ((end = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, end).trim();
        buffer = buffer.slice(end + 1);
        if (!line) continue;
        const message = JSON.parse(line);
        if (message.event === "step") onStep(message);
        if (message.event === "outcome") renderOutcome(message);
        if (message.event === "error") showError(message.message);
      }
    }
  } catch (error) {
    showError(`The server could not be reached: ${error.message}. Is \`aporia-rag serve\` running?`);
  } finally {
    finish();
  }
}

document.addEventListener("click", (event) => {
  const cite = event.target.closest(".cite[data-source]");
  if (!cite) return;
  event.preventDefault();
  const source = document.getElementById(`source-${cite.dataset.source}`);
  source.scrollIntoView({ behavior: "smooth", block: "nearest" });
  source.classList.remove("flash");
  void source.offsetWidth; // restart the animation
  source.classList.add("flash");
});

document.querySelectorAll(".example").forEach((button) =>
  button.addEventListener("click", () => {
    question.value = button.textContent;
    question.focus();
  })
);

question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) form.requestSubmit();
});

form.addEventListener("submit", ask);

async function start() {
  drawMap();
  try {
    const { authors, k } = await (await fetch("/api/authors")).json();
    const select = $("#author");
    for (const author of authors) select.add(new Option(author, author));
    $("#k").value = k;
  } catch {
    showError("The server could not be reached. Is `aporia-rag serve` running?");
  }
}

start();
