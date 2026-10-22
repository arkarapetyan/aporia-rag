// AporiaRAG's page: a conversation. Each question gets a reply that shows the route it took,
// step by step as the backend streams them (see ui/app.py), then the answer, whose citations
// link to the passages in the panel beside the conversation (inside the reply on a narrow screen).

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
const MAIN_Y = 40;
const DETOUR_Y = 86;
const STAGES = [
  "rewriting", "retrieval", "grading", "web search", "generation", "revision", "other authors",
];
const SVG = "http://www.w3.org/2000/svg";
const WIDE = window.matchMedia("(min-width: 1000px)");

const $ = (selector, root = document) => root.querySelector(selector);
const form = $("#ask");
const question = $("#question");
const submit = $("#submit");
const threadInner = $(".thread-inner");

const exchanges = []; // { request, outcome, element }
let defaultK = 5; // the server's, from /api/authors
let running = null; // the exchange being answered: { ..., seen, path, started, timer }

// -- The route map ---------------------------------------------------------

const stationX = (index) => 32 + index * (696 / (STATIONS.length - 1));
const stationY = (station) => (station.main ? MAIN_Y : DETOUR_Y);

function svg(name, attributes, parent) {
  const element = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  parent.append(element);
  return element;
}

function drawMap(map) {
  const mains = STATIONS.map((station, i) => [station, i]).filter(([station]) => station.main);
  const first = stationX(mains[0][1]);
  const last = stationX(mains.at(-1)[1]);
  svg("path", { class: "track", d: `M ${first} ${MAIN_Y} H ${last}` }, map);
  svg("path", {
    class: "track track-detour",
    d: `M ${stationX(0)} ${DETOUR_Y} H ${stationX(STATIONS.length - 1)}`,
  }, map);
  svg("polyline", { class: "taken", points: "" }, map);
  STATIONS.forEach((station, i) => {
    const group = svg("g", {
      class: `station${station.main ? " main" : ""}`,
      "data-station": station.id,
    }, map);
    svg("circle", { cx: stationX(i), cy: stationY(station), r: station.main ? 7.5 : 5.5 }, group);
    const text = svg("text", {
      x: stationX(i),
      y: station.main ? MAIN_Y - 16 : DETOUR_Y + 22,
    }, group);
    station.label.forEach((line, n) => {
      svg("tspan", { x: stationX(i), dy: n === 0 ? 0 : 14 }, text).textContent = line;
    });
  });
}

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

function waitAt(run, id) {
  run.map.querySelectorAll(".waiting").forEach((g) => g.classList.remove("waiting"));
  if (id) run.map.querySelector(`[data-station="${id}"]`)?.classList.add("waiting");
}

function drawTaken(run) {
  const points = run.path.map((id) => {
    const i = STATIONS.findIndex((station) => station.id === id);
    return `${stationX(i)},${stationY(STATIONS[i])}`;
  });
  $(".taken", run.map).setAttribute("points", points.join(" "));
}

// -- Following a run -------------------------------------------------------

// The page scrolls; the bar and the question box stay in place over it.
const scroller = document.scrollingElement;

function nearBottom() {
  return scroller.scrollHeight - scroller.scrollTop - window.innerHeight < 160;
}

function keepInView(force = false) {
  if (force || nearBottom()) window.scrollTo({ top: scroller.scrollHeight });
}

function measureBar() {
  document.documentElement.style.setProperty("--bar-height", `${$(".bar").offsetHeight}px`);
}

function tick(run) {
  const elapsed = (performance.now() - run.started) / 1000;
  let line = $(".working", run.steps);
  if (!line) {
    line = document.createElement("li");
    line.className = "working";
    line.innerHTML = '<span>Working on the next step</span><span class="seconds"></span>';
    run.steps.append(line);
  }
  $(".seconds", line).textContent = `${elapsed.toFixed(1)} s`;
  $(".steps-summary", run.element).textContent = `Working: ${elapsed.toFixed(0)} s so far`;
}

function onStep(run, step) {
  const id = stationFor(step.node, run.seen);
  if (id) {
    run.seen.add(id);
    run.path.push(id);
    run.map.querySelector(`[data-station="${id}"]`)?.classList.add("visited");
    drawTaken(run);
    waitAt(run, id);
  }
  $(".working", run.steps)?.remove();
  const line = document.createElement("li");
  line.innerHTML = `<span></span><span class="seconds">${step.seconds.toFixed(1)} s</span>`;
  line.firstChild.textContent = step.summary;
  run.steps.append(line);
  tick(run);
  keepInView();
}

function stopRun(run, summary) {
  clearInterval(run.timer);
  $(".working", run.steps)?.remove();
  waitAt(run, null);
  const box = $(".steps-box", run.element);
  $(".steps-summary", run.element).textContent = summary;
  box.open = false;
  submit.disabled = false;
  submit.textContent = "Ask";
  document.title = "AporiaRAG";
  running = null;
}

// -- Rendering an outcome --------------------------------------------------

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

function linkCitations(html, count, exchange) {
  return html.replace(CITATION, (_, group) => {
    const links = numbersIn(group).map((n) =>
      n >= 1 && n <= count
        ? `<a class="cite" href="#" data-exchange="${exchange}" data-source="${n}">${n}</a>`
        : `<span class="cite cite-missing" title="No passage has this number">${n}</span>`
    );
    return `[${links.join(", ")}]`;
  });
}

function inline(text, count, exchange) {
  const html = escape(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*?)\*/g, "$1<em>$2</em>");
  return linkCitations(html, count, exchange);
}

function renderAnswer(text, outcome, exchange) {
  const count = outcome.sources.length;
  return text
    .trim()
    .split(/\n\s*\n/)
    .map((block) => {
      const lines = block.split("\n");
      const heading = outcome.survey && lines[0].match(/^(.+?) \[(\d+)\]$/);
      if (heading) {
        const body = lines.slice(1).join(" ");
        return `<h3>${escape(heading[1])} ${linkCitations(`[${heading[2]}]`, count, exchange)}</h3>` +
          `<p>${inline(body, count, exchange)}</p>`;
      }
      if (lines.every((line) => /^\s*[-*] /.test(line))) {
        const items = lines.map(
          (line) => `<li>${inline(line.replace(/^\s*[-*] /, ""), count, exchange)}</li>`
        );
        return `<ul>${items.join("")}</ul>`;
      }
      return `<p>${lines.map((line) => inline(line, count, exchange)).join("<br>")}</p>`;
    })
    .join("");
}

const plural = (count, one, many = `${one}s`) => `${count} ${count === 1 ? one : many}`;

function routeNote(outcome) {
  if (outcome.route === "normal") return "Answered from the corpus on the first search.";
  if (outcome.route === "loop") {
    return `Nothing relevant on the first search, so the question was searched again as “${outcome.query}”.`;
  }
  if (outcome.route === "web") {
    const why = outcome.outside_author
      ? `The rewriter named ${outcome.outside_author}, who is not in the corpus, so the web was searched.`
      : "Nothing in the corpus answered, so the web was searched.";
    return outcome.web_error ? `${why} The web search failed: ${outcome.web_error}.` : why;
  }
  return "";
}

function renderChecks(box, outcome) {
  const remaining = outcome.problems;
  box.hidden = !outcome.draft_problems && !remaining.length;
  box.classList.toggle("clean", !remaining.length);
  box.replaceChildren();
  if (box.hidden) return;
  if (outcome.draft_problems) {
    const first = document.createElement("p");
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

function scopeNote(outcome) {
  if (outcome.author_questions.length) {
    return `Asked of each author on their own: ${outcome.author_questions.join(" ")}`;
  }
  if (outcome.named.length) {
    return `Searched only the authors named in the question: ${outcome.named.join(", ")}.`;
  }
  return "";
}

// The passages of one answer: built for the panel, or for the reply on a narrow screen.
function passagesFor(outcome, exchange, where) {
  const fragment = document.createDocumentFragment();
  const list = document.createElement("ol");
  list.className = "sources";
  for (const source of outcome.sources) {
    const item = document.createElement("li");
    item.className = "source";
    item.id = `${where}-${exchange}-${source.number}`;
    item.innerHTML =
      `<span class="siglum">[${source.number}]</span><span class="citation"></span><blockquote></blockquote>`;
    const citation = $(".citation", item);
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
    $("blockquote", item).textContent = source.text.replace(/\s+/g, " ").trim();
    if (source.text.length > 480) {
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
    list.append(item);
  }
  fragment.append(list);
  const groups = [
    ["Other authors who discuss this", outcome.others],
    ["Left out as not relevant", outcome.left_out],
    ["Not graded, as two relevant passages came first", outcome.skipped],
  ];
  for (const [title, passages] of groups) {
    if (!passages.length) continue;
    const box = document.createElement("details");
    box.className = "more";
    box.innerHTML = `<summary></summary><ul class="passage-list"></ul>`;
    $("summary", box).textContent = `${title} (${passages.length})`;
    for (const passage of passages) {
      const item = document.createElement("li");
      item.textContent = passage.citation;
      item.title = passage.text.replace(/\s+/g, " ").slice(0, 300);
      $(".passage-list", box).append(item);
    }
    fragment.append(box);
  }
  return fragment;
}

// -- The passages panel: closed until a reply's button or a citation opens it ---

let panelFor = null; // the exchange whose passages the open panel shows

function passagesLabel(outcome) {
  const graded = outcome.left_out.length + outcome.skipped.length;
  if (!outcome.sources.length) return `The passages graded (${graded})`;
  return `${outcome.web ? "Web sources" : "Passages"} (${outcome.sources.length})`;
}

function syncButtons() {
  exchanges.forEach((exchange, index) => {
    $(".open-passages", exchange.element)?.setAttribute(
      "aria-expanded", String(panelFor === index)
    );
  });
}

function openPanel(index) {
  showInPanel(index);
  panelFor = index;
  $("#apparatus").hidden = false;
  $(".app").classList.add("with-panel");
  syncButtons();
}

function closePanel() {
  panelFor = null;
  $(".app").classList.remove("with-panel");
  $("#apparatus").hidden = true;
  syncButtons();
}

function showInPanel(index) {
  const { request, outcome } = exchanges[index];
  $("#apparatus-title").textContent = outcome.web && outcome.sources.length ? "Web sources" : "Passages";
  const text = request.question.length > 90 ? `${request.question.slice(0, 90)}…` : request.question;
  $("#apparatus-for").textContent = `For: “${text}”`;
  $("#apparatus-body").replaceChildren(passagesFor(outcome, index, "panel"));
}

function renderOutcome(run, outcome) {
  const index = exchanges.indexOf(run.exchange);
  run.exchange.outcome = outcome;
  const element = run.element;
  const body = $(".reply-body", element);
  const note = scopeNote(outcome);
  $(".scope-note", element).hidden = !note;
  $(".scope-note", element).textContent = note;
  const text = $(".answer-text", element);
  if (outcome.answer === null) {
    const graded = outcome.left_out.length;
    text.innerHTML = `<p>${
      graded === 1
        ? "The one passage found does not answer this question."
        : `None of the ${graded} passages found answers this question.`
    }</p>`;
  } else {
    text.innerHTML = renderAnswer(outcome.answer, outcome, index);
  }
  renderChecks($(".checks", element), outcome);
  $(".route-note", element).textContent = routeNote(outcome);
  const inlinePassages = $(".inline-passages", element);
  const count = outcome.sources.length + outcome.others.length + outcome.left_out.length + outcome.skipped.length;
  inlinePassages.hidden = count === 0;
  $("summary", inlinePassages).textContent = passagesLabel(outcome);
  $(".inline-passages-body", inlinePassages).replaceChildren(passagesFor(outcome, index, "inline"));
  const button = $(".open-passages", element);
  button.hidden = count === 0;
  button.textContent = passagesLabel(outcome);
  button.addEventListener("click", () => (panelFor === index ? closePanel() : openPanel(index)));
  body.hidden = false;
  if (panelFor !== null) openPanel(index); // an open panel follows the newest answer

  const total = (performance.now() - run.started) / 1000;
  const stages = STAGES.filter((stage) => stage in outcome.seconds).map(
    (stage) => `${stage} ${outcome.seconds[stage].toFixed(1)} s`
  );
  stopRun(run, `${plural(run.steps.children.length, "step")} in ${total.toFixed(1)} s: ${stages.join(", ")}`);
  // The question at the top of the view, its route and answer right under it.
  const top = element.getBoundingClientRect().top + scroller.scrollTop - $(".bar").offsetHeight - 8;
  window.scrollTo({ top });
}

function renderError(run, message) {
  const box = $(".error", run.element);
  box.hidden = false;
  box.textContent = message;
  stopRun(run, "Stopped");
  keepInView(true);
}

// -- Asking ----------------------------------------------------------------

function requestFromForm() {
  const author = $("#author").value;
  return {
    question: question.value.trim(),
    author: author && author !== "*" ? author : null,
    all_authors: author === "*",
    k: Number($("#k").value) || 5,
    others: $("#others").checked,
  };
}

function optionLabels(request) {
  const labels = [];
  if (request.all_authors) labels.push("All authors");
  if (request.author) labels.push(request.author);
  if (request.k !== defaultK) labels.push(plural(request.k, "passage"));
  if (request.others) labels.push("Other authors");
  return labels;
}

function startExchange(request) {
  $("#welcome").hidden = true;
  const element = $("#exchange").content.firstElementChild.cloneNode(true);
  $(".question-text", element).textContent = request.question;
  const options = $(".question-options", element);
  for (const label of optionLabels(request)) {
    const item = document.createElement("li");
    item.textContent = label;
    options.append(item);
  }
  options.hidden = !options.children.length;
  drawMap($(".map", element));
  threadInner.append(element);
  const exchange = { request, outcome: null, element };
  exchanges.push(exchange);
  const run = {
    exchange,
    element,
    map: $(".map", element),
    steps: $(".steps", element),
    seen: new Set(),
    path: [],
    started: performance.now(),
    timer: null,
  };
  waitAt(run, "search");
  tick(run);
  run.timer = setInterval(() => tick(run), 250);
  keepInView(true);
  return run;
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
  if (running) return;
  const request = requestFromForm();
  if (!request.question) {
    question.focus();
    return;
  }
  question.value = "";
  autoGrow();
  submit.disabled = true;
  submit.textContent = "Answering…";
  document.title = "Answering… AporiaRAG";
  const run = startExchange(request);
  running = run;
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      renderError(run, refusal(response.status, body));
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
        if (message.event === "step") onStep(run, message);
        if (message.event === "outcome") renderOutcome(run, message);
        if (message.event === "error") renderError(run, message.message);
      }
    }
    if (running === run) renderError(run, "The answer stopped before it was complete.");
  } catch (error) {
    if (running === run) {
      renderError(run, `The server could not be reached (${error.message}). Is \`aporia-rag serve\` running?`);
    }
  }
}

// -- Wiring ----------------------------------------------------------------

function autoGrow() {
  question.style.height = "auto";
  question.style.height = `${Math.min(question.scrollHeight + 2, 160)}px`;
}

document.addEventListener("click", (event) => {
  const cite = event.target.closest(".cite[data-source]");
  if (!cite) return;
  event.preventDefault();
  const exchange = Number(cite.dataset.exchange);
  let target;
  if (WIDE.matches) {
    openPanel(exchange);
    target = document.getElementById(`panel-${exchange}-${cite.dataset.source}`);
  } else {
    const box = $(".inline-passages", exchanges[exchange].element);
    box.open = true;
    target = document.getElementById(`inline-${exchange}-${cite.dataset.source}`);
  }
  target?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  target?.classList.remove("flash");
  void target?.offsetWidth; // restart the animation
  target?.classList.add("flash");
});

document.querySelectorAll(".example").forEach((button) =>
  button.addEventListener("click", () => {
    question.value = button.textContent;
    autoGrow();
    form.requestSubmit();
  })
);

$("#close-panel").addEventListener("click", closePanel);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && panelFor !== null) closePanel();
});

question.addEventListener("input", autoGrow);
question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", ask);

window.addEventListener("resize", measureBar);

async function start() {
  measureBar();
  try {
    const { authors, k } = await (await fetch("/api/authors")).json();
    const select = $("#author");
    for (const author of authors) select.add(new Option(author, author));
    $("#k").value = k;
    defaultK = k;
  } catch {
    const note = document.createElement("p");
    note.className = "error";
    note.textContent = "The server could not be reached. Is `aporia-rag serve` running?";
    threadInner.append(note);
  }
  question.focus();
}

start();
