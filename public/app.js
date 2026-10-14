(() => {
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];

  function csrfToken() {
    return document.cookie.split("; ").find((entry) => entry.startsWith("rag_csrf="))?.split("=").slice(1).join("=") || "";
  }

  function applyTheme() {
    const dark = localStorage.getItem("rag-theme") === "dark";
    document.body.classList.toggle("dark-preview", dark);
    const button = $("#theme");
    if (button) {
      button.textContent = dark ? "☀" : "☼";
      button.title = dark ? "Switch to light mode" : "Switch to dark mode";
      button.setAttribute("aria-label", button.title);
    }
  }

  function applyProfile() {
    const name = localStorage.getItem("rag-user-name") || "Mehul Gupta";
    const initials = name.split(/\s+/).filter(Boolean).map((part) => part[0]).join("").slice(0, 2).toUpperCase();
    if ($(".profile")) $(".profile").textContent = initials || "U";
    if ($(".welcome h1")) $(".welcome h1").textContent = `Welcome back, ${name}!`;
  }

  function clearDemoContent() {
    $$(".table-row").forEach((row) => row.remove());
    $(".source-list")?.replaceChildren();
    if ($(".source-title")) $(".source-title").textContent = "Sources (0)";
    if ($("#sample-answer")) $("#sample-answer").textContent = "Ask a question to see a grounded answer from your indexed documents.";
  }

  function readHistory() {
    try { return JSON.parse(localStorage.getItem("rag-history") || "[]"); } catch (_) { return []; }
  }

  function saveHistory(query, answer, sources) {
    const history = readHistory();
    history.unshift({ query, answer, sources, time: new Date().toLocaleString(), timestamp: new Date().toISOString() });
    localStorage.setItem("rag-history", JSON.stringify(history.slice(0, 20)));
  }

  function renderSources(sources) {
    const list = $(".source-list");
    const title = $(".source-title");
    if (!list) return;
    if (title) title.textContent = `Sources (${sources.length})`;
    list.innerHTML = sources.length
      ? sources.map((source) => {
          const name = typeof source === "string" ? source : (source.source_file || source.file || source.name || JSON.stringify(source));
          return `<div><span>▧</span><div><strong>${escapeHtml(name)}</strong><small>Retrieved from your indexed documents</small></div></div>`;
        }).join("")
      : "";
  }

  function renderChart(data) {
    const bars = $$(".bars > div");
    const labels = $$(".bars em");
    const tabs = $$(".tabs button");
    if (!bars.length) return;

    const docs = data.documents_list || [];
    const history = readHistory();
    const today = new Date();
    const queryValues = Array.from({ length: 7 }, (_, index) => {
      const day = new Date(today);
      day.setHours(0, 0, 0, 0);
      day.setDate(day.getDate() - (6 - index));
      return history.filter((item) => {
        const date = new Date(item.timestamp || item.time);
        if (Number.isNaN(date.getTime())) return false;
        date.setHours(0, 0, 0, 0);
        return date.getTime() === day.getTime();
      }).length;
    });
    const documentValues = docs.slice(0, 7).map((doc) => Number(doc.chunks) || 0);
    while (documentValues.length < 7) documentValues.push(0);

    function draw(values, names) {
      const maxValue = Math.max(...values, 0);
      const max = maxValue || 1;
      const scaleLabels = $$(".y-labels span");
      scaleLabels.forEach((label, index) => {
        label.textContent = String(Math.round(maxValue * (1 - index / Math.max(scaleLabels.length - 1, 1))));
      });
      bars.forEach((bar, index) => {
        const value = values[index] || 0;
        bar.style.height = value ? `${Math.max(10, (value / max) * 100)}%` : "10px";
        bar.title = String(value);
        if (labels[index]) labels[index].textContent = names[index] || "—";
      });
    }

    function select(mode) {
      tabs.forEach((tab) => tab.classList.toggle("selected", tab.dataset.chart === mode));
      if (mode === "queries") {
        draw(queryValues, ["6d", "5d", "4d", "3d", "2d", "1d", "Today"]);
      } else {
        draw(documentValues, docs.slice(0, 7).map((doc) => (doc.name || "Doc").slice(0, 10)));
      }
    }

    tabs.forEach((tab) => {
      const mode = tab.textContent.trim().toLowerCase();
      tab.dataset.chart = mode;
      tab.onclick = () => select(mode);
    });
    select("documents");
  }

  function renderDocumentTypes(data) {
    const legend = $(".legend");
    const donut = $(".donut");
    if (!legend || !donut) return;
    const entries = Object.entries(data.document_types || {});
    if (!entries.length) {
      donut.style.background = "conic-gradient(#9da8c0 0 100%)";
      legend.innerHTML = '<li><span class="dot other"></span>None <b>0</b></li>';
      return;
    }
    const total = entries.reduce((sum, [, value]) => sum + Number(value || 0), 0);
    const palette = ["#5278f3", "#35ae83", "#9166e9", "#ffc05b", "#9da8c0"];
    let cursor = 0;
    const stops = [];
    legend.innerHTML = entries.map(([type, count], index) => {
      const end = cursor + (Number(count || 0) / total) * 100;
      stops.push(`${palette[index % palette.length]} ${cursor}% ${end}%`);
      cursor = end;
      return `<li><span class="dot" style="background:${palette[index % palette.length]}"></span>${escapeHtml(type)} <b>${escapeHtml(count)}</b></li>`;
    }).join("");
    donut.style.background = `conic-gradient(${stops.join(",")})`;
  }

  function bindSearch() {
    const input = $(".search input");
    if (!input || input.dataset.bound) return;
    input.dataset.bound = "1";
    input.addEventListener("input", () => {
      const term = input.value.trim().toLowerCase();
      $$(".table-row").forEach((row) => { row.hidden = Boolean(term) && !row.textContent.toLowerCase().includes(term); });
      $(".history-item, .library-item").forEach((item) => { item.hidden = Boolean(term) && !item.textContent.toLowerCase().includes(term); });
    });
  }

  function renderDashboard(data) {
    const cards = $$(".stat-card strong");
    const values = [data.documents ?? "—", data.chunks ?? "—", data.questions ?? "—", data.healthy ? "Healthy" : "Unavailable"];
    cards.forEach((card, index) => { card.textContent = values[index]; });

    const rows = $(".table");
    if (rows) {
      $$(".table-row, .empty-row").forEach((row) => row.remove());
      const docs = data.documents_list || [];
      if (!docs.length) rows.insertAdjacentHTML("beforeend", '<div class="table-row empty-row"><span>No documents indexed yet</span></div>');
      docs.slice(0, 8).forEach((doc) => rows.insertAdjacentHTML(
        "beforeend",
        `<div class="table-row${(localStorage.getItem("rag-last-uploaded") === (doc.name || doc.path || "Unnamed document")) ? " uploaded-highlight" : ""}"><span>${escapeHtml(doc.name || doc.path || "Unnamed document")}</span><span>${escapeHtml(doc.type || "OTHER")}</span><span>—</span><span>${escapeHtml(doc.date_added || "—")}</span><span class="indexed">● ${escapeHtml(doc.status || "Indexed")}</span></div>`
      ));
    }

    renderChart(data);
    renderDocumentTypes(data);
    const history = readHistory();
    if (history.length && $(".source-list") && !$(".source-list").children.length) renderSources(history[0].sources || []);
    bindSearch();
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
    }[character]));
  }

  async function loadDashboard() {
    try {
      const response = await fetch("/api/dashboard", { headers: { Accept: "application/json" }, credentials: "same-origin", cache: "no-store" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || "Dashboard unavailable");
      renderDashboard(data);
    } catch (error) {
      $$(".stat-card strong").forEach((card) => { card.textContent = "—"; });
      if ($(".table") && !$(".empty-row")) $(".table").insertAdjacentHTML("beforeend", '<div class="table-row empty-row"><span>Live document data unavailable</span></div>');
      console.warn(error.message);
    }
  }

  $("#theme")?.addEventListener("click", () => {
    localStorage.setItem("rag-theme", document.body.classList.contains("dark-preview") ? "light" : "dark");
    applyTheme();
  });

  $("#query-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = $("#question");
    const error = $("#error");
    const answerCard = $("#answer-card");
    const submit = event.currentTarget.querySelector('button[type="submit"]');
    const query = input.value.trim();
    if (!query) return;
    if (error) error.hidden = true;
    if (answerCard) answerCard.hidden = true;
    if (submit) { submit.disabled = true; submit.textContent = "…"; }

    try {
      let token = csrfToken();
      if (!token) {
        try { await fetch("/api/health", { credentials: "same-origin", cache: "no-store" }); } catch (_) {}
        token = csrfToken();
      }
      if (!token) throw new Error("Security token could not be initialized. Refresh the page and try again.");

      const response = await fetch("/api/query", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": token },
        body: JSON.stringify({ query })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || "The query could not be completed.");

      const answer = data.response_text || data.answer || "No response returned.";
      const sources = data.sources || [];
      if ($("#answer")) $("#answer").textContent = answer;
      if ($("#sources")) $("#sources").innerHTML = sources.map((source) => `<li>${escapeHtml(typeof source === "string" ? source : JSON.stringify(source))}</li>`).join("");
      renderSources(sources);
      saveHistory(query, answer, sources);
      if (answerCard) answerCard.hidden = false;
      loadDashboard();
    } catch (err) {
      if (error) { error.textContent = err.message; error.hidden = false; }
    } finally {
      if (submit) { submit.disabled = false; submit.textContent = "➤"; }
    }
  });

  applyTheme();
  applyProfile();
  bindSearch();
  clearDemoContent();
  loadDashboard();
  window.addEventListener("rag:dashboard-refresh", loadDashboard);
})();