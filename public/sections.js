(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const content = $('.content');
  if (!content) return;

  const dashboardParts = [
    $('.welcome'),
    $('.stats-grid'),
    $('.dashboard-grid')
  ];

  let sectionView = $('#section-view');
  if (!sectionView) {
    sectionView = document.createElement('div');
    sectionView.id = 'section-view';
    content.insertBefore(sectionView, content.querySelector('footer'));
  }

  const csrfToken = () => document.cookie.split('; ').find((entry) => entry.startsWith('rag_csrf='))?.split('=').slice(1).join('=') || '';

  async function ensureCsrfToken() {
    if (csrfToken()) return csrfToken();
    try {
      await fetch('/api/health', { credentials: 'same-origin', cache: 'no-store' });
    } catch (_) {}
    return csrfToken();
  }

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>'"]/g, (character) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
    }[character]));
  }

  function setActive(name) {
    $$('.nav-item').forEach((item) => item.classList.toggle('active', item.dataset.section === name));
  }

  function showDashboard() {
    dashboardParts.forEach((part) => { if (part) part.hidden = false; });
    sectionView.hidden = true;
    sectionView.innerHTML = '';
    setActive('Dashboard');
    window.dispatchEvent(new CustomEvent('rag:dashboard-refresh'));
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function page(title, subtitle, body) {
    return `<div class="section-page">
      <div class="section-page-header">
        <div><div class="section-eyebrow">RAG DIGITAL TWIN</div><h1>${escapeHtml(title)}</h1><p>${escapeHtml(subtitle)}</p></div>
        <button class="link-btn" data-action="dashboard">← Dashboard</button>
      </div>
      ${body}
    </div>`;
  }

  const templates = {
    'Document Library': () => page('Document Library', 'Browse documents indexed in this temporary browser session.',
      '<div class="section-card"><div class="section-card-heading"><h2>Indexed documents</h2><span class="session-badge">Temporary session</span></div><div id="library-list" class="library-list"><div class="loading-state">Loading your documents…</div></div></div>'),

    'Query': () => page('Query', 'Ask questions and receive grounded answers with source citations.',
      '<div class="section-card"><form id="section-query-form" class="section-query"><label for="section-question">Your question</label><textarea id="section-question" placeholder="What would you like to know about your documents?" required maxlength="4000"></textarea><div class="query-actions"><span>Answers are grounded in your indexed documents.</span><button class="primary-action" type="submit">Ask Digital Twin →</button></div></form><div id="section-query-result" class="section-result" hidden></div></div>'),

    'Sources': () => page('Sources', 'Source citations returned by your most recent queries.',
      '<div class="section-card"><div id="sources-page-list" class="source-page-list"></div></div>'),

    'History': () => page('Query History', 'Recent questions and answers stored for this browser session.',
      '<div class="section-card"><div id="history-list" class="history-list"></div></div>'),

    'Status': () => page('System Status', 'Live health and configuration information for the RAG pipeline.',
      '<div class="status-grid"><div class="section-card status-card"><span class="status-label">API</span><strong id="status-api">Checking…</strong><p>FastAPI application health</p></div><div class="section-card status-card"><span class="status-label">Vector Store</span><strong id="status-vector">Checking…</strong><p>Temporary FAISS session</p></div><div class="section-card status-card"><span class="status-label">Embedding</span><strong id="status-embedding">Checking…</strong><p>NVIDIA Nemotron</p></div><div class="section-card status-card"><span class="status-label">LLM</span><strong id="status-llm">Checking…</strong><p>NVIDIA-hosted model</p></div></div><div class="section-card" id="status-details">Loading detailed status…</div>'),

    'Settings': () => page('Settings', 'Manage local preferences for this browser session.',
      '<div class="section-card"><h2>Profile</h2><p class="muted">This information stays in your browser and is not sent to the server.</p><label for="profile-name">Display name</label><div class="settings-row"><input id="profile-name" class="settings-input" maxlength="60" placeholder="Your name"/><button class="primary-action" id="save-profile">Save profile</button></div><div class="section-note" id="profile-note"></div></div><div class="section-card"><h2>Appearance</h2><p class="muted">Choose how the workspace looks on this device.</p><button class="secondary-action" id="settings-theme">Toggle light / dark mode</button></div><div class="section-card"><h2>Session</h2><p class="muted">Documents and vector indexes are temporary and disappear when the browser session ends or the server session expires.</p></div>'),

    'Documentation': () => page('Documentation', 'Understand the architecture and how to use the demo.',
      '<div class="section-card docs-grid"><div><h2>1. Ingest</h2><p>Upload a PDF or TXT document. It is chunked, embedded with NVIDIA Nemotron, and indexed into session-only FAISS.</p></div><div><h2>2. Query</h2><p>Ask a question and the retriever selects relevant chunks before the NVIDIA-hosted LLM generates a grounded answer.</p></div><div><h2>3. Inspect</h2><p>Use Library, Sources, History, and Status to inspect what happened during the current session.</p></div></div>'),

    'Developer Tools': () => page('Developer Tools', 'Inspect the live configuration of this portfolio demo.',
      '<div class="section-card docs-grid"><div><h2>API</h2><p>FastAPI serverless entrypoint with session-based FAISS storage.</p></div><div><h2>Embeddings</h2><p>NVIDIA <code>nvidia/nemotron-3-embed-1b</code>, 2048-dimensional vectors.</p></div><div><h2>Generation</h2><p>NVIDIA-hosted <code>openai/gpt-oss-20b</code> through the OpenAI-compatible API.</p></div><div><h2>Security</h2><p>CSRF protection, rate limiting, upload validation, secure session cookies, security headers, and redacted logs.</p></div></div><button class="primary-action" data-action="status">Open live status →</button>'),

    'Ingest Documents': () => page('Ingest Documents', 'Upload PDF or TXT files and index them for this temporary browser session.',
      '<div class="section-card"><h2>Select document</h2><form id="ingest-form" class="ingest-form"><input id="document-file" name="file" type="file" accept=".pdf,.txt,application/pdf,text/plain" required/><p class="muted">PDF and TXT only · maximum 10 MB · maximum 5 unique documents per session.</p><button type="submit" class="primary-action">Upload and Index →</button></form><div id="ingest-result" class="message" hidden></div></div>')
  };

  async function renderLibrary() {
    const target = $('#library-list');
    try {
      const response = await fetch('/api/dashboard', { credentials: 'same-origin', cache: 'no-store' });
      const data = await response.json();
      const docs = data.documents_list || [];
      if (!docs.length) {
        target.innerHTML = '<div class="empty-state"><strong>No documents indexed yet.</strong><p>Go to Ingest Documents to add your first PDF or TXT file.</p><button class="primary-action" data-action="ingest">Add a document →</button></div>';
        return;
      }
      target.innerHTML = docs.map((doc) => `<div class="library-item"><div><strong>${escapeHtml(doc.name)}</strong><small>${escapeHtml(doc.type)} · ${escapeHtml(doc.chunks)} chunks · ${escapeHtml(doc.date_added || 'Current session')}</small></div><span class="indexed">● ${escapeHtml(doc.status || 'Indexed')}</span></div>`).join('');
    } catch (_) {
      target.innerHTML = '<div class="empty-state">Live document data is unavailable right now.</div>';
    }
  }

  function readHistory() {
    try { return JSON.parse(localStorage.getItem('rag-history') || '[]'); } catch (_) { return []; }
  }

  function renderHistory() {
    const target = $('#history-list');
    const history = readHistory();
    target.innerHTML = history.length ? history.map((item, index) => `<article class="history-item"><div class="history-meta"><span>Query ${history.length - index}</span><small>${escapeHtml(item.time)}</small></div><h3>${escapeHtml(item.query)}</h3><p>${escapeHtml(item.answer)}</p></article>`).join('') : '<div class="empty-state"><strong>No queries yet.</strong><p>Ask a question from the Query section to create your first history entry.</p><button class="primary-action" data-action="query">Ask a question →</button></div>';
  }

  function renderSources() {
    const target = $('#sources-page-list');
    const sources = readHistory()[0]?.sources || [];
    target.innerHTML = sources.length ? sources.map((source, index) => {
      const text = typeof source === 'string' ? source : (source.source_file || source.file || source.name || JSON.stringify(source));
      return `<div class="source-item"><span>${index + 1}</span><div><strong>${escapeHtml(text)}</strong><small>Returned by the latest grounded query</small></div></div>`;
    }).join('') : '<div class="empty-state"><strong>No sources yet.</strong><p>Run a query after indexing a document and its citations will appear here.</p></div>';
  }

  async function renderStatus() {
    try {
      const [healthResponse, dashboardResponse] = await Promise.all([
        fetch('/api/health', { credentials: 'same-origin', cache: 'no-store' }),
        fetch('/api/dashboard', { credentials: 'same-origin', cache: 'no-store' })
      ]);
      const health = await healthResponse.json();
      const dashboard = await dashboardResponse.json();
      $('#status-api').textContent = healthResponse.ok && health.healthy ? 'Operational' : 'Degraded';
      $('#status-vector').textContent = `${dashboard.chunks || 0} vectors`;
      $('#status-embedding').textContent = 'NVIDIA Nemotron';
      const llm = dashboard.components_status?.response_generator || dashboard.components_status?.llm;
      $('#status-llm').textContent = llm ? String(llm) : 'NVIDIA hosted';
      $('#status-details').innerHTML = `<div class="status-details-grid"><div><span>Documents</span><strong>${dashboard.documents || 0}</strong></div><div><span>Chunks</span><strong>${dashboard.chunks || 0}</strong></div><div><span>Questions</span><strong>${dashboard.questions || 0}</strong></div><div><span>Session mode</span><strong>Temporary</strong></div><div><span>Health</span><strong>${escapeHtml(health.health || 'ready')}</strong></div><div><span>Uptime</span><strong>${escapeHtml(dashboard.uptime_seconds || 0)}s</strong></div></div>`;
    } catch (_) {
      $('#status-api').textContent = 'Unavailable';
      $('#status-details').textContent = 'Unable to load live status.';
    }
  }

  async function handleQuery(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const input = $('#section-question');
    const result = $('#section-query-result');
    const button = form.querySelector('button[type="submit"]');
    const query = input.value.trim();
    if (!query) return;
    button.disabled = true;
    button.textContent = 'Thinking…';
    result.hidden = false;
    result.innerHTML = '<div class="loading-state">Retrieving relevant context and generating an answer…</div>';
    try {
      const token = await ensureCsrfToken();
      if (!token) throw new Error('Security token could not be initialized. Refresh the page and try again.');
      const response = await fetch('/api/query', {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': token },
        body: JSON.stringify({ query })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || 'Query failed.');
      const answer = data.response_text || data.answer || 'No answer returned.';
      const sources = data.sources || [];
      const history = readHistory();
      history.unshift({ query, answer, sources, time: new Date().toLocaleString() });
      localStorage.setItem('rag-history', JSON.stringify(history.slice(0, 20)));
      result.innerHTML = `<div class="answer-header"><span>Grounded answer</span><span>${escapeHtml(data.model_used || 'NVIDIA')}</span></div><div class="answer-text">${escapeHtml(answer)}</div><div class="answer-meta">Confidence: ${data.confidence_score == null ? '—' : escapeHtml(Number(data.confidence_score).toFixed(3))} · Context used: ${data.context_used ? 'Yes' : 'No'}</div><div class="result-sources"><strong>Sources</strong>${sources.length ? sources.map((source) => `<span>${escapeHtml(typeof source === 'string' ? source : (source.source_file || source.name || JSON.stringify(source)))}</span>`).join('') : '<span>None returned</span>'}</div>`;
      window.dispatchEvent(new CustomEvent('rag:dashboard-refresh'));
    } catch (error) {
      result.innerHTML = `<div class="message error">${escapeHtml(error.message)}</div>`;
    } finally {
      button.disabled = false;
      button.textContent = 'Ask Digital Twin →';
    }
  }

  async function handleIngest(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const file = $('#document-file')?.files?.[0];
    const result = $('#ingest-result');
    const button = form.querySelector('button[type="submit"]');
    if (!file) return;
    button.disabled = true;
    button.textContent = 'Indexing…';
    result.hidden = true;
    const body = new FormData();
    body.append('file', file);
    try {
      const token = await ensureCsrfToken();
      if (!token) throw new Error('Security token could not be initialized. Refresh the page and try again.');
      const response = await fetch('/api/ingest', {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'X-CSRF-Token': token },
        body
      });
      const raw = await response.text();
      let data;
      try { data = JSON.parse(raw); } catch (_) { throw new Error(raw || 'Upload failed.'); }
      if (!response.ok) throw new Error(data.detail || data.error || 'Upload failed.');
      result.className = 'message success';
      result.textContent = `${data.message} · ${data.chunks || 0} chunks indexed.`;
      result.hidden = false;
      form.reset();
      window.dispatchEvent(new CustomEvent('rag:dashboard-refresh'));
    } catch (error) {
      result.className = 'message error';
      result.textContent = error.message;
      result.hidden = false;
    } finally {
      button.disabled = false;
      button.textContent = 'Upload and Index →';
    }
  }

  function bindSection(name) {
    if (name === 'Document Library') renderLibrary();
    if (name === 'Sources') renderSources();
    if (name === 'History') renderHistory();
    if (name === 'Status') renderStatus();

    if (name === 'Query') {
      $('#section-query-form')?.addEventListener('submit', handleQuery);
    }

    if (name === 'Ingest Documents') {
      $('#ingest-form')?.addEventListener('submit', handleIngest);
    }

    if (name === 'Settings') {
      const input = $('#profile-name');
      input.value = localStorage.getItem('rag-user-name') || '';
      $('#save-profile')?.addEventListener('click', () => {
        const value = input.value.trim() || 'User';
        localStorage.setItem('rag-user-name', value);
        const initials = value.split(/\s+/).filter(Boolean).map((part) => part[0]).join('').slice(0, 2).toUpperCase();
        $('.profile').textContent = initials || 'U';
        $('.welcome h1').textContent = `Welcome back, ${value}!`;
        $('#profile-note').textContent = 'Profile saved on this device.';
      });
      $('#settings-theme')?.addEventListener('click', () => $('#theme')?.click());
    }
  }

  function showSection(name) {
    if (name === 'Dashboard') return showDashboard();
    dashboardParts.forEach((part) => { if (part) part.hidden = true; });
    sectionView.innerHTML = templates[name] ? templates[name]() : page(name, 'Workspace tools', '<div class="section-card"><p>This section is not configured.</p></div>');
    sectionView.hidden = false;
    setActive(name);
    bindSection(name);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  document.addEventListener('click', (event) => {
    const target = event.target.closest('[data-section], [data-action]');
    if (!target) return;
    const section = target.dataset.section || target.dataset.action;
    if (!section) return;
    if (section === 'dashboard') showDashboard();
    else if (section === 'ingest') showSection('Ingest Documents');
    else if (section === 'query') showSection('Query');
    else if (section === 'status') showSection('Status');
    else if (section === 'Documentation') showSection('Documentation');
    else if (section === 'Developer Tools') showSection('Developer Tools');
    else showSection(section);
  });

  document.querySelector('.recent .link-btn')?.addEventListener('click', () => showSection('Document Library'));

  window.addEventListener('rag:dashboard-refresh', () => {
    if (!sectionView.hidden) return;
    // app.js owns the dashboard data refresh.
  });
})();