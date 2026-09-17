(() => {
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];

  function applyTheme() {
    const dark = localStorage.getItem('rag-theme') === 'dark';
    document.body.classList.toggle('dark-preview', dark);
    const button = $('#theme');
    if (button) {
      button.textContent = dark ? '☀' : '☼';
      button.title = dark ? 'Switch to light mode' : 'Switch to dark mode';
      button.setAttribute('aria-label', button.title);
    }
  }

  function applyProfile() {
    const name = localStorage.getItem('rag-user-name') || 'Mehul';
    const initials = name.split(/\s+/).filter(Boolean).map((part) => part[0]).join('').slice(0, 2).toUpperCase();
    if ($('.profile')) $('.profile').textContent = initials || 'U';
    if ($('.welcome h1')) $('.welcome h1').textContent = `Welcome back, ${name}!`;
  }

  function clearDemoContent() {
    $$('.table-row').forEach((row) => row.remove());
    $('.source-list')?.replaceChildren();
    if ($('.source-title')) $('.source-title').textContent = 'Sources (0)';
    if ($('#sample-answer')) $('#sample-answer').textContent = 'Ask a question to see a grounded answer from your indexed documents.';
  }

  function renderDashboard(data) {
    const cards = $$('.stat-card strong');
    const values = [data.documents ?? '—', data.chunks ?? '—', data.questions ?? '—', data.healthy ? 'Healthy' : 'Unavailable'];
    cards.forEach((card, index) => { card.textContent = values[index]; });
    if ($('.live')) $('.live').innerHTML = `<i></i> ${data.healthy ? 'Live' : 'Degraded'}`;
    const rows = $('.table');
    if (rows) {
      $$('.table-row, .empty-row').forEach((row) => row.remove());
      const docs = data.documents_list || [];
      if (!docs.length) rows.insertAdjacentHTML('beforeend', '<div class="table-row empty-row"><span>No documents indexed yet</span></div>');
      docs.slice(0, 8).forEach((doc) => rows.insertAdjacentHTML('beforeend', `<div class="table-row"><span>${escapeHtml(doc.name || doc.path || 'Unnamed document')}</span><span>${escapeHtml(doc.type || 'OTHER')}</span><span>—</span><span>${escapeHtml(doc.date_added || '—')}</span><span class="indexed">● ${escapeHtml(doc.status || 'Indexed')}</span></div>`));
    }
  }

  function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[character])); }

  async function loadDashboard() {
    try {
      const response = await fetch('/api/dashboard', { headers: { Accept: 'application/json' } });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || 'Dashboard unavailable');
      renderDashboard(data);
    } catch (error) {
      $$('.stat-card strong').forEach((card) => { card.textContent = '—'; });
      if ($('.live')) $('.live').innerHTML = '<i></i> Offline';
      if ($('.table') && !$('.empty-row')) $('.table').insertAdjacentHTML('beforeend', '<div class="table-row empty-row"><span>Live document data unavailable</span></div>');
      console.warn(error.message);
    }
  }

  $('#theme')?.addEventListener('click', () => {
    localStorage.setItem('rag-theme', document.body.classList.contains('dark-preview') ? 'light' : 'dark');
    applyTheme();
  });

  $('#query-form')?.addEventListener('submit', async (event) => {
    event.preventDefault();
    const input = $('#question');
    const error = $('#error');
    const answerCard = $('#answer-card');
    const submit = event.currentTarget.querySelector('button[type="submit"]');
    const query = input.value.trim();
    if (!query) return;
    if (error) error.hidden = true;
    if (answerCard) answerCard.hidden = true;
    if (submit) { submit.disabled = true; submit.textContent = '…'; }
    try {
      const response = await fetch('/api/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || 'The query could not be completed.');
      if ($('#answer')) $('#answer').textContent = data.response_text || data.answer || 'No response returned.';
      if ($('#sources')) $('#sources').innerHTML = (data.sources || []).map((source) => `<li>${escapeHtml(typeof source === 'string' ? source : JSON.stringify(source))}</li>`).join('');
      if (answerCard) answerCard.hidden = false;
      loadDashboard();
    } catch (err) { if (error) { error.textContent = err.message; error.hidden = false; } }
    finally { if (submit) { submit.disabled = false; submit.textContent = '➤'; } }
  });

  applyTheme();
  applyProfile();
  clearDemoContent();
  loadDashboard();

  const sectionsScript = document.createElement('script');
  sectionsScript.src = '/sections.js';
  sectionsScript.defer = false;
  document.head.appendChild(sectionsScript);
})();
