(() => {
  const content = document.querySelector('.content');
  if (!content) return;

  const dashboardParts = [
    document.querySelector('.welcome'),
    document.querySelector('.stats-grid'),
    document.querySelector('.dashboard-grid'),
    content.querySelector('footer')
  ];
  let sectionView = document.querySelector('#section-view');
  if (!sectionView) {
    sectionView = document.createElement('div');
    sectionView.id = 'section-view';
    sectionView.hidden = true;
    content.insertBefore(sectionView, content.querySelector('footer'));
  }

  const templates = {
    'Ingest Documents': `<div class="section-page"><h1>Ingest Documents</h1><p>Upload documents to add them to your RAG knowledge base.</p><div class="section-card"><h2>Upload documents</h2><input type="file" id="document-upload" multiple accept=".pdf,.txt,.md,.docx"/><p class="muted">Supported formats: PDF, TXT, Markdown and DOCX.</p><button class="primary-action" id="upload-placeholder">Upload and index</button><div class="section-note" id="upload-note">Connect this control to the ingestion endpoint to index files.</div></div></div>`,
    'Document Library': `<div class="section-page"><h1>Document Library</h1><p>Browse and manage documents indexed in your knowledge base.</p><div class="section-card"><h2>Indexed documents</h2><div id="library-list">Loading live documents…</div></div></div>`,
    'Query': `<div class="section-page"><h1>Query your knowledge base</h1><p>Ask a grounded question about your indexed documents.</p><div class="section-card"><form id="section-query-form" class="section-query"><textarea id="section-question" placeholder="Ask a question…" required></textarea><button class="primary-action" type="submit">Ask Digital Twin</button></form><div id="section-query-result" class="section-result"></div></div></div>`,
    'Sources': `<div class="section-page"><h1>Sources</h1><p>Sources are displayed after a successful query.</p><div class="section-card"><div id="sources-page-list">No sources available yet. Ask a question first.</div></div></div>`,
    'History': `<div class="section-page"><h1>Query History</h1><p>Your recent questions for this browser session.</p><div class="section-card"><div id="history-list">No query history yet.</div></div></div>`,
    'Status': `<div class="section-page"><h1>System Status</h1><p>Check the health of your RAG services.</p><div class="section-card"><div class="status-row"><span>Dashboard API</span><strong id="status-dashboard">Checking…</strong></div><div class="status-row"><span>Query API</span><strong id="status-query">Available when queried</strong></div><div class="status-row"><span>Vector store</span><strong>Reported by backend</strong></div></div></div>`,
    'Settings': `<div class="section-page"><h1>Settings</h1><p>Personalize your workspace.</p><div class="section-card"><label for="profile-name">Display name</label><input id="profile-name" class="settings-input" placeholder="Enter your name"/><button class="primary-action" id="save-profile">Save profile</button><div class="section-note" id="profile-note"></div></div><div class="section-card"><h2>Appearance</h2><button class="primary-action" id="settings-theme">Toggle dark / light mode</button></div></div>`
  };

  function showDashboard() {
    dashboardParts.forEach((part) => { if (part) part.hidden = false; });
    sectionView.hidden = true;
    document.querySelectorAll('.nav-item').forEach((item) => item.classList.toggle('active', item.dataset.section === 'Dashboard'));
  }

  function showSection(name) {
    if (name === 'Dashboard') return showDashboard();
    dashboardParts.forEach((part) => { if (part) part.hidden = true; });
    sectionView.innerHTML = templates[name] || `<div class="section-page"><h1>${name}</h1><p>This workspace section is being prepared.</p></div>`;
    sectionView.hidden = false;
    document.querySelectorAll('.nav-item').forEach((item) => item.classList.toggle('active', item.dataset.section === name));
    bindSection(name);
  }

  function bindSection(name) {
    if (name === 'Settings') {
      const input = document.querySelector('#profile-name');
      input.value = localStorage.getItem('rag-user-name') || '';
      document.querySelector('#save-profile')?.addEventListener('click', () => {
        const value = input.value.trim() || 'User';
        localStorage.setItem('rag-user-name', value);
        document.querySelector('.profile').textContent = value.split(/\s+/).map((part) => part[0]).join('').slice(0, 2).toUpperCase();
        document.querySelector('.welcome h1').textContent = `Welcome back, ${value}!`;
        document.querySelector('#profile-note').textContent = 'Profile saved on this device.';
      });
      document.querySelector('#settings-theme')?.addEventListener('click', () => document.querySelector('#theme')?.click());
    }
    if (name === 'Status') {
      fetch('/api/dashboard').then((response) => { document.querySelector('#status-dashboard').textContent = response.ok ? 'Operational' : 'Unavailable'; }).catch(() => { document.querySelector('#status-dashboard').textContent = 'Unavailable'; });
    }
    if (name === 'Document Library') {
      fetch('/api/dashboard').then((response) => response.json()).then((data) => {
        const list = document.querySelector('#library-list');
        const docs = data.documents_list || [];
        list.innerHTML = docs.length ? docs.map((doc) => `<div class="library-item">${escapeHtml(doc.name || doc.path || 'Unnamed document')} <span>${escapeHtml(doc.status || 'Indexed')}</span></div>`).join('') : 'No documents indexed yet.';
      }).catch(() => { document.querySelector('#library-list').textContent = 'Live document data unavailable.'; });
    }
    if (name === 'Query') {
      document.querySelector('#section-query-form')?.addEventListener('submit', async (event) => {
        event.preventDefault();
        const query = document.querySelector('#section-question').value.trim();
        const result = document.querySelector('#section-query-result');
        result.textContent = 'Searching…';
        try {
          const response = await fetch('/api/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query }) });
          const data = await response.json();
          if (!response.ok) throw new Error(data.detail || data.error || 'Query failed');
          result.textContent = data.response_text || data.answer || 'No answer returned.';
          const history = JSON.parse(localStorage.getItem('rag-history') || '[]');
          history.unshift({ query, answer: result.textContent, time: new Date().toLocaleString() });
          localStorage.setItem('rag-history', JSON.stringify(history.slice(0, 20)));
        } catch (error) { result.textContent = error.message; }
      });
    }
    if (name === 'History') {
      const history = JSON.parse(localStorage.getItem('rag-history') || '[]');
      document.querySelector('#history-list').innerHTML = history.length ? history.map((item) => `<div class="history-item"><b>${escapeHtml(item.query)}</b><small>${escapeHtml(item.time)}</small><p>${escapeHtml(item.answer)}</p></div>`).join('') : 'No query history yet.';
    }
    if (name === 'Ingest Documents') document.querySelector('#upload-placeholder')?.addEventListener('click', () => { document.querySelector('#upload-note').textContent = 'Upload endpoint connection is required before files can be indexed.'; });
  }

  function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[character])); }

  document.querySelectorAll('[data-section]').forEach((button) => {
    button.addEventListener('click', (event) => { event.preventDefault(); event.stopImmediatePropagation(); showSection(button.dataset.section); }, true);
  });
})();
