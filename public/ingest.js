(() => {
  const content = document.querySelector('.content');
  const originalContent = content?.innerHTML;
  const navItems = [...document.querySelectorAll('[data-section]')];

  function setActive(name) {
    navItems.forEach((item) => item.classList.toggle('active', item.dataset.section === name));
  }

  function showDashboard() {
    if (originalContent) content.innerHTML = originalContent;
    setActive('Dashboard');
    window.location.reload();
  }

  function showIngest() {
    setActive('Ingest Documents');
    content.innerHTML = `<div class="panel ingest-screen" style="max-width:900px;margin:0 auto;padding:32px"><div class="panel-heading"><div><h1>Ingest Documents</h1><p>Upload PDF, TXT, MD, or DOCX files to index them in your RAG pipeline.</p></div><button class="link-btn" id="back-dashboard">← Dashboard</button></div><form id="ingest-form" style="margin-top:28px;display:grid;gap:18px"><label for="document-file"><b>Select document</b></label><input id="document-file" name="file" type="file" accept=".pdf,.txt,.md,.docx" required /><button type="submit" class="primary-button">Upload and Index</button></form><div id="ingest-result" class="message" hidden style="margin-top:18px"></div></div>`;
    document.querySelector('#back-dashboard')?.addEventListener('click', showDashboard);
    document.querySelector('#ingest-form')?.addEventListener('submit', async (event) => {
      event.preventDefault();
      const form = event.currentTarget;
      const result = document.querySelector('#ingest-result');
      const button = form.querySelector('button[type="submit"]');
      const file = document.querySelector('#document-file')?.files?.[0];
      if (!file) return;
      button.disabled = true;
      button.textContent = 'Indexing…';
      result.hidden = true;
      const body = new FormData();
      body.append('file', file);
      try {
        const response = await fetch('/api/ingest', { method: 'POST', body });
        const raw = await response.text();
        let data;
        try { data = JSON.parse(raw); } catch { throw new Error(raw || `Upload failed (${response.status})`); }
        if (!response.ok) throw new Error(data.detail || data.error || 'Upload failed.');
        result.className = 'message success';
        result.textContent = `${data.message} Chunks: ${data.chunks ?? 0}.`;
        result.hidden = false;
      } catch (error) {
        result.className = 'message error';
        result.textContent = error.message;
        result.hidden = false;
      } finally {
        button.disabled = false;
        button.textContent = 'Upload and Index';
      }
    });
  }

  navItems.forEach((item) => {
    item.addEventListener('click', (event) => {
      if (item.dataset.section === 'Ingest Documents') {
        event.stopImmediatePropagation();
        showIngest();
      }
    }, true);
  });
})();
