const form = document.querySelector('#query-form');
const question = document.querySelector('#question');
const submit = form?.querySelector('button[type="submit"]');
const error = document.querySelector('#error');
const answerCard = document.querySelector('#answer-card');

function setBusy(busy) {
  if (!submit) return;
  submit.disabled = busy;
  submit.textContent = busy ? '…' : '➤';
}

function setText(selector, value) {
  const element = document.querySelector(selector);
  if (element) element.textContent = value;
}

function clearMockContent() {
  document.querySelectorAll('.table-row').forEach((row) => row.remove());
  document.querySelector('.source-list')?.replaceChildren();
  const sourceTitle = document.querySelector('.source-title');
  if (sourceTitle) sourceTitle.textContent = 'Sources (0)';
  const sampleAnswer = document.querySelector('#sample-answer');
  if (sampleAnswer) sampleAnswer.innerHTML = '<span class="muted">Ask a question to see a grounded answer from your indexed documents.</span>';
  document.querySelectorAll('.bars > div').forEach((bar) => {
    bar.style.height = '0%';
    const label = bar.querySelector('em');
    if (label) label.textContent = '—';
  });
  const legend = document.querySelector('.legend');
  if (legend) legend.replaceChildren();
  const donut = document.querySelector('.donut');
  if (donut) donut.style.background = 'conic-gradient(#d9deeb 0 100%)';
}

function formatDate(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value).slice(0, 10) : date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

function renderDocuments(documents) {
  const table = document.querySelector('.table');
  if (!table) return;
  table.querySelectorAll('.table-row, .empty-row').forEach((row) => row.remove());
  if (!documents?.length) {
    const empty = document.createElement('div');
    empty.className = 'table-row empty-row';
    empty.innerHTML = '<span>No documents indexed yet</span>';
    table.appendChild(empty);
    return;
  }
  documents.slice(0, 8).forEach((doc) => {
    const row = document.createElement('div');
    row.className = 'table-row';
    row.innerHTML = `<span>▣ ${escapeHtml(doc.name || doc.path || 'Unnamed document')}</span><span>${escapeHtml(doc.type || 'OTHER')}</span><span>—</span><span>${escapeHtml(formatDate(doc.date_added))}</span><span class="indexed">● ${escapeHtml(doc.status || 'Indexed')}</span>`;
    table.appendChild(row);
  });
}

function renderTypes(typeCounts) {
  const entries = Object.entries(typeCounts || {}).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((sum, [, count]) => sum + count, 0);
  const palette = ['pdf', 'txt', 'docx', 'md', 'other'];
  const legend = document.querySelector('.legend');
  const donut = document.querySelector('.donut');
  if (!legend || !donut) return;
  legend.replaceChildren();
  if (!total) {
    donut.style.background = 'conic-gradient(#d9deeb 0 100%)';
    return;
  }
  let cursor = 0;
  const segments = [];
  entries.forEach(([type, count], index) => {
    const percentage = (count / total) * 100;
    const next = cursor + percentage;
    segments.push(`#${['5278ee', '36ad83', '8c62df', 'ffbd50', '9aa8c3'][index % 5]} ${cursor}% ${next}%`);
    cursor = next;
    const li = document.createElement('li');
    li.innerHTML = `<i class="dot ${palette[index % palette.length]}"></i>${escapeHtml(type)} <b>${Math.round(percentage)}%</b>`;
    legend.appendChild(li);
  });
  donut.style.background = `conic-gradient(${segments.join(', ')})`;
}

function renderMonthlyChart(documents) {
  const bars = [...document.querySelectorAll('.bars > div')];
  if (!bars.length) return;
  const now = new Date();
  const months = Array.from({ length: bars.length }, (_, index) => new Date(now.getFullYear(), now.getMonth() - (bars.length - 1 - index), 1));
  const counts = months.map((month) => documents.filter((doc) => {
    const date = new Date(doc.date_added);
    return !Number.isNaN(date.getTime()) && date.getFullYear() === month.getFullYear() && date.getMonth() === month.getMonth();
  }).length);
  const max = Math.max(...counts, 1);
  bars.forEach((bar, index) => {
    bar.style.height = `${(counts[index] / max) * 80}%`;
    const label = bar.querySelector('em');
    if (label) label.textContent = months[index].toLocaleDateString(undefined, { month: 'short' });
    bar.title = `${counts[index]} document(s)`;
  });
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[character]));
}

async function loadDashboardMetrics() {
  try {
    const response = await fetch('/api/dashboard', { headers: { Accept: 'application/json' } });
    const raw = await response.text();
    let data;
    try { data = JSON.parse(raw); } catch { throw new Error(raw || `Dashboard returned HTTP ${response.status}`); }
    if (!response.ok) throw new Error(data.detail || data.error || 'Dashboard metrics unavailable.');

    const cards = document.querySelectorAll('.stat-card');
    const values = [data.documents, data.chunks, data.questions, data.healthy ? 'Healthy' : 'Unavailable'];
    const subtitles = ['Live indexed documents', 'Live vector store size', 'Queries in this instance', data.healthy ? 'Pipeline operational' : 'Check API health'];
    cards.forEach((card, index) => {
      const value = card.querySelector('strong');
      const subtitle = card.querySelector('small');
      if (value) value.textContent = values[index] ?? '—';
      if (subtitle) subtitle.textContent = subtitles[index];
    });

    const live = document.querySelector('.live');
    if (live) live.innerHTML = `<i></i> ${data.healthy ? 'Live' : 'Degraded'}`;
    const documents = data.documents_list || [];
    renderDocuments(documents);
    renderTypes(data.document_types || {});
    renderMonthlyChart(documents);
  } catch (err) {
    document.querySelectorAll('.stat-card strong').forEach((card) => { card.textContent = '—'; });
    const live = document.querySelector('.live');
    if (live) live.innerHTML = '<i></i> Offline';
    const table = document.querySelector('.table');
    if (table && !table.querySelector('.empty-row')) {
      const empty = document.createElement('div');
      empty.className = 'table-row empty-row';
      empty.innerHTML = '<span>Live document data unavailable</span>';
      table.appendChild(empty);
    }
    console.warn('Dashboard metrics could not be loaded:', err.message);
  }
}

form?.addEventListener('submit', async (event) => {
  event.preventDefault();
  const query = question.value.trim();
  if (!query) return;
  if (error) error.hidden = true;
  if (answerCard) answerCard.hidden = true;
  setBusy(true);
  try {
    const response = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    const raw = await response.text();
    let data;
    try { data = JSON.parse(raw); } catch { throw new Error(raw || `Server returned HTTP ${response.status}`); }
    if (!response.ok) throw new Error(data.detail || data.error || 'The query could not be completed.');
    setText('#answer', data.response_text || data.answer || 'No response returned.');
    setText('#time', data.generation_time ? `${Number(data.generation_time).toFixed(2)}s` : '');
    setText('#model', data.model_used ? `Model: ${data.model_used}` : '');
    setText('#confidence', data.confidence_score != null ? `Confidence: ${Math.round(data.confidence_score * 100)}%` : '');
    const sources = document.querySelector('#sources');
    if (sources) {
      sources.replaceChildren();
      (data.sources || []).forEach((source) => {
        const li = document.createElement('li');
        li.textContent = typeof source === 'string' ? source : JSON.stringify(source);
        sources.appendChild(li);
      });
    }
    if (answerCard) answerCard.hidden = false;
    loadDashboardMetrics();
  } catch (err) {
    if (error) { error.textContent = err.message; error.hidden = false; }
  } finally { setBusy(false); }
});

document.querySelectorAll('[data-section]').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.nav-item').forEach((item) => item.classList.remove('active'));
    if (button.classList.contains('nav-item')) button.classList.add('active');
    const section = button.dataset.section;
    if (section === 'Query') question?.focus();
    else if (section !== 'Dashboard' && error) { error.textContent = `${section} view is ready to be connected to the backend.`; error.hidden = false; }
  });
});

document.querySelector('#theme')?.addEventListener('click', () => document.body.classList.toggle('dark-preview'));

clearMockContent();
loadDashboardMetrics();
