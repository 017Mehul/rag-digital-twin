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
      if (subtitle) {
        subtitle.textContent = subtitles[index];
        subtitle.classList.toggle('muted', index === 3);
      }
    });

    const live = document.querySelector('.live');
    if (live) live.innerHTML = `<i></i> ${data.healthy ? 'Live' : 'Degraded'}`;

    const rows = document.querySelectorAll('.table-row');
    if (!data.documents_list?.length) {
      rows.forEach((row) => { row.innerHTML = '<span colspan="5">No documents indexed yet</span>'; });
    }
  } catch (err) {
    const cards = document.querySelectorAll('.stat-card strong');
    cards.forEach((card) => { card.textContent = '—'; });
    const live = document.querySelector('.live');
    if (live) live.innerHTML = '<i></i> Offline';
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
    document.querySelector('#answer').textContent = data.response_text || data.answer || 'No response returned.';
    document.querySelector('#time').textContent = data.generation_time ? `${Number(data.generation_time).toFixed(2)}s` : '';
    document.querySelector('#model').textContent = data.model_used ? `Model: ${data.model_used}` : '';
    document.querySelector('#confidence').textContent = data.confidence_score != null ? `Confidence: ${Math.round(data.confidence_score * 100)}%` : '';
    const sources = document.querySelector('#sources');
    if (sources) {
      sources.replaceChildren();
      (data.sources || []).forEach((source) => { const li = document.createElement('li'); li.textContent = typeof source === 'string' ? source : JSON.stringify(source); sources.appendChild(li); });
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

document.querySelector('#theme')?.addEventListener('click', () => {
  document.body.classList.toggle('dark-preview');
});

loadDashboardMetrics();
