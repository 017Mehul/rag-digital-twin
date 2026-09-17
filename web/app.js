const form = document.querySelector('#query-form');
const question = document.querySelector('#question');
const submit = document.querySelector('#submit');
const status = document.querySelector('#status');
const error = document.querySelector('#error');
const card = document.querySelector('#answer-card');

function setBusy(busy) {
  submit.disabled = busy;
  submit.innerHTML = busy ? 'Thinking…' : 'Ask Digital Twin <span>↗</span>';
  status.innerHTML = busy ? '<span></span> Processing' : '<span></span> Ready';
}

async function readResponse(response) {
  const raw = await response.text();
  let data;
  try {
    data = raw ? JSON.parse(raw) : {};
  } catch {
    throw new Error(`Server returned ${response.status}: ${raw.slice(0, 240) || 'empty response'}`);
  }
  if (!response.ok) {
    throw new Error(data.detail || data.error || `Request failed with status ${response.status}.`);
  }
  return data;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const query = question.value.trim();
  if (!query) return;
  error.hidden = true;
  card.hidden = true;
  setBusy(true);

  try {
    const response = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    const data = await readResponse(response);

    document.querySelector('#answer').textContent = data.response_text || 'No response returned.';
    document.querySelector('#time').textContent = data.generation_time ? `${Number(data.generation_time).toFixed(2)}s` : '';
    document.querySelector('#model').textContent = data.model_used ? `Model: ${data.model_used}` : '';
    document.querySelector('#confidence').textContent = data.confidence_score != null ? `Confidence: ${Math.round(data.confidence_score * 100)}%` : '';
    const sources = document.querySelector('#sources');
    sources.replaceChildren();
    (data.sources || []).forEach((source) => { const li = document.createElement('li'); li.textContent = source; sources.appendChild(li); });
    card.hidden = false;
  } catch (err) {
    error.textContent = err.message;
    error.hidden = false;
  } finally {
    setBusy(false);
  }
});
