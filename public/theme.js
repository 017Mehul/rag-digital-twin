(() => {
  const root = document.documentElement;
  const savedTheme = localStorage.getItem('rag-theme');
  if (savedTheme === 'dark') document.body.classList.add('dark-preview');

  const style = document.createElement('style');
  style.textContent = `
    body.dark-preview { --ink:#eef2ff; --muted:#aab5d2; --line:#35415f; --bg:#0b1224; background:#0b1224; color:var(--ink); }
    body.dark-preview .main, body.dark-preview .content { background:#0b1224; }
    body.dark-preview .topbar { background:#111b31; border-color:#293653; }
    body.dark-preview .search, body.dark-preview .search input { color:#e8edff; background:#17223a; }
    body.dark-preview .search input::placeholder { color:#aab5d2; }
    body.dark-preview .panel, body.dark-preview .stat-card, body.dark-preview .quick-card { background:#121d33; border-color:#35415f; color:#eef2ff; }
    body.dark-preview .panel h2, body.dark-preview .stat-card strong, body.dark-preview .quick-card b { color:#eef2ff; }
    body.dark-preview .panel p, body.dark-preview .stat-card span, body.dark-preview .stat-card small, body.dark-preview .quick-card small, body.dark-preview .table-row, body.dark-preview .legend, body.dark-preview .bubble { color:#b7c2df; }
    body.dark-preview .welcome { background:linear-gradient(110deg,#1a2445,#17243b 55%,#25365a); }
    body.dark-preview .welcome h1, body.dark-preview .welcome p, body.dark-preview .welcome-art { color:#eef2ff; }
    body.dark-preview .tabs, body.dark-preview .tabs .selected, body.dark-preview .table-head, body.dark-preview .assistant-chat .bubble, body.dark-preview .source-list>div { background:#1b2945; color:#dce5ff; }
    body.dark-preview .table-row { border-color:#293653; }
    body.dark-preview .bubble { background:#1b2945; color:#e6ecff; }
    body.dark-preview .follow-up input { background:#17223a; border-color:#35415f; color:#eef2ff; }
    body.dark-preview .follow-up input::placeholder { color:#aab5d2; }
    body.dark-preview .donut:after { background:#121d33; }
    body.dark-preview .icon-button { color:#eef2ff; }
    body.dark-preview footer { color:#9aa9ce; }
  `;
  document.head.appendChild(style);

  function getUserName() {
    return localStorage.getItem('rag-user-name') || 'User';
  }
  function applyUserName() {
    const name = getUserName();
    const profile = document.querySelector('.profile');
    if (profile) profile.textContent = name.split(/\s+/).map(p => p[0]).join('').slice(0, 2).toUpperCase();
    const heading = document.querySelector('.welcome h1');
    if (heading) heading.textContent = `Welcome back, ${name}!`;
  }
  function applyThemeLabel() {
    const button = document.querySelector('#theme');
    if (button) {
      const dark = document.body.classList.contains('dark-preview');
      button.textContent = dark ? '☀' : '☼';
      button.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
      button.title = button.getAttribute('aria-label');
    }
  }
  document.addEventListener('DOMContentLoaded', () => {
    applyUserName();
    applyThemeLabel();
    document.querySelector('#theme')?.addEventListener('click', () => {
      const dark = document.body.classList.toggle('dark-preview');
      localStorage.setItem('rag-theme', dark ? 'dark' : 'light');
      applyThemeLabel();
    });
  });
})();
