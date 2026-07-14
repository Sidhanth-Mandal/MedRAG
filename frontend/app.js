/**
 * app.js — MedRAG Chatbot Frontend
 * Full client-side logic for the medical Q&A chatbot.
 */

'use strict';

// ── Config ────────────────────────────────────────────────────────────────────
const API_BASE = 'https://medrag-api-768669600860.asia-south1.run.app';
const STEP_DELAY = 600;  // ms between thinking step animations

// ── State ─────────────────────────────────────────────────────────────────────
let currentSessionId   = null;
let sessions           = {};     // sessionId → { messages: [], preview: '' }
let isLoading          = false;
let currentCitations   = [];     // citations for currently open panel

// ── Auth state ────────────────────────────────────────────────────────────────
let authToken    = localStorage.getItem('medrag_token') || null;
let authUsername = localStorage.getItem('medrag_username') || null;
let authMode     = 'login';   // 'login' | 'register'

/** Return headers with Authorization if a token is set. */
function authHeaders(extra = {}) {
  const h = { 'Content-Type': 'application/json', ...extra };
  if (authToken) h['Authorization'] = `Bearer ${authToken}`;
  return h;
}

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $chatArea       = document.getElementById('chatArea');
const $welcomeScreen  = document.getElementById('welcomeScreen');
const $messages       = document.getElementById('messages');
const $queryInput     = document.getElementById('queryInput');
const $sendBtn        = document.getElementById('sendBtn');
const $sessionList    = document.getElementById('sessionList');
const $newChatBtn     = document.getElementById('newChatBtn');
const $citationPanel  = document.getElementById('citationPanel');
const $panelBody      = document.getElementById('citationPanelBody');
const $closePanelBtn  = document.getElementById('closePanelBtn');
const $panelOverlay   = document.getElementById('panelOverlay');
const $thinkingOverlay = document.getElementById('thinkingOverlay');
const $topbarTitle    = document.getElementById('topbarTitle');
const $routeBadge     = document.getElementById('routeBadge');
const $sidebar        = document.getElementById('sidebar');
const $sidebarOverlay = document.getElementById('sidebarOverlay');
const $menuBtn        = document.getElementById('menuBtn');

// Thinking step elements
const thinkingSteps = {
  safety:   document.getElementById('step-safety'),
  route:    document.getElementById('step-route'),
  retrieve: document.getElementById('step-retrieve'),
  grade:    document.getElementById('step-grade'),
  answer:   document.getElementById('step-answer'),
};

// ── Utilities ─────────────────────────────────────────────────────────────────

function generateSessionId() {
  return 'sess_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/**
 * Convert simple markdown-like text to HTML.
 * Handles **bold**, *italic*, [n] citation refs, line breaks.
 */
function formatAnswer(text) {
  // Escape HTML first
  let html = escapeHtml(text);

  // Bold: **text**
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

  // Italic: *text*
  html = html.replace(/\*([^*]+?)\*/g, '<em>$1</em>');

  // Citation refs: [1], [2] → clickable
  html = html.replace(/\[(\d+)\]/g, (_, n) =>
    `<a class="cite-link" title="View source ${n}" onclick="highlightCitation(${n})">${n}</a>`
  );

  // Newlines → <br> (but double newlines → paragraphs)
  html = html.split(/\n\n+/).map(para => `<p>${para.replace(/\n/g, '<br>')}</p>`).join('');

  return html;
}

function getSourceBadgeHtml(route) {
  if (!route) return '';
  const labels = { research: 'Research', guideline: 'Guideline', both: 'Research + Guidelines' };
  const label  = labels[route] || route;
  return `<span class="badge badge-${route}">${label}</span>`;
}

// ── Session management ────────────────────────────────────────────────────────

function startNewSession() {
  currentSessionId = generateSessionId();
  sessions[currentSessionId] = { messages: [], preview: '', loaded: true };
  $messages.innerHTML = '';   // clear previous session's DOM before hiding
  showWelcome();
  renderSessionList();
  setTopbarTitle('New conversation');
  $routeBadge.style.display = 'none';
  closeSidebar();
}

async function switchSession(sessionId) {
  currentSessionId = sessionId;
  const session = sessions[sessionId];
  if (!session) return;

  // If messages haven't been loaded yet, fetch them from the API.
  // We use a `loaded` flag instead of messages.length so that a session
  // with zero messages (brand-new) is not re-fetched on every switch.
  if (!session.loaded) {
    session.loaded = true;  // mark immediately to prevent concurrent fetches
    try {
      const resp = await fetch(`${API_BASE}/api/sessions/${sessionId}/history`, {
        headers: authHeaders(),
      });
      if (resp.ok) {
        const data = await resp.json();
        session.messages = data.messages.map(m => ({
          role:                 m.role,
          content:             m.content,
          citations:           m.citations || [],
          route:               m.route || '',
          is_refused:          m.is_refused || false,
          has_contradiction:   m.has_contradiction || false,
          contradiction_details: m.contradiction_details || '',
          // answer field expected by appendAIMessage
          answer:              m.content,
        }));
      }
    } catch (e) {
      // Silently fall through — will just show empty chat
    }
  }

  // Clear route badge — it belongs to the last message of the active session,
  // not carried over from the previous session.
  $routeBadge.style.display = 'none';

  hideWelcome();
  $messages.innerHTML = '';

  session.messages.forEach(msg => {
    if (msg.role === 'human') {
      appendHumanMessage(msg.content);
    } else {
      appendAIMessage(msg);
    }
  });

  setTopbarTitle(session.preview || 'Conversation');
  renderSessionList();
  closeSidebar();
  scrollToBottom();
}

function deleteSession(sessionId, event) {
  event.stopPropagation();

  fetch(`${API_BASE}/api/sessions/${sessionId}`, {
    method: 'DELETE',
    headers: authHeaders(),
  }).catch(() => {});
  delete sessions[sessionId];

  if (currentSessionId === sessionId) {
    currentSessionId = null;
    showWelcome();
  }

  renderSessionList();
}

function renderSessionList() {
  const ids = Object.keys(sessions);
  if (ids.length === 0) {
    $sessionList.innerHTML = '<div class="session-empty">No conversations yet</div>';
    return;
  }

  $sessionList.innerHTML = ids.map(id => {
    const s = sessions[id];
    const isActive = id === currentSessionId;
    const preview  = escapeHtml((s.preview || 'New conversation').slice(0, 42));
    const count    = s.messages.filter(m => m.role === 'human').length;
    return `
      <div class="session-item ${isActive ? 'active' : ''}" onclick="switchSession('${id}')" id="session-${id}">
        <div class="session-item-icon">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
        </div>
        <div class="session-item-info">
          <div class="session-item-preview">${preview}</div>
          <div class="session-item-meta">${count} question${count !== 1 ? 's' : ''}</div>
        </div>
        <button class="session-item-delete" onclick="deleteSession('${id}', event)" title="Delete">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/>
          </svg>
        </button>
      </div>
    `;
  }).join('');
}

// ── View helpers ──────────────────────────────────────────────────────────────

function showWelcome() {
  $welcomeScreen.style.display = 'flex';
  $messages.style.display = 'none';
}

function hideWelcome() {
  $welcomeScreen.style.display = 'none';
  $messages.style.display = 'block';
}

function setTopbarTitle(text) {
  $topbarTitle.textContent = text;
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    $chatArea.scrollTop = $chatArea.scrollHeight;
  });
}

// ── Message rendering ─────────────────────────────────────────────────────────

function appendHumanMessage(text) {
  const group = document.createElement('div');
  group.className = 'message-group';
  group.innerHTML = `
    <div class="message-human">
      <div class="message-human-bubble">${escapeHtml(text)}</div>
    </div>
  `;
  $messages.appendChild(group);
  scrollToBottom();
  return group;
}

function appendTypingIndicator() {
  const group = document.createElement('div');
  group.className = 'message-group';
  group.id = 'typingIndicator';
  group.innerHTML = `
    <div class="message-ai">
      <div class="message-ai-header">
        <div class="ai-avatar">M</div>
        <span class="ai-label">MedRAG</span>
      </div>
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>
  `;
  $messages.appendChild(group);
  scrollToBottom();
  return group;
}

function removeTypingIndicator() {
  const el = document.getElementById('typingIndicator');
  if (el) el.remove();
}

function appendAIMessage(responseData) {
  const {
    answer: rawAnswer,
    content,
    citations       = [],
    route           = '',
    is_refused      = false,
    has_contradiction = false,
    contradiction_details = '',
  } = responseData;

  // Fallback: history messages stored locally use `content`; API responses use `answer`
  const answer = rawAnswer ?? content ?? '';

  // Build conflict / refused banners
  let banners = '';
  if (has_contradiction) {
    banners += `
      <div class="alert-banner conflict">
        <div class="alert-icon">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
        </div>
        <div>
          <strong>Sources conflict</strong>
          ${escapeHtml(contradiction_details || 'Different sources disagree on this topic.')}
        </div>
      </div>`;
  }
  if (is_refused) {
    banners += `
      <div class="alert-banner refused">
        <div class="alert-icon">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/>
          </svg>
        </div>
        <div>
          <strong>Cannot answer this question</strong>
          This system provides general medical information only and cannot give personal medical advice.
          Please consult a qualified healthcare professional.
        </div>
      </div>`;
  }

  // Build citation chips
  let citBar = '';
  if (citations.length > 0) {
    const chips = citations.slice(0, 4).map(c => `
      <button class="citation-chip ${c.source_type}" onclick="openCitationPanel(event)" title="${escapeHtml(c.title)}">
        [${c.index}] ${c.source_type === 'research' ? '🔬' : '📋'} ${escapeHtml((c.title || '').slice(0, 25))}
      </button>
    `).join('');

    const moreBtn = citations.length > 4
      ? `<button class="view-all-citations" onclick="openCitationPanel(event)">+${citations.length - 4} more</button>`
      : '';

    citBar = `<div class="citation-bar">${chips}${moreBtn}</div>`;
  }

  const badgeHtml = route && !is_refused ? getSourceBadgeHtml(route) : (is_refused ? '<span class="badge badge-refused">Refused</span>' : '');

  const group = document.createElement('div');
  group.className = 'message-group';
  group.innerHTML = `
    <div class="message-ai">
      <div class="message-ai-header">
        <div class="ai-avatar">M</div>
        <span class="ai-label">MedRAG</span>
        ${badgeHtml}
      </div>
      <div class="message-ai-body">${formatAnswer(answer)}</div>
      ${banners}
      ${citBar}
    </div>
  `;

  // Store citations on this group for retrieval
  group.dataset.citations = JSON.stringify(citations);

  $messages.appendChild(group);
  scrollToBottom();
  return group;
}

// ── Citation panel ────────────────────────────────────────────────────────────

function openCitationPanel(event) {
  // Find the parent message group
  const group = event.target.closest('.message-group');
  if (!group) return;

  const citations = JSON.parse(group.dataset.citations || '[]');
  renderCitationPanel(citations);

  $citationPanel.classList.add('open');
  $panelOverlay.style.display = 'block';
}

function renderCitationPanel(citations) {
  currentCitations = citations;
  $panelBody.innerHTML = citations.map(c => {
    const score    = Math.round((c.rerank_score || 0) * 100);
    const typeIcon = c.source_type === 'research' ? '🔬' : '📋';
    const typeLabel = c.source_type === 'research' ? 'PubMed Research' : 'MedlinePlus Guideline';
    return `
      <div class="citation-card" id="cite-card-${c.index}">
        <div class="citation-card-header">
          <div class="citation-number">${c.index}</div>
          <div class="citation-card-info">
            <div class="citation-title">${escapeHtml(c.title)}</div>
            <div class="citation-meta">
              <span class="citation-meta-item">
                <span class="badge ${c.source_type === 'research' ? 'badge-research' : 'badge-guideline'}" style="font-size:10px">${typeIcon} ${c.source_type}</span>
              </span>
              ${c.pub_date ? `<span class="citation-meta-item">📅 ${escapeHtml(c.pub_date)}</span>` : ''}
              ${c.condition ? `<span class="citation-meta-item">🏥 ${escapeHtml(c.condition.replace(/_/g,' '))}</span>` : ''}
            </div>
          </div>
        </div>
        ${c.snippet ? `<div class="citation-snippet">"${escapeHtml(c.snippet)}…"</div>` : ''}
        <div class="citation-footer">
          <a class="citation-link" href="${escapeHtml(c.url)}" target="_blank" rel="noopener">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
              <polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/>
            </svg>
            ${c.source_type === 'research' ? 'View on PubMed' : 'View on MedlinePlus'}
          </a>
          ${score > 0 ? `
          <div class="score-bar">
            <span>Relevance</span>
            <div class="score-fill">
              <div class="score-fill-inner" style="width:${score}%"></div>
            </div>
            <span>${score}%</span>
          </div>` : ''}
        </div>
      </div>
    `;
  }).join('');
}

function closeCitationPanel() {
  $citationPanel.classList.remove('open');
  $panelOverlay.style.display = 'none';
}

function highlightCitation(n) {
  const group = $messages.querySelector(`.message-group:last-child`);
  if (!group) return;
  const citations = JSON.parse(group.dataset.citations || '[]');
  renderCitationPanel(citations);
  $citationPanel.classList.add('open');
  $panelOverlay.style.display = 'block';

  // Scroll to the specific card
  setTimeout(() => {
    const card = document.getElementById(`cite-card-${n}`);
    if (card) {
      card.scrollIntoView({ behavior: 'smooth', block: 'center' });
      card.style.borderColor = 'var(--border-accent)';
      card.style.boxShadow   = '0 0 0 2px var(--accent-glow)';
      setTimeout(() => {
        card.style.borderColor = '';
        card.style.boxShadow   = '';
      }, 2000);
    }
  }, 100);
}

// ── Thinking overlay ──────────────────────────────────────────────────────────

async function runThinkingAnimation() {
  const steps = ['safety', 'route', 'retrieve', 'grade', 'answer'];
  let current = 0;

  // Reset all steps
  steps.forEach(s => {
    thinkingSteps[s].className = 'thinking-step';
  });

  // Activate first step immediately
  thinkingSteps[steps[0]].classList.add('active');

  // Animate through steps
  return new Promise(resolve => {
    const interval = setInterval(() => {
      // Mark current as done
      thinkingSteps[steps[current]].className = 'thinking-step done';
      current++;

      if (current < steps.length) {
        thinkingSteps[steps[current]].classList.add('active');
      } else {
        clearInterval(interval);
        resolve();
      }
    }, STEP_DELAY);
  });
}

function showThinking() {
  $thinkingOverlay.style.display = 'flex';
  runThinkingAnimation();
}

function hideThinking() {
  $thinkingOverlay.style.display = 'none';
}

// ── API calls ─────────────────────────────────────────────────────────────────

async function sendMessage(query) {
  if (!query.trim() || isLoading) return;

  // Ensure we have a session
  if (!currentSessionId) {
    currentSessionId = generateSessionId();
    sessions[currentSessionId] = { messages: [], preview: query.slice(0, 60) };
  } else if (!sessions[currentSessionId].preview) {
    sessions[currentSessionId].preview = query.slice(0, 60);
  }

  isLoading = true;
  $sendBtn.disabled  = true;
  $queryInput.disabled = true;

  // Hide welcome, show messages
  hideWelcome();

  // Append human message
  appendHumanMessage(query);

  // Store in local session
  sessions[currentSessionId].messages.push({ role: 'human', content: query });
  renderSessionList();
  setTopbarTitle(sessions[currentSessionId].preview);

  // Show typing + thinking overlay
  const typingEl = appendTypingIndicator();
  showThinking();

  try {
    const resp = await fetch(`${API_BASE}/api/chat`, {
      method:  'POST',
      headers: authHeaders(),
      body:    JSON.stringify({
        session_id:  currentSessionId,
        message:     query,
        new_session: false,
      }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }

    const data = await resp.json();

    // Wait for thinking animation to nearly finish
    await new Promise(r => setTimeout(r, 800));

    hideThinking();
    removeTypingIndicator();

    // Append AI response
    appendAIMessage(data);

    // Store in session
    sessions[currentSessionId].messages.push({
      role:      'ai',
      content:   data.answer,
      answer:    data.answer,   // needed by appendAIMessage when re-rendering on session switch
      citations: data.citations,
      route:     data.route,
      is_refused: data.is_refused,
      has_contradiction: data.has_contradiction,
      contradiction_details: data.contradiction_details,
    });

    // Update topbar badge
    if (data.route && !data.is_refused) {
      $routeBadge.className  = `badge badge-${data.route}`;
      $routeBadge.textContent = { research: 'Research', guideline: 'Guideline', both: 'Research + Guidelines' }[data.route] || data.route;
      $routeBadge.style.display = 'inline-flex';
    } else if (data.is_refused) {
      $routeBadge.className   = 'badge badge-refused';
      $routeBadge.textContent = 'Refused';
      $routeBadge.style.display = 'inline-flex';
    }

  } catch (err) {
    hideThinking();
    removeTypingIndicator();

    // Show error message
    const group = document.createElement('div');
    group.className = 'message-group';
    group.innerHTML = `
      <div class="message-ai">
        <div class="message-ai-header">
          <div class="ai-avatar">M</div>
          <span class="ai-label">MedRAG</span>
        </div>
        <div class="message-ai-body">
          <div class="alert-banner refused">
            <div class="alert-icon">⚠</div>
            <div><strong>Connection error</strong>${escapeHtml(err.message)}</div>
          </div>
        </div>
      </div>
    `;
    group.dataset.citations = '[]';
    $messages.appendChild(group);
    scrollToBottom();
  } finally {
    isLoading            = false;
    $sendBtn.disabled    = !$queryInput.value.trim();
    $queryInput.disabled = false;
    $queryInput.focus();
  }
}

// ── Sidebar mobile toggle ─────────────────────────────────────────────────────

function openSidebar() {
  $sidebar.classList.add('open');
  $sidebarOverlay.classList.add('active');
}

function closeSidebar() {
  $sidebar.classList.remove('open');
  $sidebarOverlay.classList.remove('active');
}

// ── Load sessions from API ────────────────────────────────────────────────────

async function loadSessions() {
  if (!authToken) return;
  try {
    const resp = await fetch(`${API_BASE}/api/sessions`, {
      headers: authHeaders(),
    });
    if (!resp.ok) return;
    const data = await resp.json();
    sessions = {};  // clear stale data before repopulating
    data.forEach(s => {
      sessions[s.session_id] = {
        messages: [],
        preview:  s.preview || 'Conversation',
        loaded:   false,  // will fetch history from API on first switchSession()
      };
    });
    renderSessionList();
  } catch (e) {
    // Silently fail — API might not be ready yet
  }
}

// ── Event listeners ───────────────────────────────────────────────────────────

// Auto-resize textarea
$queryInput.addEventListener('input', () => {
  $queryInput.style.height = 'auto';
  $queryInput.style.height = Math.min($queryInput.scrollHeight, 180) + 'px';
  $sendBtn.disabled = !$queryInput.value.trim() || isLoading;
});

// Send on Enter (not Shift+Enter)
$queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    const query = $queryInput.value.trim();
    if (query) {
      $queryInput.value = '';
      $queryInput.style.height = 'auto';
      $sendBtn.disabled = true;
      sendMessage(query);
    }
  }
});

// Send button
$sendBtn.addEventListener('click', () => {
  const query = $queryInput.value.trim();
  if (query) {
    $queryInput.value = '';
    $queryInput.style.height = 'auto';
    $sendBtn.disabled = true;
    sendMessage(query);
  }
});

// New chat button
$newChatBtn.addEventListener('click', startNewSession);

// Suggestion cards
document.querySelectorAll('.suggestion-card').forEach(card => {
  card.addEventListener('click', () => {
    const query = card.dataset.query;
    $queryInput.value = query;
    $queryInput.dispatchEvent(new Event('input'));
    sendMessage(query);
    $queryInput.value = '';
  });
});

// Citation panel
$closePanelBtn.addEventListener('click', closeCitationPanel);
$panelOverlay.addEventListener('click', closeCitationPanel);

// Mobile sidebar
$menuBtn.addEventListener('click', openSidebar);
$sidebarOverlay.addEventListener('click', closeSidebar);

// Make functions global for inline onclick handlers
window.switchSession    = switchSession;
window.deleteSession    = deleteSession;
window.openCitationPanel = openCitationPanel;
window.highlightCitation = highlightCitation;

// ── Database Stats Modal ───────────────────────────────────────────────────────

const $dbStatsBtn       = document.getElementById('dbStatsBtn');
const $statsModal       = document.getElementById('statsModal');
const $statsModalOverlay = document.getElementById('statsModalOverlay');
const $statsModalClose  = document.getElementById('statsModalClose');
const $statsModalBody   = document.getElementById('statsModalBody');

function openStatsModal() {
  $statsModal.classList.add('open');
  $statsModalOverlay.classList.add('open');
  fetchAndRenderStats();
}

function closeStatsModal() {
  $statsModal.classList.remove('open');
  $statsModalOverlay.classList.remove('open');
}

async function fetchAndRenderStats() {
  $statsModalBody.innerHTML = `
    <div class="stats-loading">
      <div class="stats-spinner"></div>
      <span>Loading stats…</span>
    </div>`;

  try {
    const res  = await fetch(`${API_BASE}/api/stats`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const total      = data.total_documents ?? 0;
    const research   = data.by_source?.research   ?? 0;
    const guideline  = data.by_source?.guideline  ?? 0;
    const diabetes   = data.by_condition?.diabetes     ?? 0;
    const hypertension = data.by_condition?.hypertension ?? 0;
    const asthma     = data.by_condition?.asthma       ?? 0;

    const pct = (n) => total > 0 ? Math.round((n / total) * 100) : 0;

    $statsModalBody.innerHTML = `
      <!-- Summary cards -->
      <div class="stats-summary">
        <div class="stat-card">
          <div class="stat-card-value">${total.toLocaleString()}</div>
          <div class="stat-card-label">Total Vectors</div>
        </div>
        <div class="stat-card">
          <div class="stat-card-value">${research.toLocaleString()}</div>
          <div class="stat-card-label">PubMed Chunks</div>
        </div>
        <div class="stat-card">
          <div class="stat-card-value">${guideline.toLocaleString()}</div>
          <div class="stat-card-label">Guideline Chunks</div>
        </div>
      </div>

      <!-- By source -->
      <div>
        <div class="stats-section-title">By Source</div>
        <div class="stats-breakdown">
          <div>
            <div class="stats-row">
              <div class="stats-row-label">
                <div class="stats-row-dot" style="background:var(--accent)"></div>
                PubMed Research
              </div>
              <div class="stats-row-value">${research.toLocaleString()} <span style="color:var(--text-muted);font-weight:400">(${pct(research)}%)</span></div>
            </div>
            <div class="stats-bar-wrap"><div class="stats-bar-fill" style="width:${pct(research)}%;background:var(--accent)"></div></div>
          </div>
          <div>
            <div class="stats-row">
              <div class="stats-row-label">
                <div class="stats-row-dot" style="background:var(--accent)"></div>
                MedlinePlus Guidelines
              </div>
              <div class="stats-row-value">${guideline.toLocaleString()} <span style="color:var(--text-muted);font-weight:400">(${pct(guideline)}%)</span></div>
            </div>
            <div class="stats-bar-wrap"><div class="stats-bar-fill" style="width:${pct(guideline)}%;background:var(--accent)"></div></div>
          </div>
        </div>
      </div>

      <!-- By condition -->
      <div>
        <div class="stats-section-title">By Condition</div>
        <div class="stats-breakdown">
          <div>
            <div class="stats-row">
              <div class="stats-row-label">
                <div class="stats-row-dot" style="background:var(--accent)"></div>
                Type 2 Diabetes
              </div>
              <div class="stats-row-value">${diabetes.toLocaleString()} <span style="color:var(--text-muted);font-weight:400">(${pct(diabetes)}%)</span></div>
            </div>
            <div class="stats-bar-wrap"><div class="stats-bar-fill" style="width:${pct(diabetes)}%;background:var(--accent)"></div></div>
          </div>
          <div>
            <div class="stats-row">
              <div class="stats-row-label">
                <div class="stats-row-dot" style="background:var(--accent)"></div>
                Hypertension
              </div>
              <div class="stats-row-value">${hypertension.toLocaleString()} <span style="color:var(--text-muted);font-weight:400">(${pct(hypertension)}%)</span></div>
            </div>
            <div class="stats-bar-wrap"><div class="stats-bar-fill" style="width:${pct(hypertension)}%;background:var(--accent)"></div></div>
          </div>
          <div>
            <div class="stats-row">
              <div class="stats-row-label">
                <div class="stats-row-dot" style="background:var(--accent)"></div>
                Asthma
              </div>
              <div class="stats-row-value">${asthma.toLocaleString()} <span style="color:var(--text-muted);font-weight:400">(${pct(asthma)}%)</span></div>
            </div>
            <div class="stats-bar-wrap"><div class="stats-bar-fill" style="width:${pct(asthma)}%;background:var(--accent)"></div></div>
          </div>
        </div>
      </div>

      <!-- Collection name -->
      <div style="font-size:11px;color:var(--text-muted);text-align:center">
        Collection: <span style="font-family:var(--font-mono)">${escapeHtml(data.collection ?? '')}</span>
      </div>
    `;
  } catch (err) {
    $statsModalBody.innerHTML = `<div class="stats-error">Failed to load stats: ${escapeHtml(err.message)}</div>`;
  }
}

$dbStatsBtn.addEventListener('click', openStatsModal);
$statsModalClose.addEventListener('click', closeStatsModal);
$statsModalOverlay.addEventListener('click', closeStatsModal);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeStatsModal(); });

// ── Auth gate ─────────────────────────────────────────────────────────────────

const $authGate      = document.getElementById('authGate');
const $authForm      = document.getElementById('authForm');
const $authUsername  = document.getElementById('authUsername');
const $authPassword  = document.getElementById('authPassword');
const $authError     = document.getElementById('authError');
const $authSubmitBtn = document.getElementById('authSubmitBtn');
const $authBtnText   = document.getElementById('authBtnText');
const $authBtnSpinner = document.getElementById('authBtnSpinner');
const $authToggleBtn = document.getElementById('authToggleBtn');
const $authToggleText = document.getElementById('authToggleText');
const $authTitle     = document.getElementById('authTitle');
const $authSubtitle  = document.querySelector('.auth-subtitle');
const $sidebarUser   = document.getElementById('sidebarUser');
const $sidebarUsername = document.getElementById('sidebarUsername');
const $logoutBtn     = document.getElementById('logoutBtn');

function showAuthGate() {
  $authGate.classList.add('visible');
  setTimeout(() => $authUsername.focus(), 100);
}

function hideAuthGate() {
  $authGate.classList.remove('visible');
}

function setAuthMode(mode) {
  authMode = mode;
  if (mode === 'login') {
    $authTitle.textContent       = 'Welcome back';
    $authSubtitle.textContent    = 'Sign in to access your conversations';
    $authBtnText.textContent     = 'Sign in';
    $authToggleText.textContent  = "Don't have an account?";
    $authToggleBtn.textContent   = 'Create one';
  } else {
    $authTitle.textContent       = 'Create account';
    $authSubtitle.textContent    = 'Username ≥ 3 chars · Password ≥ 6 chars';
    $authBtnText.textContent     = 'Register';
    $authToggleText.textContent  = 'Already have an account?';
    $authToggleBtn.textContent   = 'Sign in';
  }
  $authError.style.display = 'none';
  $authPassword.value = '';
}

function showAuthError(detail) {
  let msg;
  if (Array.isArray(detail)) {
    // FastAPI validation errors: [{msg: '...', loc: [...]}, ...]
    msg = detail.map(e => e.msg || String(e)).join(' · ');
  } else if (detail && typeof detail === 'object') {
    msg = detail.msg || JSON.stringify(detail);
  } else {
    msg = detail || 'Something went wrong';
  }
  $authError.textContent   = msg;
  $authError.style.display = 'block';
}

function onAuthSuccess(token, username) {
  authToken    = token;
  authUsername = username;
  localStorage.setItem('medrag_token', token);
  localStorage.setItem('medrag_username', username);

  // Show user info in sidebar
  $sidebarUsername.textContent = username;
  $sidebarUser.style.display   = 'flex';

  hideAuthGate();
  loadSessions();
  $queryInput.focus();
}

$authToggleBtn.addEventListener('click', () => {
  setAuthMode(authMode === 'login' ? 'register' : 'login');
});

$authForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const username = $authUsername.value.trim();
  const password = $authPassword.value;

  // Client-side validation
  if (authMode === 'register') {
    if (username.length < 3) { showAuthError('Username must be at least 3 characters'); return; }
    if (password.length < 6) { showAuthError('Password must be at least 6 characters'); return; }
  }

  // Show spinner
  $authSubmitBtn.disabled      = true;
  $authBtnText.style.display   = 'none';
  $authBtnSpinner.style.display = 'block';
  $authError.style.display     = 'none';

  const endpoint = authMode === 'login' ? `${API_BASE}/api/auth/login` : `${API_BASE}/api/auth/register`;

  try {
    const resp = await fetch(endpoint, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ username, password }),
    });

    const data = await resp.json();

    if (!resp.ok) {
      showAuthError(data.detail || 'Something went wrong');
    } else {
      onAuthSuccess(data.access_token, data.username);
    }
  } catch (err) {
    showAuthError('Could not reach the server. Is the API running?');
  } finally {
    $authSubmitBtn.disabled       = false;
    $authBtnText.style.display    = 'inline';
    $authBtnSpinner.style.display = 'none';
  }
});

$logoutBtn.addEventListener('click', () => {
  authToken    = null;
  authUsername = null;
  localStorage.removeItem('medrag_token');
  localStorage.removeItem('medrag_username');
  sessions     = {};
  currentSessionId = null;

  $sidebarUser.style.display = 'none';
  renderSessionList();
  showWelcome();
  setAuthMode('login');
  showAuthGate();
});

// ── Init ──────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  // Try to verify existing token
  if (authToken) {
    try {
      const resp = await fetch(`${API_BASE}/api/auth/verify`, {
        headers: authHeaders(),
      });
      if (resp.ok) {
        const user = await resp.json();
        // Token valid — boot the app
        $sidebarUsername.textContent = user.username;
        $sidebarUser.style.display   = 'flex';
        hideAuthGate();
        await loadSessions();
        $queryInput.focus();
        return;
      }
    } catch (_) { /* fall through */ }
  }
  // No valid token — show login
  showAuthGate();
});
