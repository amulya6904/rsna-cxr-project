document.addEventListener('DOMContentLoaded', () => {
  const RAD_API = 'http://127.0.0.1:8000';
  const CARD_API = 'http://127.0.0.1:8001';

  const state = {
    activeModule: 'radiology',
    file: null,
    radParams: {
      spo2: '',
      symptoms: new Set(),
      urgency: 'Routine'
    },
    cases: loadCases()
  };

  // === ROUTING & MODULE SWITCHING ===
  const navLinks = document.querySelectorAll('.nav-link');
  const pages = document.querySelectorAll('.page');
  const moduleTabs = document.querySelectorAll('.module-tab');
  const appShell = document.querySelector('.app-shell');
  // moduleBadge removed in index.html, so skip it

  function navigateTo(routeId) {
    if (routeId !== 'home' && !state.activeModule) {
      alert('Please select a clinical specialty first.');
      return;
    }

    navLinks.forEach(l => l.classList.toggle('is-active', l.dataset.route === routeId));
    pages.forEach(p => {
      if (p.dataset.page === routeId) {
        p.classList.add('is-active');
        if (routeId === 'results') animateGauges();
      } else {
        p.classList.remove('is-active');
      }
    });

    // Show/hide module tabs
    const tabsContainer = document.getElementById('module-tabs-container');
    if (tabsContainer) {
      tabsContainer.style.display = routeId === 'home' ? 'none' : 'flex';
    }
  }

  navLinks.forEach(btn => {
    btn.addEventListener('click', () => navigateTo(btn.dataset.route));
  });

  function setModule(mod) {
    state.activeModule = mod;
    moduleTabs.forEach(t => t.classList.toggle('is-active', t.dataset.module === mod));
    appShell.setAttribute('data-module', mod);

    document.getElementById('radiology-input').classList.toggle('hidden', mod !== 'radiology');
    document.getElementById('cardiology-input').classList.toggle('hidden', mod !== 'cardiology');
    document.getElementById('radiology-results').classList.toggle('hidden', mod !== 'radiology');
    document.getElementById('cardiology-results').classList.toggle('hidden', mod !== 'cardiology');
    
    // Update Chatbot titles
    if (mod === 'radiology') {
      document.getElementById('chat-title').textContent = 'Pulmonology Knowledge Base Chat';
      document.getElementById('chat-welcome').textContent = 'Hello! I am your clinical assistant for Pulmonology guidelines. Ask me anything.';
    } else {
      document.getElementById('chat-title').textContent = 'Cardiology Knowledge Base Chat';
      document.getElementById('chat-welcome').textContent = 'Hello! I am your clinical assistant for Cardiology guidelines. Ask me anything.';
    }

    if (document.getElementById('results-page').classList.contains('is-active')) {
      animateGauges();
    }
  }

  moduleTabs.forEach(tab => {
    tab.addEventListener('click', () => setModule(tab.dataset.module));
  });

  // Home Page Specialty Selection
  document.querySelectorAll('.specialty-card').forEach(card => {
    card.addEventListener('click', (e) => {
      const module = e.currentTarget.dataset.selectModule;
      setModule(module);
      navigateTo('input');
    });
  });

  // === HEALTH CHECKS ===
  async function checkHealth(url, elId) {
    const el = document.getElementById(elId);
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error();
      el.className = 'status-pill status-ready';
      el.innerHTML = `<span class="status-dot"></span> ${elId === 'rad-status' ? 'Rad' : 'Card'} API`;
    } catch {
      el.className = 'status-pill status-offline';
      el.innerHTML = `<span class="status-dot"></span> ${elId === 'rad-status' ? 'Rad' : 'Card'} Offline`;
    }
  }
  setInterval(() => {
    checkHealth(`${RAD_API}/health`, 'rad-status');
    checkHealth(`${CARD_API}/`, 'card-status');
  }, 10000);
  checkHealth(`${RAD_API}/health`, 'rad-status');
  checkHealth(`${CARD_API}/`, 'card-status');

  // === RADIOLOGY INPUT ===
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('xray-upload');
  const previewImg = document.getElementById('preview-image');
  
  fileInput.addEventListener('change', e => {
    if (e.target.files.length) handleFile(e.target.files[0]);
  });
  
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('is-dragover'); });
  dropzone.addEventListener('dragleave', e => { e.preventDefault(); dropzone.classList.remove('is-dragover'); });
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('is-dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  function handleFile(f) {
    state.file = f;
    const card1 = document.getElementById('status-card-1');
    const card2 = document.getElementById('status-card-2');
    
    card1.classList.remove('empty');
    card1.classList.add('info');
    card1.innerHTML = `<div class="status-header"><span>Selected Image</span></div><div class="status-sub">${f.name}</div>`;
    
    card2.innerHTML = `<div class="status-header"><span>Ready for Analysis</span><span style="color:var(--color-success);">Ready</span></div><div class="status-sub">Formats validated.</div>`;
    
    const previewImg = document.getElementById('preview-image');
    if (previewImg && !f.name.toLowerCase().endsWith('.dcm')) {
      previewImg.src = URL.createObjectURL(f);
      previewImg.classList.remove('hidden');
    }
  }

  // Segmented control (urgency)
  document.querySelectorAll('.segmented-control .segment').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const parent = e.target.closest('.segmented-control');
      parent.querySelectorAll('.segment').forEach(s => s.classList.remove('is-selected'));
      e.target.classList.add('is-selected');
      if (parent.closest('#rad-form') || parent.closest('.grid-2')) {
        state.radParams.urgency = e.target.dataset.val;
      }
    });
  });

  // Symptom chips logic
  document.querySelectorAll('#symptom-chips .chip').forEach(chip => {
    chip.addEventListener('click', (e) => {
      e.target.classList.toggle('is-active');
      const val = e.target.dataset.val;
      if (e.target.classList.contains('is-active')) {
        state.radParams.symptoms.add(val);
      } else {
        state.radParams.symptoms.delete(val);
      }
      document.getElementById('rad-symptoms').value = Array.from(state.radParams.symptoms).join(', ');
    });
  });

  // Analyze Radiology
  document.getElementById('analyze-rad-btn').addEventListener('click', async (e) => {
    if (!state.file) return alert('Upload an image first.');
    const btn = e.target;
    btn.textContent = 'Processing...';
    btn.disabled = true;

    const fd = new FormData();
    fd.append('image', state.file);
    fd.append('spo2', document.getElementById('rad-spo2').value);
    fd.append('symptoms', document.getElementById('rad-symptoms').value);
    const historyEl = document.getElementById('rad-history');
    if (historyEl) fd.append('history', historyEl.value);
    fd.append('urgency', state.radParams.urgency);

    try {
      const res = await fetch(`${RAD_API}/predict`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error('API Error');
      const data = await res.json();
      
      renderRadResults(data);
      addCaseToHistory({ id: data.id, module: 'Radiology', risk: data.risk, prob: data.probability });
      navigateTo('results');
    } catch (err) {
      alert('Error: ' + err.message);
    } finally {
      btn.textContent = 'Analyze Case';
      btn.disabled = false;
    }
  });

  function renderRadResults(data) {
    document.getElementById('res-image').src = `data:image/png;base64,${data.image_base64}`;
    if (data.gradcam_base64) {
      document.getElementById('res-heatmap').src = `data:image/png;base64,${data.gradcam_base64}`;
    }
    
    document.getElementById('rad-risk-badge').className = `risk-badge ${data.risk}`;
    document.getElementById('rad-risk-label').textContent = `${data.risk.toUpperCase()} RISK`;
    document.getElementById('rad-gauge-value').textContent = `${Math.round(data.probability * 100)}%`;
    document.getElementById('rad-gauge-fill').dataset.prob = data.probability;
    document.getElementById('rad-model-summary').textContent = data.model_summary;

    const stepsCont = document.getElementById('rad-next-steps');
    stepsCont.innerHTML = data.next_steps.map((s, i) => `<div class="snippet-card"><div class="snippet-header"><span class="snippet-source">Step ${i+1}</span></div><div class="snippet-text">${s}</div></div>`).join('');
    
    const kbCont = document.getElementById('rad-knowledge');
    kbCont.innerHTML = data.knowledge.map((k) => `<div class="snippet-card"><div class="snippet-header"><span class="snippet-source">${k.source}</span></div><div class="snippet-text">${k.snippet}</div></div>`).join('');
  }

  // Tab switching for Radiology Guidance
  document.querySelectorAll('#radiology-results .tab').forEach(tab => {
    tab.addEventListener('click', (e) => {
      document.querySelectorAll('#radiology-results .tab').forEach(t => t.classList.remove('is-active'));
      e.target.classList.add('is-active');
      const t = e.target.dataset.tab;
      document.getElementById('rad-next-steps').classList.toggle('hidden', t !== 'next-steps');
      document.getElementById('rad-knowledge').classList.toggle('hidden', t !== 'knowledge');
    });
  });

  // Heatmap viewer toggles
  const viewOrigBtn = document.getElementById('view-orig');
  const viewGradcamBtn = document.getElementById('view-gradcam');
  const heatmapOpacityInput = document.getElementById('heatmap-opacity');
  
  if (viewOrigBtn && viewGradcamBtn) {
    viewOrigBtn.addEventListener('click', (e) => {
      e.target.classList.add('is-selected');
      viewGradcamBtn.classList.remove('is-selected');
      document.getElementById('res-heatmap-layer').classList.add('hidden');
    });
    viewGradcamBtn.addEventListener('click', (e) => {
      e.target.classList.add('is-selected');
      viewOrigBtn.classList.remove('is-selected');
      document.getElementById('res-heatmap-layer').classList.remove('hidden');
    });
  }
  
  if (heatmapOpacityInput) {
    heatmapOpacityInput.addEventListener('input', e => {
      document.getElementById('res-heatmap-layer').style.opacity = e.target.value / 100;
    });
  }


  // === CARDIOLOGY INPUT & RESULTS ===
  function safeVal(id, defaultVal) {
    const el = document.getElementById(id);
    if (!el || !el.value) return defaultVal;
    const parsed = parseFloat(el.value);
    return isNaN(parsed) ? defaultVal : parsed;
  }

  function getCardioPayload() {
    return {
      age: safeVal('card-age', 55),
      sex: safeVal('card-sex', 1),
      cp: safeVal('card-cp', 3),
      trestbps: safeVal('card-trestbps', 145),
      chol: safeVal('card-chol', 233),
      fbs: safeVal('card-fbs', 1),
      restecg: safeVal('card-restecg', 0),
      thalach: safeVal('card-thalach', 150),
      exang: safeVal('card-exang', 0),
      oldpeak: safeVal('card-oldpeak', 2.3),
      slope: safeVal('card-slope', 0),
      ca: safeVal('card-ca', 0),
      thal: safeVal('card-thal', 1)
    };
  }

  document.getElementById('fast-risk-btn').addEventListener('click', async (e) => {
    const btn = e.target; btn.textContent = '...';
    try {
      const res = await fetch(`${CARD_API}/predict`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(getCardioPayload())
      });
      const data = await res.json();
      renderCardioResults({ risk: data.risk_level, probability: data.probability, profile: "Run Full Analysis for profile." }, false);
      addCaseToHistory({ id: `CARD-${Date.now()}`, module: 'Cardiology', risk: data.risk_level, prob: data.probability });
      navigateTo('results');
    } catch(err) { alert(err.message); }
    finally { btn.textContent = 'Fast Risk Score'; }
  });

  document.getElementById('full-analysis-btn').addEventListener('click', async (e) => {
    const btn = e.target; btn.textContent = '...';
    document.getElementById('llm-output').innerHTML = '<span class="animate-reveal">Generating analysis...</span>';
    try {
      const res = await fetch(`${CARD_API}/full-analysis`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(getCardioPayload())
      });
      const data = await res.json();
      
      const payload = {
        risk: data.risk.risk_level,
        probability: data.risk.risk_probability,
        profile: data.profile,
        top_features: data.top_features,
        xai_summary: data.xai_summary,
        explanation: data.explanation
      };

      renderCardioResults(payload, true);
      addCaseToHistory({ id: `CARD-${Date.now()}`, module: 'Cardiology', risk: payload.risk, prob: payload.probability });
      navigateTo('results');
    } catch(err) { alert(err.message); }
    finally { btn.textContent = 'Full Analysis (LLM)'; }
  });

  function renderCardioResults(data, full) {
    document.getElementById('card-risk-badge').className = `risk-badge ${data.risk.toLowerCase().replace(' risk', '')}`;
    document.getElementById('card-risk-label').textContent = data.risk.toUpperCase().includes('RISK') ? data.risk.toUpperCase() : `${data.risk.toUpperCase()} RISK`;
    document.getElementById('card-gauge-value').textContent = `${Math.round(data.probability * 100)}%`;
    document.getElementById('card-gauge-fill').dataset.prob = data.probability;
    
    document.getElementById('card-profile').textContent = data.profile;

    const shapCont = document.getElementById('shap-container');
    shapCont.innerHTML = '';
    
    if (full && data.top_features) {
      data.top_features.forEach((f, i) => {
        const isPos = f.impact > 0;
        const width = Math.min(100, Math.abs(f.impact) * 100);
        shapCont.innerHTML += `
          <div class="shap-row animate-reveal delay-${i+1}">
            <div class="shap-label" title="${f.feature}">${f.feature}</div>
            <div class="shap-track">
              <div class="shap-center-line"></div>
              <div class="shap-fill ${isPos ? 'positive' : 'negative'}" style="width: ${width}%"></div>
            </div>
            <div class="shap-value" style="color: var(--${isPos ? 'color-danger' : 'color-success'})">${isPos ? '+' : ''}${f.impact.toFixed(2)}</div>
          </div>
        `;
      });
      document.getElementById('llm-output').innerHTML = `<p class="animate-reveal delay-3">${data.explanation}</p>`;
    } else {
      shapCont.innerHTML = '<p style="color:var(--text-500)">Run Full Analysis to see explainability.</p>';
      document.getElementById('llm-output').textContent = 'Run Full Analysis to generate LLM explanation...';
    }
  }


  // === SHARED UTILS ===
  function animateGauges() {
    setTimeout(() => {
      ['rad-gauge-fill', 'card-gauge-fill'].forEach(id => {
        const el = document.getElementById(id);
        if(el && el.dataset.prob) {
          const prob = parseFloat(el.dataset.prob);
          // circumference is 2 * pi * r = 2 * 3.14159 * 58 = ~364.4 (in CSS it is 440 but we set r=58 so 2*pi*58=364)
          // Wait, the new CSS has stroke-dasharray 440. I will adjust offset based on 440.
          const offset = 440 - (prob * 440);
          el.style.strokeDashoffset = offset;
          
          if (prob >= 0.7) el.style.stroke = 'var(--color-danger)';
          else if (prob >= 0.4) el.style.stroke = 'var(--color-warning)';
          else el.style.stroke = 'var(--color-success)';
        }
      });
    }, 100);
  }

  function addCaseToHistory(item) {
    state.cases.unshift(item);
    sessionStorage.setItem('nuvexav2', JSON.stringify(state.cases));
    renderHistory();
  }

  function loadCases() {
    try { return JSON.parse(sessionStorage.getItem('nuvexav2')) || []; }
    catch { return []; }
  }

  function renderHistory() {
    const tbody = document.getElementById('history-body');
    if (state.cases.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-500); padding: 48px;">No cases analyzed yet in this session.</td></tr>';
      return;
    }
    
    tbody.innerHTML = state.cases.map(c => `
      <tr>
        <td style="font-family: 'JetBrains Mono', monospace;">${c.id}</td>
        <td><span class="brand-badge" style="background: var(--color-${c.module === 'Radiology' ? 'primary' : 'cardio'})">${c.module}</span></td>
        <td><div class="risk-badge ${c.risk.toLowerCase()}"><span>${c.risk.toUpperCase()}</span></div></td>
        <td style="font-family: 'JetBrains Mono', monospace;">${Math.round(c.prob * 100)}%</td>
      </tr>
    `).join('');
  }

  // === CHATBOT LOGIC ===
  const chatInput = document.getElementById('chat-input');
  const chatSendBtn = document.getElementById('chat-send-btn');
  const chatMessages = document.getElementById('chat-messages');

  async function sendChatMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    const userMsg = document.createElement('div');
    userMsg.className = 'chat-message user';
    userMsg.textContent = text;
    chatMessages.appendChild(userMsg);
    chatInput.value = '';
    chatMessages.scrollTop = chatMessages.scrollHeight;

    const botMsg = document.createElement('div');
    botMsg.className = 'chat-message bot';
    botMsg.innerHTML = '<span class="animate-reveal">Thinking...</span>';
    chatMessages.appendChild(botMsg);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    const apiBase = state.activeModule === 'radiology' ? RAD_API : CARD_API;
    try {
      const res = await fetch(`${apiBase}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text })
      });
      if (!res.ok) throw new Error('Failed to get response');
      const data = await res.json();
      botMsg.textContent = data.answer;
    } catch (err) {
      botMsg.textContent = 'Error connecting to the chatbot service. Is the backend running?';
      botMsg.style.color = 'var(--color-danger)';
    }
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  chatSendBtn.addEventListener('click', sendChatMessage);
  chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendChatMessage();
  });

  // Init
  renderHistory();
});
