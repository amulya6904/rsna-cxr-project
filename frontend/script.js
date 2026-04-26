document.addEventListener('DOMContentLoaded', () => {
  const metaApi = document.querySelector('meta[name="api-base"]');
  const API_BASE = window.NUVEXA_API_BASE || (metaApi && metaApi.content) || 'http://127.0.0.1:8000';

  // === STATE ===
  const state = {
    file: null,
    patient: {
      age: '',
      sex: 'Male',
      symptoms: new Set(),
      spo2: '',
      urgency: 'Routine',
      history: ''
    },
    isAnalyzing: false,
    cases: loadCases(),
    currentResult: null
  };

  // === ROUTING ===
  const pages = document.querySelectorAll('.page');
  const navLinks = document.querySelectorAll('.nav-link');
  const routeButtons = document.querySelectorAll('[data-route]');
  const menuToggle = document.querySelector('[data-menu-toggle]');
  const mobileNav = document.querySelector('[data-mobile-nav]');

  if (menuToggle && mobileNav) {
    menuToggle.addEventListener('click', () => {
      const isOpen = mobileNav.classList.toggle('is-open');
      mobileNav.style.display = isOpen ? 'flex' : '';
    });
  }

  function navigateTo(routeId) {
    navLinks.forEach(link => {
      link.classList.toggle('is-active', link.dataset.route === routeId);
    });

    pages.forEach(page => {
      if (page.dataset.page === routeId) {
        page.classList.add('is-active');
        void page.offsetWidth;
      } else {
        page.classList.remove('is-active');
      }
    });

    if (routeId === 'results') {
      triggerGaugeAnimation();
    }
    if (routeId === 'history') {
      renderHistory();
    }
    if (mobileNav && mobileNav.classList.contains('is-open')) {
      mobileNav.classList.remove('is-open');
      mobileNav.style.display = '';
    }
  }

  routeButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      navigateTo(btn.dataset.route);
    });
  });

  // === HEALTH CHECK ===
  const statusPill = document.querySelector('.status-pill');
  const statusDot = document.querySelector('.status-dot');
  let statusText = statusPill.querySelector('.status-text');
  if (!statusText) {
    statusText = document.createElement('span');
    statusText.className = 'status-text';
    statusPill.innerHTML = '';
    statusPill.appendChild(statusDot);
    statusPill.appendChild(statusText);
  }
  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (!res.ok) throw new Error('offline');
      statusText.textContent = 'API Ready';
      statusPill.style.color = 'var(--success)';
      statusDot.style.background = 'var(--success)';
      statusDot.style.boxShadow = '0 0 8px var(--success)';
    } catch {
      statusText.textContent = 'API Offline';
      statusPill.style.color = 'var(--danger)';
      statusDot.style.background = 'var(--danger)';
      statusDot.style.boxShadow = '0 0 8px var(--danger)';
    }
  }
  checkHealth();
  setInterval(checkHealth, 15000);

  // === UPLOAD / DROPZONE ===
  const dropzone = document.querySelector('[data-dropzone]');
  const fileInput = document.querySelector('[data-file-input]');
  const feedbackPanel = document.querySelector('.upload-feedback');
  const previewEmpty = document.querySelector('[data-preview-empty]');
  const previewImage = document.querySelector('[data-preview-image]');
  const previewOverlay = document.querySelector('[data-preview-overlay]');
  const fileNameDisplay = document.querySelector('[data-file-name]');
  const fileSizeDisplay = document.querySelector('[data-file-size]');
  const fileTypeDisplay = document.querySelector('[data-file-type]');
  const progFill = document.querySelector('[data-progress-fill]');
  const prepLabel = document.querySelector('[data-preprocess-label]');
  const analyzeBtn = document.querySelector('[data-analyze]');
  const analyzeCopy = document.querySelector('[data-analyze-copy]');
  const uploadError = document.querySelector('[data-upload-error]');

  function resetUploadError() {
    uploadError.textContent = 'Formats are validated before analysis begins.';
    uploadError.style.color = 'var(--text-300)';
  }

  function showUploadError(message) {
    uploadError.textContent = message;
    uploadError.style.color = 'var(--danger)';
  }

  function handleFile(file) {
    if (!file) return;
    if (file.size > 20 * 1024 * 1024) {
      showUploadError('File too large. Max size is 20MB.');
      return;
    }

    state.file = file;
    resetUploadError();

    dropzone.style.display = 'none';
    feedbackPanel.classList.add('is-visible');
    previewEmpty.classList.add('hidden');
    previewOverlay.classList.remove('hidden');

    const isDicom = file.name.toLowerCase().endsWith('.dcm');
    if (isDicom) {
      previewImage.classList.add('hidden');
    } else {
      const objUrl = URL.createObjectURL(file);
      previewImage.src = objUrl;
      previewImage.classList.remove('hidden');
    }

    fileNameDisplay.textContent = file.name;
    fileSizeDisplay.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
    fileTypeDisplay.textContent = isDicom ? 'DICOM' : file.type.includes('png') ? 'PNG' : 'JPG';

    prepLabel.textContent = 'Analyzing headers...';
    progFill.style.width = '30%';

    setTimeout(() => {
      prepLabel.textContent = 'Normalizing contrast...';
      progFill.style.width = '70%';
    }, 400);

    setTimeout(() => {
      prepLabel.textContent = 'Ready for inference';
      progFill.style.width = '100%';
      progFill.style.background = 'var(--success)';

      analyzeBtn.removeAttribute('disabled');
      analyzeCopy.textContent = 'Model ready to process image';
    }, 800);
  }

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
  });

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); });
  });

  dropzone.addEventListener('dragover', () => dropzone.classList.add('is-dragover'));
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('is-dragover'));
  dropzone.addEventListener('drop', (e) => {
    dropzone.classList.remove('is-dragover');
    if (e.dataTransfer.files.length) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  // === FORM LOGIC ===
  const ageInput = document.querySelector('[data-age]');
  ageInput.addEventListener('input', e => state.patient.age = e.target.value);

  const sexButtons = document.querySelectorAll('[data-sex]');
  sexButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      sexButtons.forEach(b => b.classList.remove('is-selected'));
      btn.classList.add('is-selected');
      state.patient.sex = btn.dataset.sex;
    });
  });

  const symptomButtons = document.querySelectorAll('[data-symptom]');
  symptomButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const sym = btn.dataset.symptom;
      if (state.patient.symptoms.has(sym)) {
        state.patient.symptoms.delete(sym);
        btn.classList.remove('is-active');
      } else {
        state.patient.symptoms.add(sym);
        btn.classList.add('is-active');
      }
    });
  });

  const spo2Input = document.querySelector('[data-spo2]');
  const spo2Indicator = document.querySelector('[data-spo2-indicator]');
  const spo2Copy = document.querySelector('[data-spo2-copy]');

  spo2Input.addEventListener('input', e => {
    const val = parseInt(e.target.value, 10);
    state.patient.spo2 = e.target.value;

    spo2Indicator.className = 'spo2-indicator';
    if (isNaN(val)) {
      spo2Indicator.classList.add('status-neutral');
      spo2Copy.textContent = 'Live triage color updates based on oxygen saturation.';
      spo2Copy.style.color = 'var(--text-500)';
    } else if (val >= 95) {
      spo2Indicator.classList.add('status-good');
      spo2Copy.textContent = 'Normal saturation level.';
      spo2Copy.style.color = 'var(--success)';
    } else if (val >= 90) {
      spo2Indicator.classList.add('status-warn');
      spo2Copy.textContent = 'Caution: Slight hypoxia detected.';
      spo2Copy.style.color = 'var(--warning)';
    } else {
      spo2Indicator.classList.add('status-bad');
      spo2Copy.textContent = 'Warning: Severe hypoxia trigger added to Red Flags.';
      spo2Copy.style.color = 'var(--danger)';
    }
  });

  const urgencyButtons = document.querySelectorAll('[data-urgency]');
  const urgencyLabel = document.querySelector('[data-urgency-label]');
  urgencyButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      urgencyButtons.forEach(b => b.classList.remove('is-selected'));
      btn.classList.add('is-selected');
      state.patient.urgency = btn.dataset.urgency;
      urgencyLabel.textContent = state.patient.urgency + ' Review';

      if (state.patient.urgency === 'Urgent') {
        urgencyLabel.style.background = 'var(--danger-bg)';
        urgencyLabel.style.color = 'var(--danger)';
      } else {
        urgencyLabel.style.background = 'rgba(8,131,149,0.08)';
        urgencyLabel.style.color = 'var(--primary)';
      }
    });
  });

  const historyInput = document.querySelector('[data-history]');
  const historyCount = document.querySelector('[data-history-count]');
  historyInput.addEventListener('input', e => {
    state.patient.history = e.target.value;
    historyCount.textContent = `${e.target.value.length} / 240`;
  });

  // === ANALYZE BACKEND ===
  analyzeBtn.addEventListener('click', async () => {
    if (state.isAnalyzing || !state.file) return;
    state.isAnalyzing = true;
    resetUploadError();

    analyzeBtn.classList.add('is-loading');
    analyzeBtn.querySelector('span').textContent = 'Processing...';
    const stages = ['Uploading image...', 'Running model...', 'Retrieving guidance...'];
    let stageIndex = 0;
    analyzeCopy.textContent = stages[stageIndex];
    const stageTimer = setInterval(() => {
      stageIndex = (stageIndex + 1) % stages.length;
      analyzeCopy.textContent = stages[stageIndex];
    }, 1200);

    const formData = new FormData();
    formData.append('image', state.file);
    if (state.patient.age) formData.append('age', state.patient.age);
    if (state.patient.sex) formData.append('sex', state.patient.sex);
    if (state.patient.spo2) formData.append('spo2', state.patient.spo2);
    if (state.patient.urgency) formData.append('urgency', state.patient.urgency);
    if (state.patient.history) formData.append('history', state.patient.history);
    if (state.patient.symptoms.size) {
      formData.append('symptoms', JSON.stringify(Array.from(state.patient.symptoms)));
    }

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || 'Backend request failed.');
      }

      analyzeCopy.textContent = 'Processing model response...';
      const payload = await response.json();
      const normalized = normalizeResponse(payload);

      updateResults(normalized);
      addHistoryItem(normalized);
      saveCases();

      clearInterval(stageTimer);
      analyzeBtn.classList.remove('is-loading');
      analyzeBtn.querySelector('span').textContent = 'Analyze Case';
      analyzeCopy.textContent = 'Upload an image to continue';
      state.isAnalyzing = false;

      navigateTo('results');
    } catch (err) {
      clearInterval(stageTimer);
      state.isAnalyzing = false;
      analyzeBtn.classList.remove('is-loading');
      analyzeBtn.querySelector('span').textContent = 'Analyze Case';
      analyzeCopy.textContent = 'Upload an image to continue';
      showUploadError(err.message || 'Unable to reach the backend.');
    }
  });

  function normalizeResponse(payload) {
    const prob = pickNumber(
      payload.probability ??
      payload.pneumonia_probability ??
      payload.prob_pneu ??
      payload.score ??
      payload.pred_prob,
      0
    );

    let risk = (payload.risk || payload.risk_level || '').toString().toLowerCase();
    if (!['low', 'moderate', 'high'].includes(risk)) {
      risk = prob >= 0.7 ? 'high' : prob >= 0.4 ? 'moderate' : 'low';
    }

    const gradcamUrl = resolveUrl(payload.gradcam_url || payload.gradcam_path) ||
      toDataUrl(payload.gradcam_base64) ||
      toDataUrl(payload.heatmap);

    const imageUrl = resolveUrl(payload.original_url || payload.image_url) ||
      toDataUrl(payload.image_base64);

    const knowledgeRaw = payload.knowledge || payload.rag_results || payload.retrieved || [];
    const knowledge = Array.isArray(knowledgeRaw) ? knowledgeRaw.map((item, idx) => ({
      id: item.id || idx,
      source: item.source || item.doc || item.title || `Source ${idx + 1}`,
      score: pickNumber(item.score ?? item.relevance ?? item.similarity, 0.5),
      snippet: item.snippet || item.text || item.content || ''
    })) : [];

    const nextSteps = toArray(payload.next_steps || payload.steps || payload.recommendations);
    const redFlags = toArray(payload.red_flags || payload.alerts);

    return {
      id: payload.id || payload.case_id || payload.request_id || `CXR-${Date.now()}`,
      createdAt: payload.created_at || new Date().toISOString(),
      probability: prob,
      risk,
      modelSummary: payload.model_summary || payload.summary || 'Model inference completed.',
      confidence: pickNumber(payload.confidence, prob),
      imageUrl,
      gradcamUrl,
      knowledge,
      nextSteps,
      redFlags,
      clinical: { ...state.patient }
    };
  }

  function resolveUrl(value) {
    if (!value) return null;
    if (value.startsWith('http://') || value.startsWith('https://') || value.startsWith('data:')) return value;
    return `${API_BASE}${value.startsWith('/') ? '' : '/'}${value}`;
  }

  function toDataUrl(value) {
    if (!value) return null;
    if (value.startsWith('data:')) return value;
    return `data:image/png;base64,${value}`;
  }

  function pickNumber(value, fallback = 0) {
    if (typeof value === 'number' && !Number.isNaN(value)) return value;
    if (typeof value === 'string') {
      const parsed = Number(value);
      if (!Number.isNaN(parsed)) return parsed;
    }
    return fallback;
  }

  function toArray(value) {
    if (Array.isArray(value)) return value;
    if (typeof value === 'string') return [value];
    return [];
  }

  function updateResults(result) {
    state.currentResult = result;
    // Clinical recap
    const dl = document.querySelector('[data-recap-list]');
    dl.innerHTML = `
      <div class="recap-item"><dt>Age / Sex</dt><dd>${result.clinical.age || 'N/A'}, ${result.clinical.sex}</dd></div>
      <div class="recap-item"><dt>SpO2</dt><dd>${result.clinical.spo2 ? result.clinical.spo2 + '%' : 'N/A'}</dd></div>
      <div class="recap-item"><dt>Symptoms</dt><dd>${Array.from(result.clinical.symptoms).join(', ') || 'None reported'}</dd></div>
      <div class="recap-item"><dt>History</dt><dd class="truncate" title="${result.clinical.history}">${result.clinical.history || 'None'}</dd></div>
    `;

    // Risk badge + gauge
    const badge = document.querySelector('[data-risk-badge]');
    const riskLabel = document.querySelector('[data-risk-label]');
    const riskIcon = document.querySelector('[data-risk-icon]');
    const gaugeValue = document.querySelector('[data-gauge-value]');
    const gaugeFill = document.querySelector('[data-gauge-fill]');
    const confidenceCopy = document.querySelector('[data-model-confidence]');

    badge.className = `risk-badge risk-${result.risk}`;
    riskLabel.textContent = `${capitalize(result.risk)} Risk`;
    riskIcon.textContent = result.risk === 'low' ? '✓' : result.risk === 'high' ? '!' : 'i';

    const probPercent = Math.round(result.probability * 100);
    gaugeValue.textContent = `${probPercent}%`;
    const offset = 364 - ((probPercent / 100) * 364);
    gaugeFill.style.strokeDashoffset = offset;

    if (result.risk === 'high') gaugeFill.style.stroke = 'var(--danger)';
    else if (result.risk === 'low') gaugeFill.style.stroke = 'var(--success)';
    else gaugeFill.style.stroke = 'var(--warning)';

    confidenceCopy.textContent = `Model confidence: ${result.confidence.toFixed(2)}`;

    // Viewer images
    const viewerImg = document.querySelector('[data-viewer-base]');
    const heatmapLayer = document.querySelector('[data-viewer-heatmap]');
    const heatmapImg = document.querySelector('[data-viewer-heatmap-image]');
    if (result.imageUrl) viewerImg.src = result.imageUrl;
    else if (state.file) viewerImg.src = URL.createObjectURL(state.file);

    const heatmapToggle = document.querySelector('[data-view-mode="heatmap"]');
    const originalToggle = document.querySelector('[data-view-mode="original"]');
    if (result.gradcamUrl) {
      heatmapImg.src = result.gradcamUrl;
      heatmapLayer.classList.remove('hidden');
      heatmapToggle.classList.remove('is-disabled');
    } else {
      heatmapImg.removeAttribute('src');
      heatmapLayer.classList.add('hidden');
      heatmapToggle.classList.add('is-disabled');
    }
    originalToggle.classList.add('is-selected');
    heatmapToggle.classList.remove('is-selected');

    // Next steps
    const stepsPanel = document.querySelector('[data-tab-panel="next-steps"]');
    stepsPanel.innerHTML = '';
    const steps = result.nextSteps.length ? result.nextSteps : [
      'Correlate the radiographic pattern with symptoms, vitals, and oxygen saturation.',
      'Review the heatmap for focal attention before deciding on escalation.',
      'Consider antibiotics and follow-up imaging if clinically indicated.'
    ];
    steps.forEach((step, idx) => {
      const card = document.createElement('div');
      card.className = 'step-card glass glass-inset';
      card.innerHTML = `<span>${idx + 1}</span><p>${step}</p>`;
      stepsPanel.appendChild(card);
    });

    // Knowledge base
    const knowledgePanel = document.querySelector('[data-tab-panel="knowledge-base"]');
    knowledgePanel.innerHTML = '';
    const knowledge = result.knowledge.length ? result.knowledge : [{
      source: 'No knowledge retrieved',
      score: 0.0,
      snippet: 'No external clinical guidance returned for this case.'
    }];
    knowledge.forEach((item, idx) => {
      const details = document.createElement('details');
      details.className = 'knowledge-card glass glass-inset';
      details.open = idx === 0;
      details.innerHTML = `
        <summary>
          <div>
            <strong>${item.source}</strong>
            <span>Relevance ${(item.score * 100).toFixed(0)}</span>
          </div>
          <span class="score-bar"><i style="width: ${(item.score * 100).toFixed(0)}%"></i></span>
        </summary>
        <p>${item.snippet || 'No summary provided.'}</p>
      `;
      knowledgePanel.appendChild(details);
    });

    // Red flags
    const redFlagsPanel = document.querySelector('[data-tab-panel="red-flags"]');
    redFlagsPanel.innerHTML = '';
    const flags = result.redFlags.length ? result.redFlags : [
      'SpO2 < 90% on room air',
      'Severe respiratory distress or altered sensorium',
      'Rapid progression of symptoms',
      'Hypotension or signs of sepsis'
    ];
    flags.forEach(flag => {
      const row = document.createElement('label');
      row.className = 'check-row';
      row.innerHTML = `<input type="checkbox" /><span>${flag}</span>`;
      redFlagsPanel.appendChild(row);
    });

    if (parseInt(result.clinical.spo2, 10) < 90) {
      const redTab = document.querySelector('[data-tab="red-flags"]');
      const redPanel = document.querySelector('[data-tab-panel="red-flags"]');
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('is-active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('is-active'));
      redTab.classList.add('is-active');
      redPanel.classList.add('is-active');
    }
  }

  function triggerGaugeAnimation() {
    const fill = document.querySelector('[data-gauge-fill]');
    if (!fill) return;
    const storedOffset = fill.style.strokeDashoffset;
    fill.style.transition = 'none';
    fill.style.strokeDashoffset = '364';

    setTimeout(() => {
      fill.style.transition = 'stroke-dashoffset 1.5s var(--ease-bounce), stroke 0.5s';
      fill.style.strokeDashoffset = storedOffset;
    }, 100);
  }

  // === VIEWER TOGGLE ===
  const viewModes = document.querySelectorAll('[data-view-mode]');
  const heatmapLayer = document.querySelector('[data-viewer-heatmap]');
  const viewerFrame = document.querySelector('[data-viewer-frame]');
  const zoomInBtn = document.querySelector('[data-zoom-in]');
  const zoomOutBtn = document.querySelector('[data-zoom-out]');
  let zoomLevel = 1;

  // === GUIDANCE TABS ===
  const tabs = document.querySelectorAll('.tab');
  const panels = document.querySelectorAll('.tab-panel');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('is-active'));
      panels.forEach(p => p.classList.remove('is-active'));
      tab.classList.add('is-active');
      const panel = document.querySelector(`[data-tab-panel="${tab.dataset.tab}"]`);
      if (panel) panel.classList.add('is-active');
    });
  });

  function applyZoom() {
    if (!viewerFrame) return;
    viewerFrame.style.transform = `scale(${zoomLevel})`;
  }

  if (zoomInBtn && zoomOutBtn) {
    zoomInBtn.addEventListener('click', () => {
      zoomLevel = Math.min(2.5, Number((zoomLevel + 0.1).toFixed(2)));
      applyZoom();
    });
    zoomOutBtn.addEventListener('click', () => {
      zoomLevel = Math.max(1, Number((zoomLevel - 0.1).toFixed(2)));
      applyZoom();
    });
  }

  viewModes.forEach(btn => {
    btn.addEventListener('click', () => {
      if (btn.classList.contains('is-disabled')) return;
      viewModes.forEach(b => b.classList.remove('is-selected'));
      btn.classList.add('is-selected');
      if (btn.dataset.viewMode === 'heatmap') {
        heatmapLayer.classList.remove('hidden');
      } else {
        heatmapLayer.classList.add('hidden');
      }
    });
  });

  const opacityRange = document.querySelector('[data-opacity]');
  const opacityCopy = document.querySelector('[data-opacity-copy]');
  opacityRange.addEventListener('input', e => {
    const val = e.target.value;
    opacityCopy.textContent = `${val}%`;
    heatmapLayer.style.opacity = (val / 100).toFixed(2);
  });

  const downloadOriginalBtn = document.querySelector('[data-download-original]');
  const downloadOverlayBtn = document.querySelector('[data-download-overlay]');
  const downloadReportBtn = document.querySelector('[data-download-report]');

  function downloadImage(url, filename) {
    if (!url) return;
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
  }

  downloadOriginalBtn.addEventListener('click', () => {
    const url = state.currentResult?.imageUrl;
    if (url) downloadImage(url, 'xray.png');
  });

  downloadOverlayBtn.addEventListener('click', () => {
    const url = state.currentResult?.gradcamUrl;
    if (url) downloadImage(url, 'gradcam.png');
  });

  downloadReportBtn.addEventListener('click', () => {
    window.print();
  });

  // === HISTORY ===
  const historyBody = document.querySelector('[data-history-body]');
  const emptyState = document.querySelector('[data-empty-state]');
  const historyWrap = document.querySelector('[data-history-wrap]');
  const historySearch = document.querySelector('[data-history-search]');
  const filterButtons = document.querySelectorAll('[data-filter]');

  function addHistoryItem(result) {
    state.cases.unshift(result);
  }

  function renderHistory() {
    const query = historySearch.value.trim().toLowerCase();
    const activeFilter = document.querySelector('.filter-chip.is-active')?.dataset.filter || 'All';

    const filtered = state.cases.filter(item => {
      const matchesQuery = !query || item.id.toLowerCase().includes(query) || item.risk.toLowerCase().includes(query);
      const matchesFilter = activeFilter === 'All' || item.risk.toLowerCase() === activeFilter.toLowerCase();
      return matchesQuery && matchesFilter;
    });

    historyBody.innerHTML = '';
    if (!filtered.length) {
      historyWrap.classList.add('hidden');
      emptyState.classList.remove('hidden');
      return;
    }

    historyWrap.classList.remove('hidden');
    emptyState.classList.add('hidden');

    filtered.forEach(item => {
      const tr = document.createElement('tr');
      const now = new Date(item.createdAt);
      const riskBadgeHtml = getRiskBadgeHtml(item.risk);
      const prob = Math.round(item.probability * 100);
      tr.innerHTML = `
        <td>
          <div style="font-weight: 500">${now.toLocaleDateString()}</div>
          <div style="font-size: 11px; color: var(--text-500)">${now.toLocaleTimeString([], {timeStyle: 'short'})}</div>
        </td>
        <td>${item.id}</td>
        <td>${riskBadgeHtml}</td>
        <td>${prob}%</td>
        <td>
           <button class="cta-secondary compact" style="padding: 6px 12px; font-size: 11px;" type="button" data-route="results">View</button>
        </td>
      `;
      tr.querySelector('[data-route="results"]').addEventListener('click', () => {
        updateResults(item);
        navigateTo('results');
      });
      historyBody.appendChild(tr);
    });
  }

  historySearch.addEventListener('input', renderHistory);
  filterButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      filterButtons.forEach(b => b.classList.remove('is-active'));
      btn.classList.add('is-active');
      renderHistory();
    });
  });

  function getRiskBadgeHtml(risk) {
    if (risk === 'low') return `<div class="risk-badge risk-low"><span class="risk-icon">✓</span><span>Low</span></div>`;
    if (risk === 'moderate') return `<div class="risk-badge risk-moderate"><span class="risk-icon">i</span><span>Moderate</span></div>`;
    return `<div class="risk-badge risk-high" style="animation:none; box-shadow:none;"><span class="risk-icon">!</span><span>High</span></div>`;
  }

  function saveCases() {
    sessionStorage.setItem('nuvexaCases', JSON.stringify(state.cases));
  }

  function loadCases() {
    try {
      const raw = sessionStorage.getItem('nuvexaCases');
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  function capitalize(text) {
    return text.charAt(0).toUpperCase() + text.slice(1);
  }

  // Initialize history state
  renderHistory();
});
