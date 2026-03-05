const state = {
  transactions: [],
  analysis: null,
  currentMonth: null,
  tips: [],
  tipIndex: 0,
  selectedFile: null,
  coachHistory: [],
};

const FALLBACK_FACTS = [
  "Compounding works best when contributions are consistent over time.",
  "Emergency funds are commonly structured around 3 to 6 months of essential expenses.",
  "Paying credit card balances in full helps avoid revolving interest charges.",
  "Tracking fixed and variable expenses improves next-month forecast quality.",
];

let fallbackFactIndex = 0;

const currency = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

function qs(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatCoachMessage(message) {
  const escaped = escapeHtml(message);
  const bolded = escaped.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  return bolded.replaceAll("\n", "<br/>");
}

function nextFallbackFact() {
  const fact = FALLBACK_FACTS[fallbackFactIndex % FALLBACK_FACTS.length];
  fallbackFactIndex += 1;
  return fact;
}

function setFeedback(message) {
  qs("input-feedback").textContent = message;
}

async function jsonRequest(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed (${response.status})`);
  }
  return response.json();
}

async function uploadStatement(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch("/api/ingest", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Ingestion failed (${response.status})`);
  }
  return response.json();
}

function parseCsvLine(line) {
  const values = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      values.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  values.push(current.trim());
  return values;
}

function parseCsvFlexible(text) {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0) {
    throw new Error("No transaction rows were provided.");
  }

  const expected = ["date", "description", "merchant", "amount", "type"];
  const firstRow = parseCsvLine(lines[0]).map((cell) => cell.toLowerCase());
  const hasHeader = expected.every((column) => firstRow.includes(column));

  const rows = hasHeader ? lines.slice(1) : lines;
  const header = hasHeader ? firstRow : expected;
  const index = Object.fromEntries(header.map((column, i) => [column, i]));
  const parsed = [];

  rows.forEach((line, offset) => {
    const cells = parseCsvLine(line);
    if (cells.length < 4) {
      return;
    }
    const amount = Number(cells[index.amount ?? 3]);
    if (!Number.isFinite(amount)) {
      throw new Error(`Invalid amount at row ${offset + (hasHeader ? 2 : 1)}.`);
    }
    parsed.push({
      date: cells[index.date ?? 0],
      description: cells[index.description ?? 1],
      merchant: cells[index.merchant ?? 2] || cells[index.description ?? 1],
      amount,
      type: String(cells[index.type ?? 4] || "debit").toLowerCase() === "credit" ? "credit" : "debit",
    });
  });

  if (parsed.length === 0) {
    throw new Error("No valid transaction rows found.");
  }
  return parsed;
}

function monthOptions(transactions) {
  return Array.from(
    new Set(
      transactions
        .map((transaction) => new Date(transaction.date))
        .filter((date) => Number.isFinite(date.getTime()))
        .map(
          (date) =>
            `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}`,
        ),
    ),
  ).sort();
}

function renderMonthSelect() {
  const select = qs("month-select");
  const months = monthOptions(state.transactions);
  if (months.length === 0) {
    select.innerHTML = `<option value="">No valid month</option>`;
    state.currentMonth = null;
    return;
  }
  select.innerHTML = months.map((month) => `<option value="${month}">${month}</option>`).join("");
  if (!state.currentMonth || !months.includes(state.currentMonth)) {
    state.currentMonth = months[months.length - 1];
  }
  select.value = state.currentMonth;
}

function renderDataSummary() {
  if (state.transactions.length === 0) {
    qs("data-summary").textContent = "No transactions loaded.";
    return;
  }
  const validDates = state.transactions
    .map((transaction) => new Date(transaction.date))
    .filter((date) => Number.isFinite(date.getTime()))
    .sort((left, right) => left - right);
  const start = validDates[0]?.toISOString().slice(0, 10) || "unknown";
  const end = validDates[validDates.length - 1]?.toISOString().slice(0, 10) || "unknown";
  qs("data-summary").textContent =
    `${state.transactions.length} transactions loaded (${start} to ${end}).`;
}

function renderPreviewTable() {
  const body = qs("preview-table");
  if (state.transactions.length === 0) {
    body.innerHTML = `<tr><td colspan="5">No data loaded yet.</td></tr>`;
    return;
  }
  body.innerHTML = state.transactions
    .slice(0, 80)
    .map((transaction) => {
      return `
      <tr>
        <td>${escapeHtml(transaction.date)}</td>
        <td>${escapeHtml(transaction.description)}</td>
        <td>${escapeHtml(transaction.merchant || transaction.description)}</td>
        <td>${currency.format(Math.abs(transaction.amount))}</td>
        <td>${escapeHtml(transaction.type || "debit")}</td>
      </tr>
    `;
    })
    .join("");
}

function renderKpis(analysis) {
  qs("kpi-spending").textContent = currency.format(analysis.monthly_spending);
  const mom = analysis.month_over_month || {};
  const deltaPct = Number(mom.delta_pct || 0);
  const deltaAmount = Number(mom.delta_amount || 0);
  const spendingIncreased = deltaPct > 0;
  const spendingReduced = deltaPct < 0;
  const arrow = spendingIncreased ? "▼" : spendingReduced ? "▲" : "•";
  const momNode = qs("kpi-mom");
  momNode.classList.remove("up", "down", "neutral");
  if (spendingIncreased) {
    momNode.classList.add("down");
  } else if (spendingReduced) {
    momNode.classList.add("up");
  } else {
    momNode.classList.add("neutral");
  }
  momNode.textContent = `${arrow} ${(Math.abs(deltaPct) * 100).toFixed(1)}% (${currency.format(
    Math.abs(deltaAmount),
  )})`;
  const rawScore = Number(analysis.financial_score?.score || 0);
  const scoreOutOfTen = rawScore / 10;
  const scoreNode = qs("kpi-score");
  scoreNode.classList.remove("score-low", "score-medium", "score-high");
  if (scoreOutOfTen < 5) {
    scoreNode.classList.add("score-low");
  } else if (scoreOutOfTen < 7) {
    scoreNode.classList.add("score-medium");
  } else {
    scoreNode.classList.add("score-high");
  }
  scoreNode.textContent = `${scoreOutOfTen.toFixed(1)} / 10`;
  qs("kpi-waste").textContent = currency.format(analysis.subscription_waste);
}

function renderTipCard(forceNext = false) {
  if (state.tips.length === 0) {
    qs("tip-card").textContent = "Run analysis to get monthly insights.";
    return;
  }
  if (forceNext) {
    state.tipIndex = (state.tipIndex + 1) % state.tips.length;
  } else if (state.tipIndex >= state.tips.length) {
    state.tipIndex = 0;
  }
  qs("tip-card").textContent = state.tips[state.tipIndex];
}

function openTipModal(text, title = "Financial Tip") {
  qs("tip-title").textContent = title;
  qs("tip-modal-content").textContent = text;
  qs("tip-modal").classList.remove("hidden");
}

function closeTipModal() {
  qs("tip-modal").classList.add("hidden");
}

function renderCategoryBars(analysis) {
  const list = analysis.category_breakdown || [];
  if (list.length === 0) {
    qs("category-bars").innerHTML = `<article class="bar-item">No category data.</article>`;
    return;
  }
  const max = Math.max(...list.map((item) => item.amount), 1);
  qs("category-bars").innerHTML = list
    .slice(0, 10)
    .map(
      (item) => `
      <article class="bar-item">
        <div class="bar-top">
          <strong>${escapeHtml(item.category)}</strong>
          <span>${currency.format(item.amount)} (${(item.share * 100).toFixed(1)}%)</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${(item.amount / max) * 100}%"></div>
        </div>
      </article>
    `,
    )
    .join("");
}

function renderPriorities(analysis) {
  const priorities = analysis.spending_priorities || [];
  if (priorities.length === 0) {
    qs("priority-list").innerHTML = `<article class="stack-item">No priority scoring available.</article>`;
    return;
  }
  qs("priority-list").innerHTML = priorities
    .map(
      (item) => `
      <article class="stack-item">
        <div class="stack-top">
          <strong>${escapeHtml(item.category)}</strong>
          <span class="badge ${escapeHtml(item.signal)}">${escapeHtml(item.signal)}</span>
        </div>
        <p>Share ${(item.share * 100).toFixed(1)}% · priority points ${item.points}</p>
      </article>
    `,
    )
    .join("");
}

function renderForecast(analysis) {
  const forecast = analysis.forecast || {};
  const baseline = Number(forecast.baseline_next_month || 0);
  const optimized = Number(forecast.optimized_next_month || 0);
  const saving = Number(forecast.potential_saving || 0);
  const savingClass = saving > 0 ? "good" : "bad";
  qs("forecast-box").innerHTML = `
    <p><strong>Baseline next month:</strong> ${currency.format(baseline)}</p>
    <p><strong>Optimized next month:</strong> ${currency.format(optimized)}</p>
    <p class="${savingClass}"><strong>Potential saving:</strong> ${currency.format(saving)}</p>
  `;

  const trend = analysis.monthly_trend || [];
  if (trend.length === 0) {
    qs("trend-bars").innerHTML = "";
    return;
  }
  const max = Math.max(...trend.map((item) => item.amount), baseline, 1);
  const trendRows = trend
    .map(
      (item) => `
      <article class="bar-item">
        <div class="bar-top">
          <strong>${escapeHtml(item.month)}</strong>
          <span>${currency.format(item.amount)}</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${(item.amount / max) * 100}%"></div>
        </div>
      </article>
    `,
    )
    .join("");
  const forecastRow = `
    <article class="bar-item">
      <div class="bar-top">
        <strong>Forecast (baseline)</strong>
        <span>${currency.format(baseline)}</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${(baseline / max) * 100}%"></div>
      </div>
    </article>
  `;
  qs("trend-bars").innerHTML = `${trendRows}${forecastRow}`;
}

function renderSubscriptions(analysis) {
  const list = analysis.recurring_subscriptions || [];
  if (list.length === 0) {
    qs("subscription-list").innerHTML = `<article class="stack-item">No recurring subscriptions detected.</article>`;
    return;
  }
  qs("subscription-list").innerHTML = list
    .map(
      (item) => `
      <article class="stack-item">
        <div class="stack-top">
          <strong>${escapeHtml(item.merchant)}</strong>
          <span>${currency.format(item.monthly_estimate)}</span>
        </div>
        <p>${escapeHtml(item.category)} · ${escapeHtml(item.priority)} · estimated waste ${currency.format(item.waste_estimate)}</p>
      </article>
    `,
    )
    .join("");
}

function renderAnomalies(analysis) {
  const list = analysis.anomalies || [];
  if (list.length === 0) {
    qs("anomaly-list").innerHTML = `<article class="stack-item">No anomaly alerts for the selected month.</article>`;
    return;
  }
  qs("anomaly-list").innerHTML = list
    .map(
      (item) => `
      <article class="stack-item">
        <div class="stack-top">
          <strong>${escapeHtml(item.merchant)}</strong>
          <span class="badge ${escapeHtml(item.severity)}">${escapeHtml(item.severity)}</span>
        </div>
        <p>${escapeHtml(item.date)} · ${currency.format(item.amount)} · ${escapeHtml(item.category)} · confidence ${Number(item.anomaly_confidence || 0).toFixed(1)}%</p>
        <p>${escapeHtml(item.reasons.join("; "))}</p>
      </article>
    `,
    )
    .join("");
}

function renderQualityQueue(analysis) {
  const rows = (analysis.review_queue || []).slice(0, 10);
  if (rows.length === 0) {
    qs("quality-list").innerHTML =
      `<article class="stack-item">No low-confidence category rows this month.</article>`;
    return;
  }
  qs("quality-list").innerHTML = rows
    .map(
      (item) => `
      <article class="stack-item">
        <div class="stack-top">
          <strong>${escapeHtml(item.merchant)}</strong>
          <span class="badge review">Needs review</span>
        </div>
        <p>${escapeHtml(item.date)} · ${escapeHtml(item.category)} · ${currency.format(item.amount)}</p>
        <p>${escapeHtml(item.description)}</p>
        <p>Category confidence: ${Number(item.category_confidence || 0).toFixed(1)}%</p>
        <p>Why shown: category prediction has lower confidence than usual.</p>
      </article>
    `,
    )
    .join("");
}

function renderAnalysis(analysis) {
  renderKpis(analysis);
  renderCategoryBars(analysis);
  renderPriorities(analysis);
  renderForecast(analysis);
  renderSubscriptions(analysis);
  renderAnomalies(analysis);
  renderQualityQueue(analysis);
  state.tips = analysis.tips || [];
  state.tipIndex = 0;
  renderTipCard(false);
  renderPreviewTable();
}

function appendCoachMessage(role, message) {
  state.coachHistory.push({ role, message });
  const chat = qs("coach-chat");
  const roleClass = role === "user" ? "user" : "assistant";
  const content = role === "assistant" ? formatCoachMessage(message) : escapeHtml(message).replaceAll("\n", "<br/>");
  chat.insertAdjacentHTML(
    "beforeend",
    `<article class="chat-msg ${roleClass}">${content}</article>`,
  );
  chat.scrollTop = chat.scrollHeight;
}

function showCoachTyping() {
  const chat = qs("coach-chat");
  if (document.getElementById("coach-typing-msg")) {
    return;
  }
  chat.insertAdjacentHTML(
    "beforeend",
    `<article id="coach-typing-msg" class="chat-msg assistant typing" aria-label="Coach is typing">
      <span class="typing-dots"><span></span><span></span><span></span></span>
    </article>`,
  );
  chat.scrollTop = chat.scrollHeight;
}

function hideCoachTyping() {
  const node = document.getElementById("coach-typing-msg");
  if (node) {
    node.remove();
  }
}

function openCoachWindow() {
  qs("coach-window").classList.remove("hidden");
  qs("coach-fab").classList.add("hidden");
  qs("coach-input").focus();
}

function closeCoachWindow() {
  qs("coach-window").classList.add("hidden");
  qs("coach-window").classList.remove("minimized", "maximized");
  qs("coach-fab").classList.remove("hidden");
}

function toggleCoachMinimize() {
  qs("coach-window").classList.toggle("minimized");
}

function toggleCoachMaximize() {
  qs("coach-window").classList.toggle("maximized");
}

async function fetchFactFromBackend() {
  const payload = await jsonRequest("/api/fact");
  const factText = payload.fact || nextFallbackFact();
  return factText;
}

async function askCoach(message) {
  const payload = await jsonRequest("/api/coach", {
    method: "POST",
    body: JSON.stringify({
      message,
      analysis: state.analysis || null,
      history: state.coachHistory.slice(-8),
    }),
  });
  return payload.response || "I could not produce a response right now.";
}

async function runAnalysis() {
  if (state.transactions.length === 0) {
    setFeedback("Load sample data, extract from file, or paste CSV first.");
    return;
  }
  if (!state.currentMonth) {
    setFeedback("No valid month selected.");
    return;
  }
  setFeedback("Running analysis...");
  try {
    const result = await jsonRequest("/api/analyze", {
      method: "POST",
      body: JSON.stringify({
        transactions: state.transactions,
        month: state.currentMonth,
      }),
    });
    state.analysis = result;
    renderAnalysis(result);
    setFeedback(`Analysis complete for ${result.month}.`);
  } catch (error) {
    setFeedback(`Analysis error: ${error.message}`);
  }
}

async function loadSample() {
  const payload = await jsonRequest("/api/sample-transactions");
  state.transactions = payload.transactions || [];
  renderMonthSelect();
  renderDataSummary();
  renderPreviewTable();
  setFeedback(`Loaded ${state.transactions.length} sample transactions.`);
}

function clearAll() {
  state.transactions = [];
  state.analysis = null;
  state.currentMonth = null;
  state.tips = [];
  state.tipIndex = 0;
  state.selectedFile = null;
  state.coachHistory = [];
  qs("paste-input").value = "";
  qs("statement-file").value = "";
  qs("preview-table").innerHTML = `<tr><td colspan="5">No data loaded yet.</td></tr>`;
  qs("category-bars").innerHTML = "";
  qs("priority-list").innerHTML = "";
  qs("trend-bars").innerHTML = "";
  qs("subscription-list").innerHTML = "";
  qs("anomaly-list").innerHTML = "";
  qs("quality-list").innerHTML = "";
  qs("forecast-box").innerHTML = "";
  qs("kpi-spending").textContent = "--";
  qs("kpi-mom").textContent = "--";
  qs("kpi-mom").classList.remove("up", "down");
  qs("kpi-mom").classList.add("neutral");
  qs("kpi-score").textContent = "--";
  qs("kpi-waste").textContent = "--";
  qs("tip-card").textContent = "Run analysis to get monthly insights.";
  qs("coach-chat").innerHTML = "";
  appendCoachMessage(
    "assistant",
    "Ask me about spending optimization, category tradeoffs, or next-month planning.",
  );
  qs("coach-window").classList.add("hidden");
  qs("coach-window").classList.remove("minimized", "maximized");
  qs("coach-fab").classList.add("hidden");
  hideCoachTyping();
  closeTipModal();
  renderMonthSelect();
  renderDataSummary();
  setFeedback("Cleared data.");
}

async function bootstrap() {
  clearAll();
}

qs("statement-file").addEventListener("change", (event) => {
  state.selectedFile = event.target.files?.[0] || null;
});

qs("extract-file").addEventListener("click", async () => {
  if (!state.selectedFile) {
    setFeedback("Choose a PDF, image, or CSV file first.");
    return;
  }
  try {
    setFeedback("Extracting transactions from file...");
    const payload = await uploadStatement(state.selectedFile);
    state.transactions = payload.transactions || [];
    renderMonthSelect();
    renderDataSummary();
    renderPreviewTable();
    const warnings = payload.meta?.warnings || [];
    if (warnings.length > 0) {
      setFeedback(`Extracted ${state.transactions.length} rows. Note: ${warnings.join(" ")}`);
    } else {
      setFeedback(`Extracted ${state.transactions.length} rows from ${payload.meta?.source_type || "file"}.`);
    }
  } catch (error) {
    setFeedback(`Extraction error: ${error.message}`);
  }
});

qs("use-paste").addEventListener("click", () => {
  const raw = qs("paste-input").value.trim();
  if (!raw) {
    setFeedback("Paste CSV text first.");
    return;
  }
  try {
    state.transactions = parseCsvFlexible(raw);
    renderMonthSelect();
    renderDataSummary();
    renderPreviewTable();
    setFeedback(`Loaded ${state.transactions.length} rows from pasted CSV.`);
  } catch (error) {
    setFeedback(`Paste error: ${error.message}`);
  }
});

qs("load-sample").addEventListener("click", async () => {
  try {
    await loadSample();
  } catch (error) {
    setFeedback(`Sample load error: ${error.message}`);
  }
});

qs("clear-data").addEventListener("click", clearAll);

qs("month-select").addEventListener("change", (event) => {
  state.currentMonth = event.target.value || null;
});

qs("analyze-button").addEventListener("click", async () => {
  await runAnalysis();
});

qs("financial-tip").addEventListener("click", async () => {
  try {
    const fact = await fetchFactFromBackend();
    if (fact) {
      openTipModal(fact, "Financial Tip");
      return;
    }
    openTipModal(nextFallbackFact(), "Financial Tip");
  } catch (_error) {
    openTipModal(nextFallbackFact(), "Financial Tip");
  }
});

qs("modal-new-fact").addEventListener("click", async () => {
  try {
    const fact = await fetchFactFromBackend();
    if (fact) {
      openTipModal(fact, "Financial Tip");
      return;
    }
    openTipModal(nextFallbackFact(), "Financial Tip");
  } catch (_error) {
    openTipModal(nextFallbackFact(), "Financial Tip");
  }
});

qs("close-tip-modal").addEventListener("click", closeTipModal);
qs("tip-modal").addEventListener("click", (event) => {
  if (event.target.id === "tip-modal") {
    closeTipModal();
  }
});

qs("open-coach").addEventListener("click", openCoachWindow);
qs("coach-fab").addEventListener("click", openCoachWindow);
qs("coach-close").addEventListener("click", closeCoachWindow);
qs("coach-minimize").addEventListener("click", toggleCoachMinimize);
qs("coach-maximize").addEventListener("click", toggleCoachMaximize);

qs("coach-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = qs("coach-input");
  const message = input.value.trim();
  if (!message) {
    return;
  }
  appendCoachMessage("user", message);
  input.value = "";
  showCoachTyping();
  const typingStartedAt = Date.now();
  try {
    const reply = await askCoach(message);
    const minTypingMs = 450;
    const waitMs = Math.max(0, minTypingMs - (Date.now() - typingStartedAt));
    if (waitMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, waitMs));
    }
    hideCoachTyping();
    appendCoachMessage("assistant", reply);
  } catch (error) {
    const minTypingMs = 450;
    const waitMs = Math.max(0, minTypingMs - (Date.now() - typingStartedAt));
    if (waitMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, waitMs));
    }
    hideCoachTyping();
    appendCoachMessage("assistant", "I could not reach the coach service. Try again in a moment.");
    setFeedback(`Coach error: ${error.message}`);
  }
});

bootstrap();
