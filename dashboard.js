/* All inputs are bundled, sanitized snapshots. There are no network requests. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const state = {record: 0, scenario: 0, filter: 'all'};
  const records = CLIO_RECORDED.records;
  const cases = CLIO_REPLAY.cases;
  const labels = {'conflicting-flow':'Opposing flow','reconfirmed':'Reconfirmed flow','missing-time':'Missing observation time','no-prior-thesis':'No prior thesis exit'};
  const pageNames = {records:'Decision journal',synthetic:'Synthetic cases',about:'Evidence & scope'};
  const text = (tag, value, className) => { const node = document.createElement(tag); node.textContent = value; if (className) node.className = className; return node; };
  function pair(list, label, value) { list.append(text('dt',label), text('dd',value)); }
  function currentRecord() { return records[state.record]; }
  function renderRecords() {
    const body = $('records-body'); body.replaceChildren();
    const indices = records.map((_, index) => index).filter((i) => state.filter === 'all' || records[i].kind === state.filter);
    $('empty-state').hidden = indices.length !== 0;
    if (!indices.includes(state.record) && indices.length) state.record = indices[0];
    for (const i of indices) {
      const row = records[i], tr = document.createElement('tr');
      if (i === state.record) tr.className = 'selected-row';
      const model = row.recorded_model === 'muse-wonder-1.0' ? 'Wonder 1.0' : row.recorded_model === 'muse-spark-1.3' ? 'Spark 1.3' : 'Unknown';
      const cells = [row.anonymous_case.replace('Recorded PAPER case ','Case '), row.action, model, row.attempts[0].status === 'ok' ? 'Recorded success' : 'Unknown'];
      cells.forEach((value, n) => tr.append(text('td', value, n === 2 ? 'model' : n === 3 ? 'small-tag' : '')));
      const td = document.createElement('td'), button = text('button', 'Inspect', 'inspect-button');
      button.type = 'button'; button.setAttribute('aria-label', `Inspect ${cells[0]}, ${row.action}`); button.setAttribute('aria-pressed', String(i === state.record));
      button.addEventListener('click', () => { state.record = i; renderRecords(); renderRecordDetail(); });
      td.append(button); tr.append(td); body.append(tr);
    }
  }
  function renderRecordDetail() {
    const row = currentRecord(), detail = $('record-detail'); detail.replaceChildren();
    $('trace-subtitle').textContent = row.anonymous_case;
    detail.append(text('p',row.action + ' · ' + row.kind,'detail-title'), text('p','Recorded PAPER decision category','detail-label'));
    const list = document.createElement('dl'), attempt = row.attempts[0], jev = row.jev_observed; list.className = 'detail-list';
    pair(list, 'Provider', attempt.final_provider === 'meta' ? 'Meta' : 'Unknown');
    pair(list, 'Requested model', attempt.requested_model);
    pair(list, 'Served model', attempt.served_model);
    pair(list, 'Attempt status', attempt.status === 'ok' ? 'Success' : 'Unknown');
    pair(list, 'Fallback', attempt.fallback_used === false ? 'No' : attempt.fallback_used === true ? 'Yes' : 'Unknown');
    pair(list, 'Expected / matched', `${row.expected_attempts} / ${row.matching_attempts}`);
    pair(list, 'Jev observation', jev.called === true ? (jev.allow === true ? 'Called · allow' : jev.allow === false ? 'Called · not allowed' : 'Called · unknown') : jev.called === false ? 'Not called' : 'Unknown');
    pair(list, 'Jev mode', jev.mode === 'veto' ? 'Veto' : jev.mode === 'shadow' ? 'Shadow' : jev.mode === 'off' ? 'Off' : 'Unknown');
    pair(list, 'Fill / cost', 'Unknown');
    detail.append(list, text('p','Decision category and model attempt only. This record does not establish an order fill or financial outcome.','trace-foot'));
  }
  function renderCases() {
    const list = $('cases-list'); list.replaceChildren();
    cases.forEach((row,i) => {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'scenario' + (i === state.scenario ? ' active' : '');
      b.setAttribute('aria-pressed',String(i === state.scenario));
      b.append(text('span',labels[row.id] || row.id),text('small',row.result.disposition));
      b.addEventListener('click',() => {state.scenario=i;renderCases();renderCaseDetail();}); list.append(b);
    });
  }
  function renderCaseDetail() {
    const row = cases[state.scenario], r = row.result, input = row.inputs, detail = $('case-detail'); detail.replaceChildren();
    detail.append(text('p',r.disposition,'result'),text('p',labels[row.id] + ' · synthetic input','detail-label'));
    const list = document.createElement('dl'); list.className = 'detail-list';
    pair(list,'Direction',input.direction); pair(list,'NOFA flow',r.nofa); pair(list,'Premium flow',r.premium);
    pair(list,'Price confirms',r.price_confirmed === true ? 'Yes' : r.price_confirmed === false ? 'No' : 'Unknown');
    pair(list,'History coverage',r.history_coverage); pair(list,'History scope','Retained positions');
    pair(list,'Provider freshness','Unknown'); pair(list,'Reasons',r.reason_codes.join(', ') || 'None recorded');
    detail.append(list,text('p','Replay of saved source-evaluator output. This does not place or prevent an order.','trace-foot'));
  }
  function selectPage(page) {
    if (!Object.hasOwn(pageNames,page)) return;
    document.querySelectorAll('[data-panel]').forEach((panel) => {panel.hidden = panel.dataset.panel !== page;});
    document.querySelectorAll('[data-page]').forEach((button) => {const active=button.dataset.page===page;button.classList.toggle('active',active);if(active)button.setAttribute('aria-current','page');else button.removeAttribute('aria-current');});
    $('page-title').textContent = pageNames[page];
  }
  function exportJSON(name, data) {
    const blob = new Blob([JSON.stringify(data,null,2) + '\n'],{type:'application/json'});
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href=url; a.download=name; document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url),0);
  }
  document.querySelectorAll('[data-page]').forEach((button) => button.addEventListener('click',() => selectPage(button.dataset.page)));
  $('record-filter').addEventListener('change',(event) => {state.filter=event.target.value;renderRecords();renderRecordDetail();});
  $('reload').addEventListener('click',() => {state.filter='all';state.record=0;$('record-filter').value='all';renderRecords();renderRecordDetail();$('reload').textContent='Snapshot reloaded';});
  function theme(name) { document.documentElement.dataset.theme=name; $('theme-marble').setAttribute('aria-pressed',String(name==='marble')); $('theme-basalt').setAttribute('aria-pressed',String(name==='basalt')); }
  $('theme-marble').addEventListener('click',() => theme('marble'));
  $('theme-basalt').addEventListener('click',() => theme('basalt'));
  $('copy-command').addEventListener('click', async () => {
    const input = $('telegram-command'), status = $('copy-status');
    try {
      if (!navigator.clipboard || !navigator.clipboard.writeText) throw new Error('clipboard unavailable');
      await navigator.clipboard.writeText(input.value);
      status.textContent = 'Command copied. Paste it in Telegram when you choose.';
    } catch (_error) {
      input.focus(); input.select();
      status.textContent = 'Command selected. Copy it manually, then paste it in Telegram.';
    }
  });
  $('export-record').addEventListener('click',() => exportJSON('clio-recorded-event.json',{label:'HISTORICAL_PAPER_NOT_LIVE',cutoff:CLIO_RECORDED.recorded_cutoff,record:currentRecord()}));
  $('export-case').addEventListener('click',() => exportJSON('clio-synthetic-case.json',{label:'SYNTHETIC_NOT_MARKET_DATA',case:cases[state.scenario]}));
  renderRecords();renderRecordDetail();renderCases();renderCaseDetail();
})();
