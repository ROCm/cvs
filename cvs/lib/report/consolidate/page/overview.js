(function () {
  'use strict';

  var data = JSON.parse(document.getElementById('report-data').textContent);
  var summary = data.summary;
  var details = data.details;
  var suites = summary.suites;
  var areas = data.areas;
  var STATUS_ORDER = ['fail', 'pass', 'record', 'na', 'unknown'];
  var STATUS_LABEL = { fail: 'Failed', pass: 'Passed', record: 'Record only', na: 'N/A', unknown: 'Unavailable' };
  var VERDICT = {
    needs_attention: ['Needs attention', 'At least one suite failed.'],
    incomplete_data: ['Incomplete data', 'Nothing failed, but a report is missing, unreadable, or partial.'],
    evaluation_incomplete: ['Evaluation incomplete', 'Nothing failed, but at least one suite was never judged pass or fail.'],
    all_passed: ['All suites passed', 'Every included suite passed.']
  };
  var REC_MARK = { failed_suite: 'fail', unavailable: 'unknown', record_only: 'record', not_evaluated: 'na', all_passed: 'pass' };
  var REC_ROWS = 5;
  var state = { area: 'all', status: 'all', query: '', sort: 'area', desc: false, recsOpen: {} };
  var refs = {};
  var byId = {};
  var manifestById = {};
  suites.forEach(function (s) { byId[s.id] = s; });
  ((summary.manifest || {}).suites || []).forEach(function (m) { manifestById[m.id] = m; });

  function h(tag, attrs) {
    var el = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (key) {
        var value = attrs[key];
        if (value === null || value === undefined || value === false) return;
        if (key === 'class') el.className = value;
        else if (key.indexOf('on') === 0) el.addEventListener(key.slice(2), value);
        else el.setAttribute(key, value === true ? '' : value);
      });
    }
    for (var i = 2; i < arguments.length; i++) add(el, arguments[i]);
    return el;
  }
  function add(el, child) {
    if (child === null || child === undefined || child === false) return;
    if (Array.isArray(child)) { child.forEach(function (c) { add(el, c); }); return; }
    el.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }
  function svg(tag, attrs) {
    var el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.keys(attrs).forEach(function (k) { el.setAttribute(k, attrs[k]); });
    for (var i = 2; i < arguments.length; i++) el.appendChild(arguments[i]);
    return el;
  }

  function spark() {
    return svg('svg', { class: 'spark', viewBox: '0 0 120 120', fill: 'none', 'aria-hidden': 'true' },
      svg('g', { stroke: 'currentColor', 'stroke-width': '7' },
        svg('path', { d: 'M60 10V110M10 60H110' }),
        svg('path', { d: 'M32 32L88 88M88 32L32 88', opacity: '.45' })),
      svg('circle', { cx: '60', cy: '60', r: '16', fill: 'var(--f-canvas)' }),
      svg('circle', { cx: '60', cy: '60', r: '10', fill: 'var(--f-brand-red)' }));
  }
  function mark(status, label) {
    return h('span', { class: 'status status-' + status },
      h('span', { class: 'status-glyph', 'aria-hidden': 'true' }), label || STATUS_LABEL[status]);
  }
  function suiteCounts(list) {
    var c = { fail: 0, pass: 0, record: 0, na: 0, unknown: 0 };
    list.forEach(function (s) { c[s.status] += 1; });
    return c;
  }
  function bar(counts) {
    var total = STATUS_ORDER.reduce(function (n, k) { return n + (counts[k] || 0); }, 0);
    var label = STATUS_ORDER.filter(function (k) { return counts[k]; })
      .map(function (k) { return counts[k] + ' ' + STATUS_LABEL[k]; }).join(', ') || 'No outcomes';
    var el = h('div', { class: 'outcome-bar', role: 'img', 'aria-label': label });
    STATUS_ORDER.forEach(function (k) {
      if (!counts[k]) return;
      var seg = h('span', { class: 'segment segment-' + k });
      seg.style.width = (counts[k] / total * 100) + '%';
      el.appendChild(seg);
    });
    return el;
  }
  function link(href, label, primary) {
    if (!href || href.indexOf(':') >= 0) return null;
    return h('a', { class: 'artifact-link' + (primary ? ' primary' : ''), href: href, target: '_blank', rel: 'noopener' },
      label, h('span', { 'aria-hidden': 'true' }, '\u2197'));
  }
  function areaIndex(area) { var i = areas.indexOf(area); return i < 0 ? areas.length : i; }
  function areaCaption(counts, total) {
    if (!total) return 'No suites';
    if (counts.fail) return counts.fail + ' of ' + total + ' suites failed';
    if (counts.unknown) return 'Report data is incomplete';
    if (counts.record === total) return 'Recorded without pass/fail gates';
    if (counts.na === total) return 'No pass/fail verdict';
    return 'No failed suites';
  }

  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('cvs-theme', theme); } catch (e) { /* storage is optional */ }
    renderThemeButton();
  }
  function renderThemeButton() {
    var light = document.documentElement.dataset.theme === 'light';
    clear(refs.themeButton);
    add(refs.themeButton, [h('span', { 'aria-hidden': 'true' }, light ? '\u25D1' : '\u25D0'), light ? 'Dark' : 'Light']);
    refs.themeButton.setAttribute('aria-label', 'Switch to ' + (light ? 'dark' : 'light') + ' theme');
  }

  function header() {
    refs.themeButton = h('button', {
      class: 'theme-toggle', type: 'button',
      onclick: function () { setTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light'); }
    });
    renderThemeButton();
    return h('header', { class: 'site-header' },
      h('span', { class: 'brand' }, spark(), h('span', null, h('strong', null, 'AMD'), ' Cluster Validation Suite')),
      h('span', { class: 'header-divider', 'aria-hidden': 'true' }),
      h('span', { class: 'header-context' }, 'Consolidated qualification report'),
      h('span', { class: 'header-right' }, refs.themeButton));
  }

  function intro() {
    return h('div', { class: 'page-intro' },
      h('div', null, h('div', { class: 'eyebrow' }, 'CVS / Consolidated report'), h('h1', null, 'Cluster Validation')),
      h('div', { class: 'snapshot-time' }, h('span', { class: 'eyebrow' }, 'Runs from'), h('span', { class: 'mono' }, summary.generated)));
  }

  function summaryPanel() {
    var counts = suiteCounts(suites);
    var verdict = VERDICT[summary.verdict] || [summary.verdict, ''];
    var runs = suites.reduce(function (n, s) { return n + 1 + s.history.length; }, 0);
    var facts = [suites.length + ' suites'].concat(runs !== suites.length ? [runs + ' runs'] : [], ['collection-wide verdict']);
    refs.statusButtons = {};
    var buttons = STATUS_ORDER.filter(function (k) { return k !== 'unknown' || counts.unknown; }).map(function (k) {
      var button = h('button', {
        class: 'status-filter status-filter-' + k, type: 'button', 'aria-pressed': 'false',
        onclick: function () { state.status = state.status === k ? 'all' : k; update(); }
      }, h('strong', { class: 'mono' }, counts[k]), mark(k), h('span', { class: 'status-filter-check', 'aria-hidden': 'true' }));
      refs.statusButtons[k] = button;
      return button;
    });
    return h('section', { class: 'health-summary panel', 'aria-label': 'Collection assessment' },
      h('div', { class: 'health-verdict' },
        h('span', { class: 'eyebrow' }, 'Collection assessment'),
        h('h2', { title: verdict[1] }, h('span', { class: 'headline-glyph verdict-' + summary.verdict, 'aria-hidden': 'true' }), verdict[0]),
        h('p', null, facts.join(' \u00b7 '))),
      h('div', { class: 'health-outcomes' },
        h('p', null, 'Select a status to filter suites'),
        h('div', { class: 'status-filters', role: 'group', 'aria-label': 'Filter suites by status' }, buttons)));
  }

  function recMark(value) {
    return h('span', { class: 'status status-' + value + ' rec-mark', 'aria-hidden': 'true' }, h('span', { class: 'status-glyph' }));
  }
  function recDetail(r, suite) {
    var message = r.message || '', rest = message;
    if (message.indexOf(suite.name + ': ') === 0) rest = message.slice(suite.name.length + 2);
    else if (message.indexOf(suite.name + ' ') === 0) rest = message.slice(suite.name.length + 1);
    return rest.charAt(0).toUpperCase() + rest.slice(1);
  }
  function hostRec(x) {
    var names = x.failed_in, detail = h('div', { class: 'rec-detail' }, names.slice(0, 3).join(', '));
    if (names.length > 3) {
      var more = h('button', { class: 'rec-more', type: 'button' }, '+' + (names.length - 3) + ' more');
      more.addEventListener('click', function () { clear(detail); add(detail, names.join(', ')); });
      add(detail, [' ', more]);
    }
    return h('li', null,
      h('div', { class: 'rec-line' }, recMark('fail'), h('span', { class: 'rec-name mono' }, x.host),
        h('span', { class: 'rec-meta' }, 'failed in ' + names.length + ' of ' + x.suites.length + ' suites')),
      detail);
  }
  function suiteRec(r) {
    var suite = r.suite ? byId[r.suite] : null;
    return h('li', { title: r.message },
      h('div', { class: 'rec-line' }, recMark(REC_MARK[r.kind] || 'unknown'), suite
        ? h('button', { class: 'link-button rec-name', type: 'button', onclick: function () { openDrawer(suite); } }, suite.name)
        : h('span', { class: 'rec-name' }, r.message)),
      suite ? h('div', { class: 'rec-detail' }, recDetail(r, suite)) : null);
  }
  function recGroup(key, label, items, render) {
    var list = h('ul', { class: 'rec-list' });
    var toggle = items.length > REC_ROWS ? h('button', { class: 'text-button', type: 'button' }) : null;
    function fill() {
      clear(list);
      (state.recsOpen[key] ? items : items.slice(0, REC_ROWS)).forEach(function (item) { list.appendChild(render(item)); });
      if (toggle) { clear(toggle); add(toggle, state.recsOpen[key] ? 'Show fewer' : 'Show all ' + items.length); }
    }
    if (toggle) toggle.addEventListener('click', function () { state.recsOpen[key] = !state.recsOpen[key]; fill(); });
    fill();
    return h('section', { class: 'rec-group', 'aria-label': label },
      h('div', { class: 'rec-group-head' }, h('span', { class: 'eyebrow' }, label), h('span', { class: 'rec-count mono' }, items.length)),
      list, toggle);
  }
  function recommendationsPanel() {
    var recs = summary.recommendations || [];
    var hosts = ((summary.manifest || {}).hosts || []).filter(function (x) { return (x.failed_in || []).length >= 2; });
    hosts.sort(function (a, b) { return b.failed_in.length - a.failed_in.length || (a.host < b.host ? -1 : a.host > b.host ? 1 : 0); });
    var failed = recs.filter(function (r) { return r.kind === 'failed_suite'; });
    var review = recs.filter(function (r) { return ['unavailable', 'record_only', 'not_evaluated'].indexOf(r.kind) >= 0; });
    var passed = recs.filter(function (r) { return r.kind === 'all_passed'; })[0];
    return h('section', { class: 'panel recs', 'aria-labelledby': 'recs-title' },
      h('div', { class: 'recs-heading' },
        h('h2', { id: 'recs-title' }, 'Recommendations'),
        h('p', null, 'Facts from each Run Deck. No thresholds are added.')),
      passed ? h('p', { class: 'rec-all-passed' }, recMark('pass'), passed.message) : h('div', { class: 'recs-grid' },
        hosts.length ? recGroup('hosts', 'Hosts failing in 2+ suites', hosts, hostRec) : null,
        failed.length ? recGroup('failed', 'Failed suites', failed, suiteRec) : null,
        review.length ? recGroup('review', 'Needs review', review, suiteRec) : null));
  }

  function warningsNote() {
    var warnings = summary.warnings || [];
    if (!warnings.length) return null;
    return h('details', { class: 'collection-note' },
      h('summary', null, warnings.length + (warnings.length === 1 ? ' warning' : ' warnings') + ' while reading runs'),
      h('ul', null, warnings.map(function (w) { return h('li', null, h('span', { class: 'mono' }, w.code), ' ', w.message); })));
  }

  function workspace() {
    refs.areaButtons = {};
    var groups = [{ id: 'all', name: 'All suites' }].concat(
      areas.filter(function (a) { return suites.some(function (s) { return s.category === a; }); })
        .map(function (a) { return { id: a, name: a }; }));
    var nav = h('nav', { class: 'assessment-nav', 'aria-label': 'Areas' }, groups.map(function (g) {
      var list = g.id === 'all' ? suites : suites.filter(function (s) { return s.category === g.id; });
      var counts = suiteCounts(list);
      var button = h('button', {
        class: 'area-option', type: 'button', 'aria-pressed': 'false',
        onclick: function () { state.area = g.id; update(); }
      },
        h('span', { class: 'area-option-heading' }, h('strong', null, g.name), h('span', { class: 'mono' }, list.length)),
        h('span', { class: 'area-option-caption' }, areaCaption(counts, list.length)),
        bar(counts));
      refs.areaButtons[g.id] = button;
      return button;
    }));
    var legend = h('div', { class: 'assessment-legend' },
      STATUS_ORDER.filter(function (k) { return suites.some(function (s) { return s.status === k; }); }).map(function (k) { return mark(k); }));
    var areaPanel = h('aside', { class: 'assessment-panel panel', 'aria-labelledby': 'areas-title' },
      h('div', { class: 'assessment-heading' }, h('h2', { id: 'areas-title' }, 'Assessment by area'), h('p', null, 'Select an area to explore')),
      nav, legend);

    refs.title = h('h2', { id: 'suites-title' });
    refs.count = h('span', { class: 'inventory-count mono', role: 'status', 'aria-live': 'polite' });
    refs.search = h('input', {
      type: 'search', 'aria-label': 'Search suites', placeholder: 'Find a suite\u2026',
      oninput: function (e) { state.query = e.target.value; update(); }
    });
    refs.select = h('select', { 'aria-label': 'Filter by status', onchange: function (e) { state.status = e.target.value; update(); } },
      h('option', { value: 'all' }, 'All outcomes'),
      STATUS_ORDER.map(function (k) { return h('option', { value: k }, STATUS_LABEL[k]); }));
    refs.reset = h('button', {
      class: 'reset', type: 'button',
      onclick: function () { state.area = 'all'; state.status = 'all'; state.query = ''; refs.search.value = ''; update(); }
    }, 'Reset');
    refs.tbody = h('tbody');
    refs.headButtons = {};
    var columns = [['name', 'Suite'], ['status', 'Verdict'], ['area', 'Area'], ['fail', 'Failed'], ['pass', 'Passed'], [null, 'Result unit']];
    var thead = h('thead', null, h('tr', null, columns.map(function (c) {
      if (!c[0]) return h('th', { scope: 'col' }, c[1]);
      var button = h('button', {
        type: 'button',
        onclick: function () {
          if (state.sort === c[0]) state.desc = !state.desc;
          else { state.sort = c[0]; state.desc = false; }
          update();
        }
      }, c[1], h('span', { 'aria-hidden': 'true' }));
      refs.headButtons[c[0]] = button;
      return h('th', { scope: 'col' }, button);
    })));
    refs.empty = h('div', { class: 'empty-state', hidden: true }, 'No suites match these filters.');
    var suitesPanel = h('section', { class: 'comparison-panel panel', 'aria-labelledby': 'suites-title' },
      h('div', { class: 'comparison-controls' },
        h('div', { class: 'comparison-heading' }, refs.title, refs.count),
        h('div', { class: 'comparison-toolbar' },
          h('label', { class: 'search-field' }, h('span', { 'aria-hidden': 'true' }, '\u2315'), refs.search),
          refs.select, refs.reset)),
      h('div', { class: 'comparison-scroll', tabindex: '0', role: 'region', 'aria-label': 'Suites' },
        h('table', { class: 'comparison-table' }, thead, refs.tbody), refs.empty),
      h('div', { class: 'comparison-note' },
        'Counts use each suite\u2019s result unit. A suite\u2019s verdict is the overall status its Run Deck reports.'));
    return h('div', { class: 'health-workspace' }, areaPanel, suitesPanel);
  }

  function visibleSuites() {
    var query = state.query.trim().toLowerCase();
    var list = suites.filter(function (s) {
      return (state.area === 'all' || s.category === state.area) &&
        (state.status === 'all' || s.status === state.status) &&
        (!query || (s.name + ' ' + s.category + ' ' + s.id).toLowerCase().indexOf(query) >= 0);
    });
    var dir = state.desc ? -1 : 1;
    list.sort(function (a, b) {
      var c = 0;
      if (state.sort === 'name') c = a.name.localeCompare(b.name);
      else if (state.sort === 'status') c = STATUS_ORDER.indexOf(a.status) - STATUS_ORDER.indexOf(b.status);
      else if (state.sort === 'area') c = areaIndex(a.category) - areaIndex(b.category);
      else if (state.sort === 'fail') c = a.counts.fail - b.counts.fail;
      else if (state.sort === 'pass') c = a.counts.pass - b.counts.pass;
      return dir * c || a.name.localeCompare(b.name);
    });
    return list;
  }

  function row(s) {
    var runs = 1 + s.history.length;
    return h('tr', null,
      h('th', { scope: 'row' },
        h('button', { type: 'button', onclick: function () { openDrawer(s); }, 'aria-label': s.name + ': ' + STATUS_LABEL[s.status] + '. Open details.' },
          h('span', { class: 'suite-name' }, s.name, runs > 1 ? h('span', { class: 'runs-badge mono' }, runs + ' runs') : null),
          h('span', { class: 'row-arrow', 'aria-hidden': 'true' }, '\u2197'))),
      h('td', null, mark(s.status)),
      h('td', { class: 'table-category' }, s.category),
      h('td', { class: s.counts.fail ? 'table-fail' : null }, s.counts.fail),
      h('td', null, s.counts.pass),
      h('td', { class: 'table-unit' }, s.unit + (s.partial ? ' \u00b7 partial' : '')));
  }

  function update() {
    Object.keys(refs.statusButtons).forEach(function (k) {
      var on = state.status === k;
      refs.statusButtons[k].setAttribute('aria-pressed', on ? 'true' : 'false');
      refs.statusButtons[k].querySelector('.status-filter-check').textContent = on ? '\u2713' : '';
    });
    Object.keys(refs.areaButtons).forEach(function (k) {
      refs.areaButtons[k].setAttribute('aria-pressed', state.area === k ? 'true' : 'false');
    });
    refs.select.value = state.status;
    var list = visibleSuites();
    refs.title.textContent = state.area === 'all' ? 'All suites' : state.area;
    refs.count.textContent = list.length + ' of ' + suites.length + ' suites';
    refs.reset.disabled = state.area === 'all' && state.status === 'all' && !state.query;
    Object.keys(refs.headButtons).forEach(function (k) {
      var button = refs.headButtons[k];
      var active = state.sort === k;
      button.parentNode.setAttribute('aria-sort', active ? (state.desc ? 'descending' : 'ascending') : 'none');
      button.lastChild.textContent = active ? (state.desc ? '\u2193' : '\u2191') : '\u2195';
    });
    clear(refs.tbody);
    list.forEach(function (s) { refs.tbody.appendChild(row(s)); });
    refs.empty.hidden = list.length > 0;
  }

  function manifestPanel() {
    var manifest = summary.manifest || {};
    var clusters = Object.keys(manifest.clusters || {});
    var hosts = manifest.hosts || [];
    var hostTable = hosts.length ? h('div', { class: 'table-scroll' }, h('table', null,
      h('thead', null, h('tr', null, h('th', null, 'Host'), h('th', null, 'Reported by'), h('th', null, 'Failed in'))),
      h('tbody', null, hosts.map(function (x) {
        return h('tr', null, h('td', null, x.host), h('td', null, x.suites.join(', ')),
          h('td', { class: x.failed_in.length ? 'table-fail' : null }, x.failed_in.length ? x.failed_in.join(', ') : '\u2014'));
      })))) : null;
    var suiteColumns = ['Suite', 'Cluster', 'CVS version', 'Git ref', 'Image', 'Cluster file', 'Config file', 'Generated'];
    var suiteTable = h('div', { class: 'table-scroll' }, h('table', null,
      h('thead', null, h('tr', null, suiteColumns.map(function (c) { return h('th', null, c); }))),
      h('tbody', null, (manifest.suites || []).map(function (x) {
        var values = [x.name, x.cluster, x.cvs_version, x.git_ref, x.image, x.cluster_file, x.config_file, x.generated];
        return h('tr', null, values.map(function (v) { return h('td', null, v || '\u2014'); }));
      }))));
    return h('details', { class: 'panel manifest' },
      h('summary', null,
        h('span', { class: 'manifest-title' }, 'Cluster manifest'),
        h('span', { class: 'inventory-count' }, (clusters.length ? clusters.join(', ') : 'Cluster not specified') + ' \u00b7 ' + hosts.length + ' hosts in results')),
      h('div', { class: 'manifest-body' },
        h('p', { class: 'muted' }, 'Built only from what the Run Decks report. Hosts come from suites whose rows are hosts; training and inference decks add none.'),
        hostTable,
        h('h3', null, 'Per suite'),
        suiteTable));
  }

  function footer() {
    return h('footer', { class: 'report-footer' },
      h('span', null, 'Built by cvs consolidate-report. summary.json in this folder has the same results.'),
      h('span', { class: 'mono' }, 'CVS ' + data.cvs_version + ' \u00b7 schema_version ' + summary.schema_version + ' \u00b7 built ' + data.built));
  }

  function evidenceItem(e) {
    var metrics = (e.metrics || []).filter(function (m) { return m.status === 'fail'; });
    if (!metrics.length) metrics = (e.metrics || []).slice(0, 4);
    return h('div', { class: 'evidence' },
      h('div', { class: 'evidence-title' }, mark(e.status), h('strong', null, e.subject), h('span', { class: 'check-name' }, e.check)),
      e.message ? h('p', null, e.message) : null,
      metrics.length ? h('div', { class: 'table-scroll' }, h('table', null,
        h('thead', null, h('tr', null, h('th', null, 'Metric'), h('th', null, 'Actual'), h('th', null, 'Target'), h('th', null, 'Unit'))),
        h('tbody', null, metrics.map(function (m) {
          return h('tr', null, h('td', null, m.name), h('td', { class: m.status === 'fail' ? 'table-fail' : null }, m.actual), h('td', null, m.target), h('td', null, m.unit));
        })))) : null);
  }
  function evidenceSection(d) {
    var evidence = d.evidence || [];
    if (!evidence.length) return null;
    var list = h('div', { class: 'evidence-list' });
    evidence.slice(0, 8).forEach(function (e) { list.appendChild(evidenceItem(e)); });
    var more = null;
    if (evidence.length > 8) {
      more = h('button', { class: 'text-button', type: 'button' }, 'Show all ' + evidence.length);
      more.addEventListener('click', function () {
        evidence.slice(8).forEach(function (e) { list.appendChild(evidenceItem(e)); });
        more.remove();
      });
    }
    return h('section', { class: 'detail-section' }, h('h3', null, 'Failures to inspect', h('span', null, evidence.length)), list, more);
  }
  function matrixSection(d) {
    var m = d.matrix;
    if (!m || !m.nodes.length || !m.groups.length) return null;
    return h('section', { class: 'detail-section' },
      h('h3', null, 'Node / check outcomes', h('span', null, m.nodes.length + ' \u00d7 ' + m.groups.length)),
      h('div', { class: 'table-scroll' }, h('table', { class: 'grid-table' },
        h('thead', null, h('tr', null, h('th', null, 'Node'), m.groups.map(function (g) { return h('th', null, g.replace(/_/g, ' ')); }))),
        h('tbody', null, m.nodes.map(function (n) {
          return h('tr', null, h('td', null, n), m.groups.map(function (g) {
            var v = (m.grid[n] || {})[g] || 'unknown';
            return h('td', { class: 'cell-' + v }, mark(v));
          }));
        })))));
  }
  function tableSection(d) {
    var t = d.table;
    if (!t || !t.rows.length) return null;
    var rows = t.rows.slice(0, 25);
    return h('section', { class: 'detail-section' },
      h('h3', null, 'Recorded rows', h('span', null, t.rows.length)),
      h('div', { class: 'table-scroll' }, h('table', null,
        h('thead', null, h('tr', null, t.headers.map(function (x) { return h('th', null, x); }))),
        h('tbody', null, rows.map(function (r) { return h('tr', null, r.map(function (c) { return h('td', null, c); })); })))),
      t.rows.length > rows.length ? h('p', { class: 'muted' }, 'Showing ' + rows.length + ' of ' + t.rows.length + ' rows. The Run Deck has all of them.') : null);
  }
  function historySection(s) {
    if (!s.history.length) return null;
    return h('section', { class: 'detail-section' },
      h('h3', null, 'Earlier runs', h('span', null, s.history.length)),
      h('ul', { class: 'history-list' }, s.history.map(function (r) {
        return h('li', null, mark(r.status), h('span', { class: 'mono' }, r.generated), link(r.reportUrl, 'Run Deck'));
      })));
  }
  function contextSection(s, d) {
    var rows = (d.metadata || []).slice();
    function missing(label) { return !rows.some(function (r) { return String(r[0]).toLowerCase() === label.toLowerCase(); }); }
    var cvsVersion = (manifestById[s.id] || {}).cvs_version;
    if (missing('Generated')) rows.push(['Generated', s.generated]);
    if (cvsVersion && missing('CVS version')) rows.push(['CVS version', cvsVersion]);
    return h('section', { class: 'detail-section' },
      h('h3', null, 'Run context'),
      h('dl', { class: 'metadata' }, rows.map(function (r) { return h('div', null, h('dt', null, r[0]), h('dd', null, r[1])); })));
  }

  function openDrawer(s) {
    closeDrawer();
    var d = details[s.id] || { metadata: [], evidence: [] };
    var lastFocus = document.activeElement;
    var counts = s.counts;
    var caption = s.unit === 'recorded result rows' ? s.total + ' recorded result rows'
      : s.status === 'na' ? s.total + ' ' + s.unit + ' \u00b7 not evaluated'
        : counts.fail + ' of ' + s.total + ' ' + s.unit + ' failed';
    var closeButton = h('button', { class: 'icon-button', type: 'button', 'aria-label': 'Close details', onclick: closeDrawer }, '\u00d7');
    var body = h('div', { class: 'drawer-body' },
      h('div', { class: 'detail-status' }, mark(s.status), h('span', { class: 'mono' }, caption)),
      h('div', { class: 'breakdown' }, STATUS_ORDER.filter(function (k) { return counts[k]; }).map(function (k) {
        return h('span', null, h('strong', { class: 'mono' }, counts[k]), mark(k));
      })),
      h('div', { class: 'artifact-actions' },
        link(s.reportUrl, 'Open Run Deck', true), link(s.viewerUrl, 'Interactive viewer'),
        link(s.pytestUrl, 'Pytest report'), link(s.logUrl, 'Run log'), link(s.jsonUrl, 'Source JSON')),
      s.notes.length ? h('p', { class: 'detail-note' }, s.notes.join(' ')) : null,
      evidenceSection(d), matrixSection(d), tableSection(d), historySection(s), contextSection(s, d));
    var drawer = h('div', { class: 'drawer', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'drawer-title' },
      h('div', { class: 'drawer-header' },
        h('div', null, h('div', { class: 'eyebrow' }, s.category + ' / Run Deck'), h('h2', { id: 'drawer-title' }, s.name)),
        closeButton),
      body);
    var layer = h('div', { class: 'drawer-layer' }, h('div', { class: 'drawer-backdrop', onclick: closeDrawer }), drawer);
    refs.drawer = { layer: layer, lastFocus: lastFocus };
    document.body.appendChild(layer);
    document.body.classList.add('drawer-open');
    closeButton.focus();
  }
  function closeDrawer() {
    if (!refs.drawer) return;
    var last = refs.drawer.lastFocus;
    refs.drawer.layer.remove();
    refs.drawer = null;
    document.body.classList.remove('drawer-open');
    if (last && last.focus) last.focus();
  }
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeDrawer(); });

  var app = document.getElementById('app');
  add(app, [header(), h('main', { id: 'main', class: 'report', tabindex: '-1' },
    intro(), summaryPanel(), recommendationsPanel(), warningsNote(), workspace(), manifestPanel(), footer())]);
  update();
}());
