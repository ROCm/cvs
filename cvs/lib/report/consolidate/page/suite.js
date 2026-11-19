(function () {
  'use strict';

  var page = JSON.parse(document.getElementById('suite-data').textContent);
  var d = page.deck || {};
  var suite = page.suite || {};
  var links = page.links || {};
  var STATUS_ORDER = cvsPage.STATUS_ORDER;
  var STATUS_LABEL = cvsPage.STATUS_LABEL;
  var h = cvsPage.h, add = cvsPage.add, clear = cvsPage.clear, mark = cvsPage.mark, link = cvsPage.link, safeHref = cvsPage.safeHref;
  var SERIES_VARS = ['--rv-series-1', '--rv-series-2', '--rv-series-3', '--rv-series-4', '--rv-series-5', '--rv-series-6'];

  function obj(v) { return v && typeof v === 'object' && !Array.isArray(v) ? v : {}; }
  function arr(v) { return Array.isArray(v) ? v : []; }
  function text(v) { return typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean' ? String(v) : '\u2014'; }
  function num(v) {
    if (typeof v === 'number' && isFinite(v)) return v;
    if (typeof v === 'string' && v.trim() !== '' && isFinite(Number(v))) return Number(v);
    return null;
  }
  function status(v) {
    if (v === 'pass' || v === 'fail' || v === 'record' || v === 'na') return v;
    return v === 'n/a' ? 'na' : 'unknown';
  }
  function verdict(list) {
    if (list.indexOf('fail') >= 0) return 'fail';
    if (!list.length || list.indexOf('unknown') >= 0) return 'unknown';
    if (list.indexOf('pass') >= 0) return 'pass';
    return list.indexOf('na') >= 0 ? 'na' : 'record';
  }
  function pretty(s) { return String(s).replace(/_/g, ' '); }
  function fmt(v) {
    if (typeof v !== 'number') return text(v);
    return Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumFractionDigits: 1 }) : String(Math.round(v * 1000) / 1000);
  }
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  function measurement(raw) {
    var m = obj(raw), spec = obj(m.spec);
    var target = num(spec.value !== undefined ? spec.value : m.threshold);
    var direction = text(spec.kind !== undefined ? spec.kind : m.direction !== undefined ? m.direction : '');
    var symbol = ['min', 'gte', 'higher', 'min_tok_s', 'min_ratio'].indexOf(direction) >= 0 ? '\u2265'
      : ['max', 'max_ms', 'lte', 'lower'].indexOf(direction) >= 0 ? '\u2264' : direction === 'eq' ? '=' : '';
    var actual = m.actual !== undefined ? m.actual : m.value;
    return {
      key: text(m.metric || m.name || m.label), name: text(m.label || m.name || m.metric),
      value: num(actual), actual: typeof actual === 'number' ? fmt(actual) : text(actual), unit: m.unit ? text(m.unit) : '',
      status: status(m.status), target: target,
      targetLabel: target === null ? 'No threshold supplied' : (symbol ? symbol + ' ' : 'Reported threshold: ') + fmt(target)
    };
  }
  function points(raw) {
    return arr(raw).map(function (p) { return Array.isArray(p) ? { x: text(p[0]), y: num(p[1]) } : { x: text(obj(p).x), y: num(obj(p).y) }; });
  }
  function plot(raw, name) {
    var p = obj(raw);
    return { name: name || text(p.name || p.label), unit: p.unit ? text(p.unit) : '', xLabel: p.x_label ? text(p.x_label) : 'Reported samples', points: points(p.points) };
  }
  function heatmap(raw) {
    var m = obj(raw);
    return {
      name: text(m.name), unit: m.unit ? text(m.unit) : '', rows: arr(m.rows).map(text), cols: arr(m.cols).map(text),
      values: arr(m.values).map(function (r) { return arr(r).map(num); }), threshold: num(m.threshold), direction: text(m.direction || '')
    };
  }
  function table(raw, title) {
    var t = obj(raw);
    return { title: title, headers: arr(t.headers).map(text), rows: arr(t.rows).map(function (r) { return arr(r).map(text); }) };
  }
  function pairs(raw) {
    if (Array.isArray(raw)) return raw.map(function (v, i) { return [String(i + 1), typeof v === 'object' ? JSON.stringify(v) : text(v)]; });
    return Object.keys(obj(raw)).map(function (k) { var v = raw[k]; return [k, v && typeof v === 'object' ? JSON.stringify(v) : text(v)]; });
  }

  function buildModel() {
    var datasets = obj(d.datasets);
    var model = {
      kind: 'unknown', results: [], plots: [], table: table(d.results_table, 'Source result rows'),
      lifecycle: pairs(d.lifecycle), notes: typeof d.run_card_notes === 'string' && d.run_card_notes ? [d.run_card_notes] : arr(d.run_card_notes).map(text),
      accuracy: pairs(d.accuracy)
    };
    if (datasets.status_matrix) {
      model.kind = 'matrix';
      var sm = obj(datasets.status_matrix), grid = obj(sm.grid);
      model.nodes = arr(sm.nodes).map(text);
      model.groups = arr(sm.groups).map(text);
      model.nodes.forEach(function (node, ni) {
        model.groups.forEach(function (group, gi) {
          var c = obj(obj(grid[node])[group]);
          model.results.push({
            id: 'n' + ni + 'g' + gi, label: pretty(group), subject: node, group: group, status: status(c.status),
            message: typeof c.items_summary === 'string' ? c.items_summary : '',
            dimensions: [['Node', node], ['Check group', pretty(group)]], metrics: arr(c.metrics).map(measurement), tiers: [],
            items: arr(c.items).map(function (raw) { var it = obj(raw); return { name: text(it.name), message: it.message ? text(it.message) : '', status: status(it.status) }; }),
            plots: arr(c.series).map(function (s) { return plot(s); }), heatmaps: arr(c.heatmaps).map(heatmap)
          });
        });
      });
      model.table = table(sm.results_table || d.results_table, 'Node / check source rows');
    } else if (datasets.sweep || datasets.training_sweep) {
      model.kind = 'sweep';
      var ds = obj(datasets.sweep || datasets.training_sweep);
      var cells = Array.isArray(ds.all_cells) ? ds.all_cells : Array.isArray(ds.cells) ? ds.cells : arr(d.cells);
      var dims = ['model', 'gpu', 'host', 'isl', 'osl', 'bs', 'sl', 'concurrency', 'mbs', 'gbs', 'precision', 'tp', 'pp', 'nodes', 'policy'];
      model.results = cells.map(function (raw, i) {
        var c = obj(raw), metrics = arr(c.metrics).map(measurement);
        var tiers = Object.keys(obj(c.tiers)).map(function (k) { return [k, status(c.tiers[k])]; });
        return {
          id: 'c' + i, label: text(c.label || c.cell_id || 'Configuration ' + (i + 1)), subject: c.host ? text(c.host) : '', group: c.policy ? text(c.policy) : '',
          status: verdict(tiers.length ? tiers.map(function (t) { return t[1]; }) : metrics.map(function (m) { return m.status; })),
          message: tiers.filter(function (t) { return t[1] === 'fail'; }).map(function (t) { return pretty(t[0]); }).join(', '),
          dimensions: dims.filter(function (k) { return c[k] !== undefined && c[k] !== ''; }).map(function (k) { return [k.toUpperCase(), text(c[k])]; }),
          metrics: metrics, tiers: tiers, items: [], heatmaps: arr(obj(c.metric_charts).heatmaps).map(heatmap),
          plots: arr(c.loss_curve).length ? [{ name: 'Training loss', unit: '', xLabel: 'Training step', points: points(c.loss_curve) }] : []
        };
      });
      arr(ds.chart_config).forEach(function (raw) {
        var cfg = obj(raw), series = arr(obj(ds.chart_series)[text(cfg.suffix)]);
        if (!series.length) return;
        model.plots.push({
          key: text(cfg.metric), title: text(cfg.title), kind: datasets.training_sweep ? 'bar' : 'line',
          series: series.map(function (s) { var p = plot(s); p.unit = cfg.unit ? text(cfg.unit) : p.unit; p.xLabel = datasets.training_sweep ? 'Configuration' : 'Concurrency'; return p; })
        });
      });
      model.summaries = arr(ds.sweep_summaries).map(function (raw) {
        var s = obj(raw);
        return { title: text(s.label || 'ISL=' + text(s.isl) + ' \u00b7 OSL=' + text(s.osl)), fields: Object.keys(s).filter(function (k) { return ['label', 'isl', 'osl'].indexOf(k) < 0; }).map(function (k) { return [pretty(k), text(s[k])]; }) };
      });
      model.table = table(ds.results_table || d.results_table, 'Source result rows');
    } else if (datasets.series) {
      model.kind = 'series';
      var se = obj(datasets.series), names = { bus_bw: 'Bus bandwidth', alg_bw: 'Algorithm bandwidth', time: 'Latency' };
      Object.keys(obj(se.charts)).forEach(function (key) {
        var bySeries = obj(se.charts[key]), series = [];
        Object.keys(bySeries).forEach(function (name) {
          arr(bySeries[name]).forEach(function (s) {
            var p = plot(s, name);
            p.unit = key === 'time' ? '\u00b5s' : key === 'bus_bw' || key === 'alg_bw' ? 'GB/s' : p.unit;
            p.xLabel = 'Message size';
            series.push(p);
          });
        });
        model.plots.push({ key: key, title: names[key] || pretty(key), kind: 'line', series: series });
      });
      model.table = table(se.results_table || d.results_table, 'Source result rows');
    }
    model.evidence = buildEvidence(model, datasets);
    return model;
  }

  function buildEvidence(model, datasets) {
    var ds = obj(datasets.sweep || datasets.training_sweep || datasets.status_matrix || datasets.series);
    var evidence = { charts: [], tables: [] };
    model.plots.forEach(function (p) {
      evidence.charts.push({ id: 'plot-' + p.key, title: p.title, group: model.kind === 'series' ? 'Collective communication' : 'Sweep metrics', kind: p.kind, series: p.series });
    });
    var containers = [];
    function addContainer(value) { var v = obj(value); if (Object.keys(v).length && containers.indexOf(v) < 0) containers.push(v); }
    addContainer(ds.metric_charts);
    arr(obj(d.deck_profile).cards).map(obj).forEach(function (card) {
      var parts = text(card.bind).split('.'), value = d;
      if (parts.some(function (p) { return ['__proto__', 'prototype', 'constructor'].indexOf(p) >= 0; })) return;
      parts.forEach(function (part) { value = obj(value)[part]; });
      if (card.type === 'metric_charts') addContainer(value);
      if (card.type === 'table' && value !== ds.results_table && value !== d.results_table && arr(obj(value).rows).length) evidence.tables.push(table(value, text(card.title)));
    });
    containers.forEach(function (container) {
      var grouped = {};
      arr(container.series).forEach(function (raw) {
        var s = obj(raw), title = text(s.name || s.label), group = s.group ? text(s.group) : 'Recorded series', key = group + '|' + title;
        if (!grouped[key]) {
          grouped[key] = { id: 'series-' + evidence.charts.length, title: title, group: group, kind: s.x_label === 'rank' ? 'bar' : 'line', series: [] };
          evidence.charts.push(grouped[key]);
        }
        grouped[key].series.push(plot(s, s.node ? text(s.node) : text(s.label || s.name)));
      });
      arr(container.metrics).forEach(function (raw) {
        var m = obj(raw), title = text(m.name), group = m.group ? text(m.group) : 'Measurements';
        evidence.charts.push({
          id: 'metric-' + evidence.charts.length, title: title, group: group, kind: 'metrics',
          measurements: arr(m.points).map(function (pv) {
            var p = obj(pv), result = model.results.filter(function (r) { return r.subject === text(p.node) && r.group === text(p.group || m.group); })[0];
            var metric = measurement({ name: title, value: p.value, unit: p.unit || m.unit, status: p.status, threshold: p.threshold !== undefined ? p.threshold : m.threshold, direction: p.direction || m.direction });
            return { label: text(p.node), metric: metric, resultId: result ? result.id : null };
          })
        });
      });
      arr(container.heatmaps).forEach(function (raw) {
        var m = heatmap(raw), group = [obj(raw).group, obj(raw).node].filter(function (v) { return typeof v === 'string' && v; }).join(' \u00b7 ') || 'Hardware maps';
        evidence.charts.push({ id: 'heat-' + evidence.charts.length, title: m.name, group: group, kind: 'heatmap', heatmap: m });
      });
    });
    return evidence;
  }

  var model = buildModel();
  var firstFail = model.results.filter(function (r) { return r.status === 'fail'; })[0] || model.results[0];
  var state = {
    view: model.kind === 'series' ? 'charts' : 'results', outcome: 'all', query: '', group: 'all',
    selected: firstFail ? firstFail.id : null, table: 0, chartGroup: 'all'
  };
  var liveCharts = [];
  var pendingCharts = [];

  function destroyCharts() { liveCharts.forEach(function (c) { c.destroy(); }); liveCharts = []; }
  function seriesColor(i) { return cssVar(SERIES_VARS[i % SERIES_VARS.length]) || '#1AA0F9'; }
  function chartCanvas(series, kind, unit) {
    var wrap = h('div', { class: 'rv-plot' }), canvas = h('canvas', { role: 'img', 'aria-label': series.map(function (s) { return s.name; }).join(', ') });
    wrap.appendChild(canvas);
    pendingCharts.push(function () {
      if (typeof Chart === 'undefined') {
        clear(wrap);
        add(wrap, h('p', { class: 'rv-muted' }, 'Chart.js could not be loaded; the Data table tab has the same values.'));
        return;
      }
      var labels = [];
      series.forEach(function (s) { s.points.forEach(function (p) { if (labels.indexOf(p.x) < 0) labels.push(p.x); }); });
      Chart.defaults.color = cssVar('--f-text-2');
      Chart.defaults.borderColor = cssVar('--f-border');
      Chart.defaults.font.family = cssVar('--f-mono');
      Chart.defaults.font.size = 11;
      liveCharts.push(new Chart(canvas, {
        type: kind === 'bar' ? 'bar' : 'line',
        data: {
          labels: labels,
          datasets: series.map(function (s, i) {
            var byX = {}, color = seriesColor(i);
            s.points.forEach(function (p) { byX[p.x] = p.y; });
            return { label: s.name, data: labels.map(function (x) { return byX[x] === undefined ? null : byX[x]; }), borderColor: color, backgroundColor: color, borderWidth: 2, pointRadius: 2, spanGaps: true };
          })
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          interaction: { mode: 'index', intersect: false },
          plugins: { legend: { display: series.length > 1, position: 'bottom', labels: { boxWidth: 10, boxHeight: 10 } } },
          scales: { y: { title: { display: !!unit, text: unit } }, x: { ticks: { maxRotation: 0, autoSkip: true } } }
        }
      }));
    });
    return wrap;
  }
  function measurementBars(chart) {
    var max = 1;
    chart.measurements.forEach(function (p) { max = Math.max(max, p.metric.value || 0, p.metric.target || 0); });
    max *= 1.08;
    return h('div', { class: 'rv-comparison-metrics' }, chart.measurements.map(function (p) {
      var m = p.metric, fill = h('span'), track = h('span', { class: 'rv-metric-track' }, fill);
      fill.style.width = Math.max(0, m.value || 0) / max * 100 + '%';
      fill.style.background = m.status === 'fail' ? 'var(--f-crit)' : 'var(--rv-series-1)';
      if (m.target !== null && m.target >= 0) { var tick = h('i'); tick.style.left = m.target / max * 100 + '%'; track.appendChild(tick); }
      var body = [h('span', { class: 'rv-comparison-value' }, h('strong', null, p.label), h('span', { class: 'mono' }, m.actual + (m.unit ? ' ' + m.unit : '')), mark(m.status)), track,
        h('small', null, m.targetLabel + (m.target !== null && m.unit ? ' ' + m.unit : ''))];
      var result = p.resultId ? byId(p.resultId) : null;
      return result ? h('button', { type: 'button', onclick: function () { choose(result); } }, body) : h('div', null, body);
    }));
  }
  function heatmapTable(m) {
    var values = [];
    m.values.forEach(function (r) { r.forEach(function (v) { if (v !== null) values.push(v); }); });
    var lo = Math.min.apply(null, values.concat([0])), hi = Math.max.apply(null, values.concat([1]));
    return h('div', { class: 'rv-table-scroll' }, h('table', { class: 'rv-heatmap' },
      h('thead', null, h('tr', null, h('th', null, ''), m.cols.map(function (c) { return h('th', null, c); }))),
      h('tbody', null, m.rows.map(function (row, ri) {
        return h('tr', null, h('th', { scope: 'row' }, row), m.cols.map(function (c, ci) {
          var v = (m.values[ri] || [])[ci], cell = h('td', null, v === null || v === undefined ? '\u2014' : fmt(v));
          if (v !== null && v !== undefined) {
            if (m.threshold !== null) {
              var ok = m.direction === 'lower' ? v <= m.threshold : v >= m.threshold;
              cell.className = ok ? 'rv-heat-pass' : 'rv-heat-fail';
            } else {
              cell.style.background = 'color-mix(in srgb, var(--rv-series-1) ' + Math.round(15 + 55 * (v - lo) / ((hi - lo) || 1)) + '%, transparent)';
            }
          }
          return cell;
        }));
      }))), m.unit ? h('p', { class: 'rv-muted' }, 'Values in ' + m.unit + (m.threshold !== null ? '; threshold ' + fmt(m.threshold) : '')) : null);
  }

  function byId(id) { return model.results.filter(function (r) { return r.id === id; })[0] || null; }
  function visibleResults() {
    var q = state.query.trim().toLowerCase();
    return model.results.filter(function (r) {
      return (state.outcome === 'all' || r.status === state.outcome) && (state.group === 'all' || r.group === state.group) &&
        [r.label, r.subject, r.group, r.message].concat(r.items.map(function (i) { return i.name + ' ' + i.message; })).join(' ').toLowerCase().indexOf(q) >= 0;
    });
  }
  function choose(result) { state.selected = result.id; state.view = 'results'; render(); }

  function metricCard(m) {
    var fill = h('span'), track = h('span', { class: 'rv-metric-track' }, fill), max = Math.max(m.value || 0, m.target || 0, 1) * 1.15;
    fill.style.width = Math.max(0, m.value || 0) / max * 100 + '%';
    fill.style.background = m.status === 'fail' ? 'var(--f-crit)' : 'var(--rv-series-1)';
    if (m.target !== null) { var tick = h('i'); tick.style.left = m.target / max * 100 + '%'; track.appendChild(tick); }
    return h('div', { class: 'rv-metric-card' },
      h('div', null, h('strong', null, m.name), mark(m.status)),
      h('p', null, h('b', null, m.actual), m.unit ? h('span', null, ' ' + m.unit) : null),
      m.value !== null ? track : null,
      h('div', { class: 'rv-metric-target' }, m.targetLabel + (m.target !== null && m.unit ? ' ' + m.unit : '')));
  }

  function inspector(result, outside) {
    if (!result) return h('aside', { class: 'rv-inspector' }, h('p', { class: 'rv-empty' }, 'Select an outcome to inspect its evidence.'));
    var important = result.metrics.filter(function (m) { return m.status === 'fail' || m.target !== null; });
    var other = result.metrics.filter(function (m) { return important.indexOf(m) < 0; });
    var message = result.message || (result.status === 'fail' ? 'The Run Deck marks this outcome as failed without a separate message.' : 'No additional failure message supplied.');
    return h('aside', { class: 'rv-inspector', 'aria-label': 'Selected outcome' },
      h('div', { class: 'rv-inspector-head' }, h('span', { class: 'eyebrow' }, 'Selected outcome'), mark(result.status), h('h2', null, result.label), result.subject ? h('p', null, result.subject) : null),
      h('div', { class: 'rv-inspector-content' },
        outside ? h('p', { class: 'rv-notice' }, 'This selected outcome is outside the current filter.') : null,
        h('div', { class: 'rv-evidence-message' }, h('span', { class: 'eyebrow' }, 'Reported evidence'), h('p', null, message),
          result.items.filter(function (it) { return it.status === 'fail' || result.items.length <= 6; }).slice(0, 12).map(function (it) {
            return h('div', { class: 'rv-item' }, mark(it.status), h('strong', null, it.name), it.message ? h('p', null, it.message) : null);
          })),
        important.length ? h('section', null, h('h3', null, 'Measured values against gates'), h('div', { class: 'rv-metric-grid' }, important.map(metricCard))) : null,
        result.tiers.length ? h('section', null, h('h3', null, 'Gate outcomes'), h('div', { class: 'rv-gates' }, result.tiers.map(function (t) { return h('span', null, h('span', null, pretty(t[0])), mark(t[1])); }))) : null,
        other.length ? h('details', { class: 'rv-disclosure' }, h('summary', null, 'Other measurements ', h('span', null, other.length)), h('div', { class: 'rv-metric-grid' }, other.map(metricCard))) : null,
        !result.metrics.length && !result.items.length ? h('p', { class: 'rv-muted' }, 'No individual metric values are supplied for this check.') : null,
        result.plots.map(function (p) { return h('section', null, h('h3', null, p.name), chartCanvas([p], 'line', p.unit)); }),
        result.heatmaps.map(function (m) { return h('section', null, h('h3', null, m.name), heatmapTable(m)); }),
        h('details', { class: 'rv-disclosure' }, h('summary', null, 'Configuration & context'),
          h('dl', { class: 'rv-context-list' }, result.dimensions.map(function (kv) { return h('div', null, h('dt', null, kv[0]), h('dd', null, kv[1])); })))));
  }

  function matrixPanel(visible) {
    var ids = {};
    visible.forEach(function (r) { ids[r.id] = true; });
    var nodes = model.nodes.filter(function (n) { return visible.some(function (r) { return r.subject === n; }); });
    return [h('div', { class: 'rv-table-scroll rv-matrix-scroll' }, h('table', { class: 'rv-matrix' },
      h('thead', null, h('tr', null, h('th', null, 'Node'), model.groups.map(function (g) { return h('th', null, pretty(g)); }))),
      h('tbody', null, nodes.map(function (node) {
        return h('tr', null, h('th', { scope: 'row' }, node), model.groups.map(function (g) {
          var r = model.results.filter(function (x) { return x.subject === node && x.group === g; })[0];
          return h('td', null, h('button', {
            type: 'button', class: (state.selected === r.id ? 'selected ' : '') + (ids[r.id] ? '' : 'rv-nonmatch'),
            'aria-pressed': state.selected === r.id ? 'true' : 'false', 'aria-label': node + ', ' + pretty(g) + ': ' + STATUS_LABEL[r.status],
            onclick: function () { choose(r); }
          }, mark(r.status)));
        }));
      })))), h('p', { class: 'rv-matrix-note' }, nodes.length + ' of ' + model.nodes.length + ' nodes shown. Dimmed cells provide context outside the current filter.')];
  }

  function configurationsPanel(visible) {
    return h('div', { class: 'rv-table-scroll rv-config-scroll' }, h('table', { class: 'rv-config-table' },
      h('thead', null, h('tr', null, h('th', null, 'Configuration'), h('th', null, 'Outcome'))),
      h('tbody', null, visible.map(function (r) {
        var m = r.metrics.filter(function (x) { return x.unit && x.value !== null; })[0];
        return h('tr', { class: r.id === state.selected ? 'selected' : null },
          h('th', { scope: 'row' }, h('button', { type: 'button', 'aria-pressed': r.id === state.selected ? 'true' : 'false', onclick: function () { choose(r); } },
            h('span', null, r.label), h('small', null, m ? m.name + ' ' + m.actual + ' ' + m.unit : r.subject, h('span', { 'aria-hidden': 'true' }, '\u2197')))),
          h('td', null, mark(r.status)));
      }))));
  }

  function resultsSection() {
    if (model.kind === 'series') return dataSection();
    var visible = visibleResults(), selected = byId(state.selected);
    var outside = !!selected && visible.indexOf(selected) < 0;
    var groups = [];
    model.results.forEach(function (r) { if (groups.indexOf(r.group) < 0) groups.push(r.group); });
    var search = h('input', { type: 'search', 'aria-label': 'Search outcomes', value: state.query, placeholder: model.kind === 'matrix' ? 'Find a node, check, or error\u2026' : 'Find a configuration or host\u2026' });
    search.addEventListener('change', function (e) { state.query = e.target.value; render(); });
    var groupSelect = h('select', { 'aria-label': 'Filter group', onchange: function (e) { state.group = e.target.value; render(); } },
      h('option', { value: 'all' }, model.kind === 'matrix' ? 'All check groups' : 'All policies'),
      groups.map(function (g) { return h('option', { value: g, selected: state.group === g }, pretty(g) || 'Unspecified'); }));
    var outcomeSelect = h('select', { 'aria-label': 'Filter outcome', onchange: function (e) { state.outcome = e.target.value; render(); } },
      h('option', { value: 'all' }, 'All outcomes'),
      STATUS_ORDER.map(function (k) { return h('option', { value: k, selected: state.outcome === k }, STATUS_LABEL[k]); }));
    return [
      h('div', { class: 'rv-section-title' },
        h('div', null, h('h2', null, model.kind === 'matrix' ? 'Node / check outcomes' : 'Configuration outcomes'),
          h('p', null, model.kind === 'matrix' ? 'Select a cell to inspect its checks and measurements.' : 'Select a configuration to inspect its metric gates and measurements.')),
        h('span', { class: 'rv-count-label' }, visible.length + ' / ' + model.results.length + ' ' + suite.unit + ' match')),
      h('div', { class: 'rv-controls' }, search, groupSelect, outcomeSelect,
        h('button', { type: 'button', class: 'rv-button', onclick: function () { state.outcome = 'all'; state.query = ''; state.group = 'all'; render(); } }, 'Reset')),
      h('div', { class: 'rv-results-layout' + (model.kind === 'sweep' ? ' rv-configuration-layout' : '') },
        h('section', { class: 'rv-results-panel', 'aria-label': 'Results' },
          !visible.length ? h('div', { class: 'rv-empty' }, h('h3', null, 'No outcomes match'), h('p', null, 'Try another search or reset the filters.'))
            : model.kind === 'matrix' ? matrixPanel(visible) : configurationsPanel(visible),
          h('div', { class: 'rv-panel-foot' }, model.kind === 'matrix' ? model.nodes.length + ' reported nodes \u00b7 coverage belongs to this suite only.' : 'Configuration outcomes and metric-gate outcomes are separate units.')),
        inspector(selected, outside))];
  }

  function outcomeBreakdown() {
    return h('section', { class: 'rv-chart-panel rv-outcome-breakdown' }, h('h3', null, 'Outcomes by check group'),
      h('p', { class: 'rv-muted' }, 'Absolute node/check group counts within this report.'),
      model.groups.map(function (g) {
        var rows = model.results.filter(function (r) { return r.group === g; });
        var counts = STATUS_ORDER.map(function (s) { return [s, rows.filter(function (r) { return r.status === s; }).length]; }).filter(function (c) { return c[1]; });
        var bar = h('div', { class: 'rv-stacked-bar', 'aria-hidden': 'true' });
        counts.forEach(function (c) { var seg = h('span', { class: 'rv-segment-' + c[0] }); seg.style.width = c[1] / rows.length * 100 + '%'; bar.appendChild(seg); });
        return h('div', { class: 'rv-outcome-row' }, h('strong', null, pretty(g)), bar,
          h('div', null, counts.map(function (c) { return h('span', null, h('b', null, c[1]), ' ', mark(c[0])); })));
      }));
  }

  function chartsSection() {
    var charts = model.evidence.charts, groups = [];
    charts.forEach(function (c) { if (groups.indexOf(c.group) < 0) groups.push(c.group); });
    var shown = charts.filter(function (c) { return state.chartGroup === 'all' || c.group === state.chartGroup; });
    return h('section', { class: 'rv-performance' },
      h('div', { class: 'rv-section-title' }, h('div', null, h('h2', null, 'Charts & performance'), h('p', null, 'Compare measurements from this Run Deck. Values, units, and thresholds are shown as supplied.')),
        h('span', { class: 'rv-count-label' }, shown.length + ' / ' + charts.length + ' charts')),
      (model.summaries || []).length ? h('div', { class: 'rv-sweep-summaries' }, model.summaries.map(function (s) {
        return h('article', null, h('span', { class: 'eyebrow' }, 'Reported sweep summary'), h('h3', null, s.title),
          h('dl', null, s.fields.map(function (kv) { return h('div', null, h('dt', null, kv[0]), h('dd', null, kv[1])); })));
      })) : null,
      model.kind === 'matrix' ? outcomeBreakdown() : null,
      groups.length > 1 ? h('div', { class: 'rv-controls' }, h('label', null, 'Group', h('select', { 'aria-label': 'Chart group', onchange: function (e) { state.chartGroup = e.target.value; render(); } },
        h('option', { value: 'all' }, 'All groups (' + groups.length + ')'), groups.map(function (g) { return h('option', { value: g, selected: state.chartGroup === g }, g); })))) : null,
      shown.length ? h('div', { class: 'rv-chart-gallery' }, shown.map(function (c) {
        var unit = c.series && c.series[0] ? c.series[0].unit : '';
        return h('article', { class: 'rv-chart-panel' }, h('div', { class: 'eyebrow' }, c.group), h('h3', null, c.title),
          c.kind === 'heatmap' ? heatmapTable(c.heatmap) : c.kind === 'metrics' ? measurementBars(c) : chartCanvas(c.series, c.kind, unit));
      })) : model.kind === 'matrix' ? null : h('p', { class: 'rv-empty' }, 'No chart measurements supplied for this suite.'));
  }

  function gatesSection() {
    var tiers = [];
    model.results.forEach(function (r) { r.tiers.forEach(function (t) { if (tiers.indexOf(t[0]) < 0) tiers.push(t[0]); }); });
    return h('section', null,
      h('div', { class: 'rv-section-title' }, h('div', null, h('h2', null, 'Configuration gate matrix'), h('p', null, 'Every configuration and its declared tier outcomes. Select a configuration to inspect its measured values.')),
        h('span', { class: 'rv-count-label' }, model.results.length + ' configurations \u00b7 ' + tiers.length + ' gate types')),
      h('div', { class: 'rv-source-table rv-table-scroll' }, h('table', null,
        h('thead', null, h('tr', null, h('th', null, 'Configuration'), tiers.map(function (t) { return h('th', null, pretty(t)); }))),
        h('tbody', null, model.results.map(function (r) {
          return h('tr', null, h('th', { scope: 'row' }, h('button', { type: 'button', class: 'rv-table-link', onclick: function () { choose(r); } }, r.label)),
            tiers.map(function (t) { var hit = r.tiers.filter(function (x) { return x[0] === t; })[0]; return h('td', null, mark(hit ? hit[1] : 'unknown')); }));
        })))));
  }

  function dataSection() {
    var tables = model.evidence.tables.concat([model.table]), t = tables[state.table] || tables[0];
    var q = state.query.trim().toLowerCase();
    var rows = t.rows.filter(function (r) { return r.join(' ').toLowerCase().indexOf(q) >= 0; });
    var search = h('input', { type: 'search', 'aria-label': 'Search result rows', value: state.query, placeholder: 'Search all source rows\u2026' });
    search.addEventListener('change', function (e) { state.query = e.target.value; render(); });
    return h('section', null,
      h('div', { class: 'rv-section-title' }, h('div', null, h('h2', null, t.title), h('p', null, 'Result rows keep their source units. They are not the suite-level count.')),
        h('span', { class: 'rv-count-label' }, rows.length + ' / ' + t.rows.length + ' rows')),
      h('div', { class: 'rv-controls' }, tables.length > 1 ? h('label', null, 'Table', h('select', { 'aria-label': 'Choose source table', onchange: function (e) { state.table = Number(e.target.value); render(); } },
        tables.map(function (x, i) { return h('option', { value: String(i), selected: state.table === i }, x.title + ' \u00b7 ' + x.rows.length + ' rows'); }))) : null, search),
      h('div', { class: 'rv-source-table rv-table-scroll', tabindex: '0', role: 'region', 'aria-label': 'Source result rows' }, h('table', null,
        h('thead', null, h('tr', null, t.headers.map(function (x) { return h('th', null, x); }))),
        h('tbody', null, rows.slice(0, 500).map(function (r) {
          return h('tr', null, r.map(function (v, j) { return h('td', null, /status|result/i.test(t.headers[j] || '') && STATUS_LABEL[v] ? mark(v) : v); }));
        }))), !rows.length ? h('p', { class: 'rv-empty' }, 'No result rows match.') : null,
        rows.length > 500 ? h('p', { class: 'rv-muted' }, 'Showing 500 of ' + rows.length + ' rows; the original Run Deck has all of them.') : null));
  }

  function contextSection() {
    var durations = model.lifecycle.map(function (kv) { return [kv[0], num(kv[1])]; }).filter(function (kv) { return kv[1] !== null && kv[1] >= 0; });
    var max = Math.max.apply(null, durations.map(function (kv) { return kv[1]; }).concat([1]));
    return h('section', { class: 'rv-context-grid' },
      h('div', { class: 'rv-chart-panel' }, h('h2', null, 'Run context'),
        h('dl', { class: 'rv-context-list' }, (page.metadata || []).map(function (kv) { return h('div', null, h('dt', null, kv[0]), h('dd', null, kv[1])); })),
        model.notes.map(function (n) { return h('p', { class: 'rv-notice' }, n); }),
        h('div', { class: 'rv-source-actions' }, link(links.viewer, 'Interactive viewer'), link(links.pytest, 'Pytest report'), link(links.log, 'Run log'))),
      h('div', { class: 'rv-chart-panel' }, h('h2', null, 'Reported lifecycle values'),
        durations.length ? h('div', { class: 'rv-duration-chart' }, durations.map(function (kv) {
          var bar = h('i'); bar.style.width = kv[1] / max * 100 + '%';
          return h('div', null, h('span', null, pretty(kv[0])), h('span', { class: 'rv-value-bar' }, bar), h('strong', null, fmt(kv[1]) + ' s'));
        })) : h('p', { class: 'rv-empty' }, 'No lifecycle measurements supplied.'),
        model.accuracy.length ? [h('h3', null, 'Accuracy results'), h('dl', { class: 'rv-context-list' }, model.accuracy.map(function (kv) { return h('div', null, h('dt', null, kv[0]), h('dd', null, kv[1])); }))] : null));
  }

  function suiteSwitcher() {
    var suites = page.suites || [], areas = [];
    suites.forEach(function (x) { if (areas.indexOf(x.category) < 0) areas.push(x.category); });
    return h('label', null, h('span', null, 'Suite'), h('select', {
      'aria-label': 'Choose suite', onchange: function (e) { var href = safeHref(e.target.value); if (href) window.location.href = href; }
    }, areas.map(function (area) {
      return h('optgroup', { label: area }, suites.filter(function (x) { return x.category === area; }).map(function (x) {
        return h('option', { value: x.url, selected: !!x.current }, x.name);
      }));
    })));
  }

  var counts = suite.counts || {};
  var tabs = [['results', model.kind === 'matrix' ? 'Node checks' : model.kind === 'series' ? 'Measurements' : 'Configurations']];
  if (model.evidence.charts.length || model.kind === 'matrix') tabs.push(['charts', 'Charts & performance (' + model.evidence.charts.length + ')']);
  if (model.results.some(function (r) { return r.tiers.length; })) tabs.push(['gates', 'Gate matrix']);
  tabs.push(['data', 'Data table'], ['context', 'Run context']);
  var notices = (suite.notes || []).filter(function (n) { return n !== 'Measurements are recorded without a pass/fail verdict.'; });

  function header() {
    return [
      h('div', { class: 'rv-title-row' },
        h('div', null, h('div', { class: 'eyebrow' }, (suite.category || 'Suite') + ' / Run Deck'), h('h1', null, suite.name || text(d.suite_id)), h('p', null, text(obj(d.report).subtitle || ''))),
        h('div', { class: 'rv-source-actions' }, link(links.deck, 'Original Run Deck'), link(links.json, 'Source JSON'), link(links.viewer, 'Interactive viewer'))),
      h('div', { class: 'rv-run-meta' },
        h('span', null, h('span', { class: 'rv-muted' }, 'Generated'), ' ', suite.generated || text(d.generated_at)),
        h('span', null, h('span', { class: 'rv-muted' }, 'Cluster'), ' ', suite.cluster || 'Not specified'),
        h('span', null, h('span', { class: 'rv-muted' }, 'CVS'), ' ', text(d.cvs_version))),
      h('section', { class: 'rv-assessment', 'aria-label': 'Run outcome' },
        h('div', { class: 'rv-verdict' }, h('span', { class: 'eyebrow' }, 'Declared suite verdict'), h('strong', null, mark(suite.status || status(d.overall_status))),
          h('p', null, model.kind === 'series' ? 'Recorded measurements; no pass/fail verdict per row.' : (counts.fail || 0) + ' of ' + (suite.total || 0) + ' ' + (suite.unit || 'results') + ' failed.')),
        h('div', { class: 'rv-outcome-controls', 'aria-label': 'Filter outcomes' },
          h('button', { type: 'button', 'aria-pressed': state.outcome === 'all' ? 'true' : 'false', onclick: function () { state.outcome = 'all'; state.view = 'results'; render(); } },
            h('b', null, suite.total || model.results.length), h('span', null, model.kind === 'series' ? 'All rows' : 'All results')),
          STATUS_ORDER.filter(function (k) { return counts[k]; }).map(function (k) {
            return h('button', { type: 'button', 'aria-pressed': state.outcome === k ? 'true' : 'false', onclick: function () { state.outcome = k; state.view = 'results'; render(); } },
              h('b', { class: k === 'fail' ? 'rv-red' : null }, counts[k]), mark(k));
          }))),
      notices.length ? h('div', { class: 'rv-notice' }, notices.join(' ')) : null,
      h('nav', { class: 'rv-tabs', 'aria-label': 'Report sections' }, tabs.map(function (t) {
        return h('button', { type: 'button', 'data-view': t[0], 'aria-current': state.view === t[0] ? 'page' : null, onclick: function () { state.view = t[0]; render(); } }, t[1]);
      }), h('span', null, (suite.total || 0) + ' ' + (suite.unit || '') + ' \u00b7 source coverage'))
    ];
  }

  var main = h('main', { id: 'main', class: 'rv-main', tabindex: '-1' });
  add(document.getElementById('app'), [
    cvsPage.siteHeader('Consolidated report', links.overview, function () { render(); }),
    h('div', { class: 'rv-breadcrumb' }, h('a', { href: safeHref(links.overview) || '#' }, '\u2190 Consolidated report'), suiteSwitcher()),
    main
  ]);

  function render() {
    destroyCharts();
    pendingCharts = [];
    clear(main);
    var body = state.view === 'charts' ? chartsSection() : state.view === 'gates' ? gatesSection() : state.view === 'data' ? dataSection()
      : state.view === 'context' ? contextSection() : resultsSection();
    add(main, [header(), body, h('footer', { class: 'rv-footer' },
      h('span', null, 'Built by cvs consolidate-report from this suite\u2019s Run Deck JSON. Values, units, and thresholds are shown as supplied; N/A and unavailable results are not passes.'),
      h('span', { class: 'mono' }, 'CVS ' + text(page.cvs_version)))]);
    pendingCharts.forEach(function (fn) { fn(); });
    pendingCharts = [];
  }

  render();
}());
