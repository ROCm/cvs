var cvsPage = (function () {
  'use strict';

  var STATUS_ORDER = ['fail', 'pass', 'record', 'na', 'unknown'];
  var STATUS_LABEL = { fail: 'Failed', pass: 'Passed', record: 'Record only', na: 'N/A', unknown: 'Unavailable' };

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
    // CodeQL accepts a typeof test, not instanceof alone, as proof that report text never reaches appendChild.
    if (typeof child === 'object' && child instanceof Node) el.appendChild(child);
    else el.appendChild(document.createTextNode(String(child)));
  }
  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }
  function svg(tag, attrs) {
    var el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.keys(attrs).forEach(function (k) { el.setAttribute(k, attrs[k]); });
    for (var i = 2; i < arguments.length; i++) el.appendChild(arguments[i]);
    return el;
  }

  function mark(status, label) {
    return h('span', { class: 'status status-' + status },
      h('span', { class: 'status-glyph', 'aria-hidden': 'true' }), label || STATUS_LABEL[status]);
  }
  // Every link comes from report data; only paths inside the report are followed.
  function safeHref(href) { return typeof href === 'string' && href && href.indexOf(':') < 0 ? href : null; }
  function link(href, label, primary) {
    var safe = safeHref(href);
    if (!safe) return null;
    return h('a', { class: 'artifact-link' + (primary ? ' primary' : ''), href: safe, target: '_blank', rel: 'noopener' },
      label, h('span', { 'aria-hidden': 'true' }, '\u2197'));
  }

  function spark() {
    return svg('svg', { class: 'spark', viewBox: '0 0 120 120', fill: 'none', 'aria-hidden': 'true' },
      svg('g', { stroke: 'currentColor', 'stroke-width': '7' },
        svg('path', { d: 'M60 10V110M10 60H110' }),
        svg('path', { d: 'M32 32L88 88M88 32L32 88', opacity: '.45' })),
      svg('circle', { cx: '60', cy: '60', r: '16', fill: 'var(--f-canvas)' }),
      svg('circle', { cx: '60', cy: '60', r: '10', fill: 'var(--f-brand-red)' }));
  }
  function themeButton(onChange) {
    var button = h('button', { class: 'theme-toggle', type: 'button' });
    function paint() {
      var light = document.documentElement.dataset.theme === 'light';
      clear(button);
      add(button, [h('span', { 'aria-hidden': 'true' }, light ? '\u25D1' : '\u25D0'), light ? 'Dark' : 'Light']);
      button.setAttribute('aria-label', 'Switch to ' + (light ? 'dark' : 'light') + ' theme');
    }
    button.addEventListener('click', function () {
      var theme = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
      document.documentElement.dataset.theme = theme;
      try { localStorage.setItem('cvs-theme', theme); } catch (e) { /* storage is optional */ }
      paint();
      if (onChange) onChange(theme);
    });
    paint();
    return button;
  }
  function siteHeader(context, brandHref, onThemeChange) {
    var brand = [spark(), h('span', null, h('strong', null, 'AMD'), ' Cluster Validation Suite')];
    var safe = safeHref(brandHref);
    return h('header', { class: 'site-header' },
      safe ? h('a', { class: 'brand', href: safe }, brand) : h('span', { class: 'brand' }, brand),
      h('span', { class: 'header-divider', 'aria-hidden': 'true' }),
      h('span', { class: 'header-context' }, context),
      h('span', { class: 'header-right' }, themeButton(onThemeChange)));
  }

  return {
    STATUS_ORDER: STATUS_ORDER, STATUS_LABEL: STATUS_LABEL,
    h: h, add: add, clear: clear, mark: mark, safeHref: safeHref, link: link, siteHeader: siteHeader
  };
}());
