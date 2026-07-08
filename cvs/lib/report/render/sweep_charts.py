'''Shared CSS for static inference sweep bar-chart hover tooltips.'''

from __future__ import annotations


def chart_tooltip_css() -> str:
    return """
.chart-has-tip { position: relative; cursor: pointer; }
.chart-has-tip::before {
  content: attr(data-tip);
  position: absolute;
  bottom: calc(100% + 6px);
  left: 50%;
  transform: translateX(-50%);
  padding: 0.35rem 0.55rem;
  background: #2a3142;
  border: 1px solid var(--border);
  border-radius: 6px;
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--text);
  white-space: nowrap;
  opacity: 0;
  visibility: hidden;
  pointer-events: none;
  z-index: 20;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  transition: opacity 0.12s ease, visibility 0.12s ease;
}
.chart-has-tip:hover::before,
.chart-has-tip:focus-visible::before {
  opacity: 1;
  visibility: visible;
}
.chart-bar:hover { filter: brightness(1.12); }
"""
