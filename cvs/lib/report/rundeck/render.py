'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Compose static Run Deck HTML from payload and deck profile cards.
'''

from __future__ import annotations

import html

from cvs.lib.report.formatting import status_badge_html
from cvs.lib.report.rundeck.runtime.theme import report_css
from cvs.lib.report.rundeck.payload import default_inference_cards
from cvs.lib.report.rundeck.runtime.cards import close_gate_matrix_section, render_card


def _cards_from_payload(payload: dict) -> list[dict]:
    deck_profile = payload.get("deck_profile") or {}
    cards = deck_profile.get("cards")
    if cards:
        return cards
    return default_inference_cards()


def _build_nav(nav_items: list[tuple[str, str]], viewer_name: str | None) -> str:
    links = "".join(f"<a href='#{html.escape(section_id)}'>{html.escape(label)}</a>" for section_id, label in nav_items)
    viewer_nav = f"<a href='{html.escape(viewer_name)}'>Viewer</a>" if viewer_name else ""
    return f"<nav class='report-nav'>{links}{viewer_nav}</nav>"


def render_rundeck_html(payload: dict) -> str:
    """Profile-driven static Run Deck HTML."""
    report = payload.get("report") or {}
    cards = _cards_from_payload(payload)
    sections: list[str] = []
    nav_items: list[tuple[str, str]] = []
    pending_gate_section = ""
    pending_gate_nav: tuple[str, str] | None = None

    for card in cards:
        section_id, section_html, in_nav = render_card(payload, card)
        if not section_html:
            continue

        if card.get("type") == "gate_matrix":
            pending_gate_section = section_html
            if in_nav:
                pending_gate_nav = (section_id, card.get("title") or "Gates")
            continue

        if card.get("type") == "gate_heatmap":
            if pending_gate_section:
                combined = close_gate_matrix_section(pending_gate_section, section_html)
                sections.append(combined)
                if pending_gate_nav:
                    nav_items.append(pending_gate_nav)
                    nav_items.append(("heatmap", "Heatmap"))
                pending_gate_section = ""
                pending_gate_nav = None
            elif section_html:
                sections.append(f"<section class='panel' id='heatmap'><h2>Heatmap</h2>{section_html}</section>")
                nav_items.append(("heatmap", "Heatmap"))
            continue

        if pending_gate_section:
            sections.append(close_gate_matrix_section(pending_gate_section, "") + "</div></section>")
            if pending_gate_nav:
                nav_items.append(pending_gate_nav)
            pending_gate_section = ""
            pending_gate_nav = None

        sections.append(section_html)
        if in_nav and section_id:
            nav_items.append((section_id, card.get("title") or section_id.replace("-", " ").title()))

    if pending_gate_section:
        sections.append(close_gate_matrix_section(pending_gate_section, "") + "</div></section>")
        if pending_gate_nav:
            nav_items.append(pending_gate_nav)

    model_label = next((v for lbl, v, _ in payload.get("run_card_display", []) if lbl == "Model"), "run")
    overall = payload.get("overall_status", "na")
    viewer_name = (payload.get("summary") or {}).get("viewer_html")

    nav = _build_nav(nav_items, viewer_name)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(report.get('title', 'Run Deck'))} &mdash; {html.escape(str(model_label))}</title>
<style>{report_css()}</style></head><body><div class="wrap">
<div class="hero-head"><div><h1>{html.escape(report.get('title', 'Run Deck'))}</h1>
<p class="subtitle">{html.escape(report.get('subtitle', ''))}</p></div>{status_badge_html(overall)}</div>
{nav}
{''.join(sections)}
<footer class="page-foot">{html.escape(report.get('footer', ''))}</footer></div></body></html>"""
