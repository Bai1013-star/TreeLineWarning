"""Small offline batch dashboard projection for the existing local UI shell."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape

from core.stage_a.models import Project


def render_batch_dashboard(
    project: Project,
    *,
    selected_span_ids: tuple[str, ...] | list[str] = (),
    statuses: Mapping[str, str] | None = None,
) -> str:
    """Render a self-contained Line/Tower/Span batch panel.

    The panel is intentionally schematic and uses no network assets.  The
    science/display boundary is explicit: this UI lists product records and
    status only; it never turns display samples into membership evidence.
    """

    selected = set(selected_span_ids)
    status_map = dict(statuses or {})
    rows: list[str] = []
    tower_by_id = {tower.tower_id: tower for tower in project.towers}
    for line in project.lines:
        rows.append(f"<h2>Line {escape(line.display_name)}</h2>")
        points = []
        for tower_id in line.ordered_tower_ids:
            tower = tower_by_id[tower_id]
            points.append(
                f"<span class='tower' data-id='{escape(tower.tower_id)}'>"
                f"{escape(tower.display_name)}</span>"
            )
        rows.append("<div class='topology'>" + " → ".join(points) + "</div>")
        for span in (item for item in project.spans if item.line_id == line.line_id):
            checked = " checked" if span.span_id in selected else ""
            status = escape(str(status_map.get(span.span_id, "WAITING")))
            rows.append(
                "<label class='span-row'>"
                f"<input type='checkbox' name='span_ids' value='{escape(span.span_id)}'{checked}>"
                f"<span>{escape(span.span_id)} {escape(span.start_tower_id)} → "
                f"{escape(span.end_tower_id)}</span><b>{status}</b></label>"
            )
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><title>TreeLineWarning Batch</title>
<style>body{{font-family:system-ui,sans-serif;background:#f7f8fa;color:#18212f;margin:2rem}}
.panel{{max-width:900px;background:white;border:1px solid #d7dce2;
border-radius:12px;padding:1.25rem}}
.topology{{padding:.7rem;background:#eef5ff;border-radius:8px;margin:.5rem 0 1rem}}
.tower{{display:inline-block;padding:.25rem .5rem;border:1px solid #8aa8c8;border-radius:999px}}
.span-row{{display:flex;gap:1rem;padding:.55rem 0;border-bottom:1px solid #edf0f3}}
.span-row b{{margin-left:auto;color:#6b7280}}button{{margin-top:1rem;padding:.6rem 1rem}}
canvas{{width:100%;height:140px;border:1px solid #d7dce2;
border-radius:8px;background:#fbfdff}}</style>
</head><body><main class='panel'><h1>{escape(project.name)} · Batch Analyze</h1>
<p>Offline local topology canvas — Viewer samples are display-only;
science keeps full-resolution original_index membership.</p>
<canvas id='line-canvas' aria-label='local Line Tower Span canvas'></canvas>
{''.join(rows)}<button id='batch-analyze' type='button'>Batch Analyze</button>
<button id='open-editor' type='button'>Open Span Viewer / Editor</button>
</main><script>
const canvas=document.getElementById('line-canvas'); const ctx=canvas.getContext('2d');
ctx.strokeStyle='#2563eb';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(40,70);ctx.lineTo(canvas.clientWidth-40,70);ctx.stroke();
document.getElementById('batch-analyze').addEventListener('click',()=>document.body.dataset.action='batch-analyze');
document.getElementById('open-editor').addEventListener('click',()=>document.body.dataset.action='open-editor');
</script></body></html>"""


__all__ = ["render_batch_dashboard"]
