'''Chart.js 4.4.1 and chartjs-plugin-zoom 2.0.1, inlined into viewer pages so their charts draw offline.

Both are vendored unmodified from their npm packages, with their MIT licenses, in ``vendor/``.
'''

from pathlib import Path

VENDOR_DIR = Path(__file__).with_name("vendor")
CHART_JS = "chart.umd.js"
ZOOM_PLUGIN = "chartjs-plugin-zoom.min.js"


def chart_scripts(zoom=False):
    '''Inline ``<script>`` elements for Chart.js and, when ``zoom`` is set, its zoom plugin.'''
    names = (CHART_JS, ZOOM_PLUGIN) if zoom else (CHART_JS,)
    # A literal "</script" in the library would end its inline element early.
    sources = [(VENDOR_DIR / name).read_text(encoding="utf-8").replace("</script", "<\\/script") for name in names]
    return "\n".join("<script>%s</script>" % source for source in sources)
