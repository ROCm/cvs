'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import re
import json
import math
from html import escape as html_escape

from cvs.lib.rocm_plib import *


def build_html_page_header(filename):
    """
    Create (or overwrite) an HTML file and write a standard header section.

    Args:
      filename (str): Path to the HTML file to create. The file is opened in write mode ('w'),
                      truncating any existing content.

    Behavior:
      - Prints a simple status message for visibility.
      - Opens the target file in write mode using a context manager.
      - Writes a multi-line HTML header containing:
          * DOCTYPE and basic <html>/<head> structure
          * UTF-8 meta charset
          * Title: 'CVS Cluster View'
          * Link to DataTables CSS
          * Simple inline CSS classes (.highlight-red, .label-danger)
          * Opening <body> tag to prepare for subsequent content
      - File is automatically closed when exiting the context manager.

    Notes:
      - The explicit fp.close() is unnecessary because the with-statement closes the file.
      - Consider specifying encoding='utf-8' when opening the file for portability.
      - If this function is called repeatedly, it will overwrite the existing file content each time.
      - For larger or templated HTML, consider using a template engine (e.g., Jinja2).
    """
    # Announce action for operator/logs

    log.info('Build HTML Page header')

    # Open the file in write mode; this truncates any existing file.
    # Use a context manager to ensure the file is properly closed even if an exception occurs.
    with open(filename, 'w') as fp:
        # Static HTML header content including basic document structure and CSS references
        html_lines = '''
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>CVS Cluster View</title>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
<!-- DataTables CSS -->
<style>

   /* ---- Design tokens ---- */
  :root{
    --sidebar-w: 220px;
    --radius: 14px;
    --gap: 6px;
    --border: 1px solid color-mix(in oklab, canvasText 8%, transparent);
    --accent: #60a5f1; /* tweak as you like */
  }
  @media (prefers-color-scheme: dark){
    :root{ color-scheme: dark; }
    aside{
    //background: #0b1b3a;
    linear-gradient(180deg, #0b1b3a 0%, #0a2540 45%, #0b2e59 100%);
    color: #e6f0ff;
    border-right: 1px solid rgba(255,255,255,.18);
    }
    .item:hover{ background: rgba(255,255,255,.04) }
    .item[aria-current="page"]{
       background: rgba(255,255,255,.16);
       outline-color: var(--accent);
    }
  }

  /* ---- Base ---- */
  *{ box-sizing:border-box }
  body{
    margin:0;
    font:12px/1.5 system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    background: canvas; color: canvasText;
  }
    a{ color: inherit; text-decoration:none }
  a:hover{ text-decoration:underline }
  .btn{
    display:inline-flex; align-items:center; gap:.5rem;
    border: var(--border); padding:.55rem .8rem; border-radius: 999px;
    background: color-mix(in oklab, canvas 85%, canvasText 5%);
  }

  /* ---- Layout ---- */
  .app{
    min-height: 100dvh;
    display: grid;
    grid-template-columns: var(--sidebar-w) 1fr;
    overflow-x:hidden;
  }
  header{
    position: sticky; top:0; z-index: 5;
    grid-column: 1 / -1;
    display:flex; align-items:center; gap: .75rem; padding: .75rem .9rem;
    border-bottom: var(--border);
    backdrop-filter: saturate(1.1) blur(4px);
    background: color-mix(in oklab, canvas 92%, transparent);
  }
  .toggle{
    display:none; /* shown on small screens */
    //display:inline-flex;
  }

  aside{
    position: sticky; top:0; height: calc(100dvh - 56px); /* 56px ~ header */
    align-self:start;
    border-right: var(--border);
    padding: 1rem;
    overflow:auto;
    background: #0f172a;    /* dark slate */
    color: #e5e7eb;         /* light text for contrast */
    //background: color-mix(in oklab, canvas 98%, canvasText 2%);
  }
  main{
    padding: 1.25rem 1.25rem 4rem;
    max-width: 1200px;
    min-width: 0;
  }

  /* ---- Sidebar nav ---- */
  nav[aria-label="Sidebar"] ul{ list-style:none; margin:0; padding:0 }
  .group{ margin-bottom: 1rem }
  .group > .title{
    font-size: .85rem; letter-spacing:.02em; text-transform:uppercase;
    opacity:.7; margin: .5rem 0 .25rem .75rem;
  }
  .item{
    display:flex; align-items:center; gap:.6rem;
    padding:.55rem .7rem; border-radius: var(--radius);
    margin:.15rem 0;
  }
  .item:hover{ background: color-mix(in oklab, canvasText 6%, transparent) }
  .item[aria-current="page"]{
    background: color-mix(in oklab, var(--accent) 14%, transparent);
    outline:1.5px solid color-mix(in oklab, var(--accent) 55%, transparent);
  }
  .icon{ inline-size:1.1em }

  /* ---- Mobile: off-canvas ---- */
  @media (max-width: 900px){
    .app{ grid-template-columns: 1fr }
    .toggle{ display:inline-flex }
    aside{
      position: fixed; inset: 56px auto 0 0; /* below header */
      width: min(88vw, var(--sidebar-w));
      transform: translateX(-105%);
      transition: transform .28s ease;
      box-shadow: 0 10px 30px color-mix(in oklab, black 15%, transparent);
      z-index: 20;
    }
   body.menu-open aside{ transform: translateX(0) }
    /* dim overlay */
    .overlay{
      position: fixed; inset: 56px 0 0 0;
      background: color-mix(in oklab, black 45%, transparent);
      opacity: 0; pointer-events: none; transition: opacity .2s;
      z-index: 15;
    }
    body.menu-open .overlay{ opacity:1; pointer-events:auto }
  }

  /* ---- Demo content cards ---- */
  .card{
    border: var(--border); border-radius: var(--radius);
    padding: 1rem; margin: .75rem 0;
    background: color-mix(in oklab, canvas 95%, transparent);
  }


   .highlight-red {
      color: red;
   }
   .label-danger {
      color: red;
   }
</style>
</head>
<body>

<div class="app">
  <header>
    <button class="btn toggle" id="menuBtn" aria-label="Open menu" aria-controls="sidebar" aria-expanded="true"> Menu</button>
    <center>    <strong>CVS - Cluster Validation Suite</strong> </center>
  </header>

  <aside id="sidebar" aria-hidden="false">
    <nav aria-label="Sidebar">



      <div class="group">
        <div class="title"><b>Cluster Summary</b></div>
        <ul>
          <li><a class="item" href="#clustsummary"><span class="icon">*</span>Summary</a></li>
          <li><a class="item" href="#lldpid"><span class="icon">*</span>LLDP Neighbors </a></li>
        </ul>
      </div>


      <div class="group">
        <div class="title"><b>GPU Info</b></div>
        <ul>
          <li><a class="item" href="#prodid"><span class="icon">*</span>GPU Firmware</a></li>
          <li><a class="item" href="#gpuuseid"><span class="icon">*</span>GPU Utilization</a></li>
          <li><a class="item" href="#memuseid"><span class="icon">*</span>GPU Memory Utilization</a></li>
          <li><a class="item" href="#pciexgmimetid"><span class="icon">*</span>GPU PCIe XGMI Metrics</a></li>
          <li><a class="item" href="#gpuerrorid"><span class="icon">*</span>GPU Error Metrics</a></li>
        </ul>
      </div>


      <div class="group">
        <div class="title"><b>NIC Info</b></div>
        <ul>
          <li><a class="item" href="#nicid"><span class="icon">*</span>NIC Info</a></li>
          <li><a class="item" href="#rdmastatsid"><span class="icon">*</span>RDMA Statistics</a></li>
          <li><a class="item" href="#ethtoolstatsid"><span class="icon">*</span>Ethtool Statistics</a></li>
        </ul>
      </div>


      <div class="group">
        <div class="title"><b>Historic Err Logs</b></div>
        <ul>
          <li><a class="item" href="#dmesgerrid"><span class="icon">*</span>Dmesg Logs</a></li>
          <li><a class="item" href="#journlctlerrid"><span class="icon">*</span>Journctl Logs</a></li>
          <li><a class="item" href="#gpupcieerrid"><span class="icon">*</span>GPU PCIe Error Logs</a></li>
          <li><a class="item" href="#gpupcielinkid"><span class="icon">*</span>GPU PCIe Link State Errors</a></li>
          <li><a class="item" href="#hostpcielinkid"><span class="icon">*</span>Host PCIe Link Errors</a></li>
          <li><a class="item" href="#niclinkflapid"><span class="icon">*</span>NIC Link Flap Errors</a></li>
        </ul>
      </div>


      <div class="group">
        <div class="title"> Snapshot Diffs</div>
        <ul>
          <li><a class="item" href="#snapdmesg"><span class="icon">*</span>Dmesg Diff</a></li>
          <li><a class="item" href="#snaperrlogsethid"><span class="icon">*</span>NIC Ethtool stats Diff logs</a></li>
          <li><a class="item" href="#snapethstatsid"><span class="icon">*</span>Ethtool Stats Diff</a></li>
          <li><a class="item" href="#snaperrlogsrdmaid"><span class="icon">*</span>NIC RDMA stats Diff logs</a></li>
          <li><a class="item" href="#snaprdmastatsid"><span class="icon">*</span>RDMA Stats Diff</a></li>
          <li><a class="item" href="#snaperrlogspcieid"><span class="icon">*</span>PCIe Error stats Diff logs</a></li>
          <li><a class="item" href="#snaperrlogsrasid"><span class="icon">*</span>GPU RAS Metrics Diff logs</a></li>
          <li><a class="item" href="#snaprasstatsid"><span class="icon">*</span>GPU RAS Stats Diff</a></li>
        </ul>
      </div>

      </nav>
    </aside>

    <div class="overlay" hidden></div>
    <main id="content" tabindex="-1">


         '''
        fp.write(html_lines)
        fp.close()


def build_html_page_footer(
    filename,
):
    """
    Append a standard HTML footer section with JS dependencies and DataTables initialization.

    Args:
      filename (str): Path to the HTML file to append to. The file is opened in append mode ('a').

    Behavior:
      - Prints a status message (currently says "header"; consider updating to "footer").
      - Appends the following to the file:
        * jQuery and DataTables JS includes (via CDN).
        * A document.ready block that initializes multiple DataTables by table ID.
      - Leaves the file open context automatically (with-statement handles closing).

    Notes:
      - The print message says 'Build HTML Page header' but this is a footer builder; consider fixing it.
      - This function assumes the page already includes matching table elements with the given IDs:
          #prod, #gpuuse, #memuse, #nic, #training, #ethtoolstats, #rdmastats, #pciexgmimetrics
        If an ID is missing in the DOM, DataTables initialization for that selector will fail.
      - Consider appending closing tags (</body></html>) after the script block if this is the final footer.
      - Consider using encoding='utf-8' for cross-platform consistency.
      - For large pages or to avoid CDN dependency at runtime, you may want to host/serve the JS locally.
    """

    log.info('Build HTML Page header')
    with open(filename, 'a') as fp:
        # Open the file in append mode; footer content is added at the end of the document.
        html_lines = '''
<!-- jQuery -->
<script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
<!-- DataTables JS -->
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>

<script>
  // Initialize DataTable
  $(document).ready(function() {
    $('#prodtable').DataTable({
     "pageLength": 100,
     "autoWidth": true
    });
    $('#gpuusetable').DataTable({
     "scrollX": true,
     "pageLength": 100,
     "autoWidth": true
    });
    $('#pciexgmimettable').DataTable({
     "scrollX": true,
     "pageLength": 100,
     "autoWidth": true
    });                  
    $('#memusetable').DataTable({
     "scrollX": true,    
     "pageLength": 100,  
     "autoWidth": true   
    });
    $('#gpuerrortable').DataTable({
     "scrollX": true,    
     "pageLength": 100,  
     "autoWidth": true   
    });                  
    $('#lldptable').DataTable({
     "scrollX": true,
     "pageLength": 100,
     "autoWidth": true
    });
    $('#nictable').DataTable({
     "scrollX": true,
     "pageLength": 100,
     "autoWidth": true
    });
    $('#training').DataTable({
     "scrollX": true,
     "pageLength": 100,
     "autoWidth": true
    });
    $('#ethtoolstatstable').DataTable({
     "scrollX": true,
     "pageLength": 100,
     "autoWidth": true
    });
    $('#rdmastatstable').DataTable({
     "scrollX": true,
     "pageLength": 100,
     "autoWidth": true
    });
    $('#pciexgmimetrics').DataTable({
     "scrollX": true,
     "pageLength": 100,
     "autoWidth": true
    });
    $('#histdmesgtable').DataTable({
     "scrollX": true,
     "pageLength": 100,
     "autoWidth": true
    });
    $('#snapdmesgtable').DataTable({
     //"scrollX": true,
     "pageLength": 100,
     "autoWidth": true
    });
    $('#snaperrlogsethtable').DataTable({
     //"scrollX": true,
     "pageLength": 100,
     "autoWidth": true
    });
    $('#snaperrlogspcietable').DataTable({
     //"scrollX": true,
     "pageLength": 100,
     "autoWidth": true
    });
    $('#snaperrlogsrastable').DataTable({
     //"scrollX": true,
     "pageLength": 100,
     "autoWidth": true
    });
    $('#snaperrlogsrdmatable').DataTable({
     //"scrollX": true,
     "pageLength": 100,
     "autoWidth": true
    });
    $('#snaprdmastatstable').DataTable({
     "scrollX": true,
     "pageLength": 100,
     "autoWidth": true
    });
    $('#snapethstatstable').DataTable({
     "scrollX": true,
     "pageLength": 100,
     "autoWidth": true
    });
    $('#snappcieerrtable').DataTable({
     "scrollX": true,
     "pageLength": 100,
     "autoWidth": true
    });
    $('#snaprastatstable').DataTable({
     "scrollX": true,
     "pageLength": 100,
     "autoWidth": true
    });
    $('#error').DataTable({
     "scrollX": true,
     "pageLength": 100,
     "autoWidth": true
    });


  });
</script>

</body>
</html>
         '''
        fp.write(html_lines)
        fp.close()


def normalize_bytes(n_bytes, si=False, precision=2):
    """
    Convert a byte count to a human string using B, KB, MB, or GB.

    Args:
        n_bytes: Size in bytes.
        si: If True, use decimal (kB=1000). If False (default), binary (KB=1024).
        precision: Decimal places for KB/MB/GB.

    Returns:
        A formatted string like "932 B", "1.23 KB", "45.6 MB", or "3.14 GB".
        Values >= 1 GB stay in GB (not TB+) since only KB/MB/GB are requested.
    """
    if not isinstance(n_bytes, (int, float)):
        raise TypeError("n_bytes must be an int or float")

    sign = "-" if n_bytes < 0 else ""
    n = abs(float(n_bytes))

    step = 1000.0 if si else 1024.0
    units = ["B", "kB" if si else "KB", "MB", "GB"]

    if n < step:
        return f"{sign}{int(n)} {units[0]}"
    elif n < step**2:
        val, unit = n / step, units[1]
    elif n < step**3:
        val, unit = n / (step**2), units[2]
    else:
        # cap at GB per request; if you want TB+, extend units and logic
        val, unit = n / (step**3), units[3]

    s = f"{val:.{precision}f}".rstrip("0").rstrip(".")
    return f"{sign}{s} {unit}"


def _stale_rccl_reference_hint(act_key, ref_data_dict):
    """Explain a missing reference key caused by per-data-type RCCL graph series, or return None."""
    from cvs.lib.rccl_lib import RcclVerifier

    base, dtype = RcclVerifier.split_series_label(act_key)
    if dtype is not None:
        if base in ref_data_dict:
            return (
                f"{act_key} not found in reference data, which has the plain series {base}: the reference "
                "predates per-data-type RCCL series and must be regenerated"
            )
        return None
    typed = [key for key in ref_data_dict if key != act_key and RcclVerifier.split_series_label(key)[0] == act_key]
    if typed:
        return (
            f"{act_key} not found in reference data, which has per-data-type series ({', '.join(typed)}): "
            "the reference and results use different RCCL series schemes; regenerate the reference with the "
            "same CVS version and data_types"
        )
    return None


def build_rccl_heatmap(filename, chart_name, title, act_data_json, ref_data_json):
    try:
        with open(act_data_json, 'r') as fp1:
            act_data_dict = json.load(fp1)
    except Exception as e:
        log.error(f'Error reading file {act_data_json} - {e}')

    try:
        with open(ref_data_json, 'r') as fp2:
            ref_data_dict_t = json.load(fp2)
            # Handle both wrapped (metadata/result) and unwrapped reference formats
            if 'result' in ref_data_dict_t:
                ref_data_dict = ref_data_dict_t['result']
            else:
                ref_data_dict = ref_data_dict_t
    except Exception as e:
        log.error(f'Error reading file {ref_data_json} - {e}')

    with open(filename, 'a') as fp:
        html_lines = (
            '''
         <h2 style="background-color: lightblue">'''
            + str(title)
            + '''</h2>


<!-- Styles -->
<style>
#'''
            + str(chart_name)
            + '''{
  width: 120%;
  height: 1400px;
}
</style>

<!-- Resources -->
<script src="https://cdn.amcharts.com/lib/5/index.js"></script>
<script src="https://cdn.amcharts.com/lib/5/xy.js"></script>
<script src="https://cdn.amcharts.com/lib/5/themes/Animated.js"></script>

<!-- Chart code -->
<script>
am5.ready(function() {

// Create root element
// https://www.amcharts.com/docs/v5/getting-started/#Root_element
var root = am5.Root.new("'''
            + str(chart_name)
            + '''");

// Set themes
// https://www.amcharts.com/docs/v5/concepts/themes/
root.setThemes([
  am5themes_Animated.new(root)
]);

// Create chart
// https://www.amcharts.com/docs/v5/charts/xy-chart/
var chart = root.container.children.push(
  am5xy.XYChart.new(root, {
    panX: false,
    panY: false,
    wheelX: "none", 
    wheelY: "none",
    paddingLeft: 0,
    paddingRight: 0,
    layout: root.verticalLayout
  })
);


// Create axes and their renderers
var yRenderer = am5xy.AxisRendererY.new(root, {
  visible: false,
  minGridDistance: 20,
  inversed: true,
  minorGridEnabled: true
});

yRenderer.grid.template.set("visible", false);

yRenderer.labels.template.setAll({
     fontSize: 10 // You can also use a number for pixel size
});

var yAxis = chart.yAxes.push(
  am5xy.CategoryAxis.new(root, {
    renderer: yRenderer,
    categoryField: "category"
  })
);

var xRenderer = am5xy.AxisRendererX.new(root, {
  visible: false,
  minGridDistance: 10,
  inversed: true,
  minorGridEnabled: true
});

xRenderer.grid.template.set("visible", false);

xRenderer.labels.template.setAll({
     fontSize: 10 // You can also use a number for pixel size
});


var xAxis = chart.xAxes.push(
  am5xy.CategoryAxis.new(root, {
    renderer: xRenderer,
    categoryField: "category"
  })
);


xAxis.set("numberFormatter", am5.NumberFormatter.new(root, {
 numberFormat: "#.0b",
   numericFields: ["valueX"]
  }));


// Create series
// https://www.amcharts.com/docs/v5/charts/xy-chart/#Adding_series
var series = chart.series.push(
  am5xy.ColumnSeries.new(root, {
    calculateAggregates: true,
    stroke: am5.color(0xffffff),
    clustered: false,
    xAxis: xAxis,
    yAxis: yAxis,
    categoryXField: "x",
    categoryYField: "y",
    valueField: "value"
  })
);

series.columns.template.setAll({
  tooltipText: "{categoryY}: {categoryX} - act = {value1} ref = {value2} GB/s",
  strokeOpacity: 1,
  strokeWidth: 2,
  cornerRadiusTL: 5,
  cornerRadiusTR: 5,
  cornerRadiusBL: 5,
  cornerRadiusBR: 5,
  width: am5.percent(100),
  height: am5.percent(100),
  templateField: "columnSettings"
});

var squareTemplate = am5.Template.new({});

// Add heat rule
// https://www.amcharts.com/docs/v5/concepts/settings/heat-rules/
series.set("heatRules", [{
  target: squareTemplate,
  min: 10,
  max: 10,
  dataField: "value",
  //key: "radius"
}]);
series.bullets.push(function () {
  return am5.Bullet.new(root, {
    sprite: am5.Rectangle.new(
      root,
      {
        fill: am5.color(0x000000),
        fillOpacity: 0.5,
        strokeOpacity: 0
      },
      squareTemplate
    )
  });
});



series.bullets.push(function () {
  return am5.Bullet.new(root, {
   sprite: am5.Label.new(root, {
      fill: am5.color(0xffffff),
      populateText: true,
      centerX: am5.p50,
      centerY: am5.p50,
      fontSize: 9,
      text: "{value}%"
    })
  });
});



var colors = {
  critical: am5.color(0xca0101),
  bad: am5.color(0xe17a2d),
  medium: am5.color(0xe1d92d),
  good: am5.color(0x5dbe24),
  verygood: am5.color(0x0b7d03)
};

// Set data
// https://www.amcharts.com/docs/v5/charts/xy-chart/#Setting_data
var data = [
         '''
        )
        fp.write(html_lines)
        log.info("%s", act_data_dict)
        for collective_name in act_data_dict.keys():
            log.info("%s", collective_name)
            # Skip if reference data doesn't have this collective
            if collective_name not in ref_data_dict:
                hint = _stale_rccl_reference_hint(collective_name, ref_data_dict)
                log.warning(f"Warning: {hint or f'{collective_name} not found in reference data'}, skipping")
                continue
            for msg_size in act_data_dict[collective_name].keys():
                norm_msg_size = normalize_bytes(int(msg_size))
                # calculate % diff between actual value and ref value
                act_bus_bw = act_data_dict[collective_name][msg_size]['bus_bw']
                # Skip if reference data doesn't have this message size
                if msg_size not in ref_data_dict[collective_name]:
                    log.warning(
                        f"Warning: msg_size {msg_size} not found in reference for {collective_name}, using actual as 100%"
                    )
                    pct_val = 100.0
                else:
                    ref_bus_bw = ref_data_dict[collective_name][msg_size]['bus_bw']
                    log.info("%s %s", act_bus_bw, ref_bus_bw)
                    if ref_bus_bw == 0 or ref_bus_bw == 0.0:
                        pct_val = 100
                    else:
                        pct_incr = ((act_bus_bw - ref_bus_bw) / ref_bus_bw) * 100
                        pct_val = round(float(pct_incr + 100), 2)

                if pct_val > 100:
                    fill_color = "colors.verygood"
                elif pct_val == 100:
                    fill_color = "colors.good"
                elif (pct_val < 100) and (pct_val > 95):
                    fill_color = "colors.good"
                elif (pct_val < 95) and (pct_val > 75):
                    fill_color = "colors.medium"
                elif pct_val < 75:
                    fill_color = "colors.critical"
                else:
                    fill_color = "colors.bad"

                collect_graph_name = collective_name.replace("_perf", "")
                html_lines = (
                    '''
  {
    y: "'''
                    + str(collect_graph_name)
                    + '''",
    x: "'''
                    + str(norm_msg_size)
                    + '''",
    columnSettings: {
      fill: '''
                    + fill_color
                    + '''
    },
    value: '''
                    + str(pct_val)
                    + ''',
    value1: '''
                    + str(act_bus_bw)
                    + ''',
    value2: '''
                    + str(ref_bus_bw)
                    + '''
  },
                 '''
                )
                fp.write(html_lines)

        html_lines = '''
];

series.data.setAll(data);

yAxis.data.setAll([
         '''
        fp.write(html_lines)
        for collective_name in act_data_dict.keys():
            collect_graph_name = collective_name.replace("_perf", "")
            html_lines = (
                '''
             { category: "'''
                + collect_graph_name
                + '''"},
             '''
            )
            fp.write(html_lines)
        html_lines = '''
]);

xAxis.data.setAll([
         '''
        fp.write(html_lines)

        first_collective = list(act_data_dict.keys())[0]
        msg_size_list = list(act_data_dict[first_collective])

        for msg_size in msg_size_list:
            norm_msg_size = normalize_bytes(int(msg_size))
            html_lines = (
                '''
             { category: "'''
                + str(norm_msg_size)
                + '''"},
             '''
            )
            fp.write(html_lines)
        html_lines = (
            '''
]);

// Make stuff animate on load
// https://www.amcharts.com/docs/v5/concepts/animations/#Initial_animation
chart.appear(1000, 100);

}); // end am5.ready()
</script>

<!-- HTML -->
<div id="'''
            + str(chart_name)
            + '''"></div>



</script>
         '''
        )
        fp.write(html_lines)


def build_rccl_amcharts_graph(filename, chart_name, rccl_dict):
    with open(filename, 'a') as fp:
        html_lines = (
            '''
         <h2 style="background-color: lightblue">RCCL Perf Results Bandwidth Graph</h2>
<style>

.highlight-red {
   color: red;
}
.label-danger {
   color: red;
}


#'''
            + str(chart_name)
            + '''{
  width: 120%;
  height: 550px;
  max-width: 120%;
}
</style>

<!-- Resources -->
<script src="https://cdn.amcharts.com/lib/5/index.js"></script>
<script src="https://cdn.amcharts.com/lib/5/xy.js"></script>
<script src="https://cdn.amcharts.com/lib/5/themes/Animated.js"></script>

<!-- Chart code -->
<script>
am5.ready(function() {

// Create root element
// https://www.amcharts.com/docs/v5/getting-started/#Root_element 
var root = am5.Root.new("'''
            + str(chart_name)
            + '''");

const myTheme = am5.Theme.new(root);

myTheme.rule("AxisLabel", ["minor"]).setAll({
  dy:1
});

myTheme.rule("Grid", ["x"]).setAll({
  strokeOpacity: 0.05
});

myTheme.rule("Grid", ["x", "minor"]).setAll({
  strokeOpacity: 0.05
});

// Create chart
// https://www.amcharts.com/docs/v5/charts/xy-chart/
var chart = root.container.children.push(am5xy.XYChart.new(root, {
  panX: true,
  panY: true,
  wheelX: "panX",
  wheelY: "zoomX",
  maxTooltipDistance: 0,
  pinchZoomX:true
}));


// Create axes
// https://www.amcharts.com/docs/v5/charts/xy-chart/axes/
var xAxis = chart.xAxes.push(am5xy.ValueAxis.new(root, {
  maxDeviation: 0.2,
  renderer: am5xy.AxisRendererX.new(root, {
    minorGridEnabled: true
  }),
  tooltip: am5.Tooltip.new(root, {})
}));

var yAxis = chart.yAxes.push(am5xy.ValueAxis.new(root, {
  renderer: am5xy.AxisRendererY.new(root, {}),
  numberFormat: "#.# 'GB'"
  })
);


// covert message size to MB

xAxis.set("numberFormatter", am5.NumberFormatter.new(root, {
 numberFormat: "#.0b",
  numericFields: ["valueX"]
}));

         '''
        )
        fp.write(html_lines)

        for series_name in rccl_dict.keys():
            html_lines = (
                '''
var series = chart.series.push(am5xy.LineSeries.new(root, {
    name: "'''
                + str(series_name)
                + '''",
    xAxis: xAxis,
    yAxis: yAxis,
    valueYField: "bus_bw",
    valueXField: "msg_size",
    legendValueText: "{valueY.formatNumber('#.00')}",
    tooltip: am5.Tooltip.new(root, {
      pointerOrientation: "horizontal",
      labelText: "{name}\\n: {valueY.formatNumber('#.00')} GB"
    })
   }));
             '''
            )
            fp.write(html_lines)
            data_list = []
            for msg_size in rccl_dict[series_name].keys():
                data_list.append({'msg_size': msg_size, 'bus_bw': rccl_dict[series_name][msg_size]['bus_bw']})
            html_lines = f'''
   data = {data_list}
   series.data.setAll(data);
   // Make stuff animate on load
   // https://www.amcharts.com/docs/v5/concepts/animations/
   series.appear();
             '''
            fp.write(html_lines)
        html_lines = '''
// Add cursor
// https://www.amcharts.com/docs/v5/charts/xy-chart/cursor/
var cursor = chart.set("cursor", am5xy.XYCursor.new(root, {
  behavior: "none"
}));
cursor.lineY.set("visible", false);


// Add scrollbar
// https://www.amcharts.com/docs/v5/charts/xy-chart/scrollbars/
chart.set("scrollbarX", am5.Scrollbar.new(root, {
  orientation: "horizontal"
}));

chart.set("scrollbarY", am5.Scrollbar.new(root, {
  orientation: "vertical"
}));


// Add legend
// https://www.amcharts.com/docs/v5/charts/xy-chart/legend-xy-series/
var legend = chart.rightAxesContainer.children.push(am5.Legend.new(root, {
  width: 270,
  paddingLeft: 8,
  height: am5.percent(100)
}));

legend.labels.template.setAll({
  fontSize: 10, // Change the font size of the labels
  maxWidth: 200, // Set a maximum width for the labels
  oversizedBehavior: "truncate" // Truncate oversized text
});


// When legend item container is hovered, dim all the series except the hovered one
legend.itemContainers.template.events.on("pointerover", function(e) {
  var itemContainer = e.target;

  // As series list is data of a legend, dataContext is series
  var series = itemContainer.dataItem.dataContext;

  chart.series.each(function(chartSeries) {
    if (chartSeries != series) {
      chartSeries.strokes.template.setAll({
        strokeOpacity: 0.15,
        stroke: am5.color(0x000000)
      });
    } else {
      chartSeries.strokes.template.setAll({
        strokeWidth: 3
      });
    }
  })
})
// When legend item container is unhovered, make all series as they are
legend.itemContainers.template.events.on("pointerout", function(e) {
  var itemContainer = e.target;
  var series = itemContainer.dataItem.dataContext;

  chart.series.each(function(chartSeries) {
    chartSeries.strokes.template.setAll({
      strokeOpacity: 1,
      strokeWidth: 1,
      stroke: chartSeries.get("fill")
    });
  });
})

legend.valueLabels.template.setAll({
  fontSize: 10
});

legend.itemContainers.template.set("width", am5.p100);
legend.valueLabels.template.setAll({
  width: am5.p100,
  textAlign: "right"
});

// It's is important to set legend data after all the events are set on template, otherwise events won't be copied
legend.data.setAll(chart.series.values);


// Make stuff animate on load
// https://www.amcharts.com/docs/v5/concepts/animations/
chart.appear(1000, 100);

}); // end am5.ready()
</script>
             '''
        fp.write(html_lines)


def add_html_begin(filename):
    with open(filename, 'w') as fp:
        html_lines = '''
         <html>
         <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
         '''
        fp.write(html_lines)


def add_html_end(filename):
    with open(filename, 'a') as fp:
        html_lines = '''
<!-- jQuery -->
<script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
<!-- DataTables JS -->
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>

<script>
  // Initialize DataTable
  $(document).ready(function() {
    $('#metatable').DataTable({
      "autoWidth": true
    });
    $('#rccltable').DataTable({
     "pageLength": 100,
     "autoWidth": true
    });

  });
</script>

         </html>
         '''
        fp.write(html_lines)


def add_json_data(filename, json_data):
    with open(filename, 'a') as fp:
        html_lines = (
            '''
         <h2 style="background-color: lightblue">RCCL Results JSON Format</h2>
         <pre id="json-display"></pre>
         <script>
         formatted_json = JSON.stringify( '''
            + str(json_data)
            + ''', null, 4 )
         document.getElementById('json-display').textContent = formatted_json
         </script>
         '''
        )
        fp.write(html_lines)


def build_rccl_result_default_table(filename, res_dict, bw_dip_threshold=10.0, time_dip_threshold=10.0):
    log.info('Build HTML RCCL Result default table')
    with open(filename, 'a') as fp:
        html_lines = '''
<h2 style="background-color: lightblue">RCCL Results Table</h2>
<table id="rccltable" class="display cell-border">
  <thead>
  <tr>
  <th>Collective</th>
  <th>Msg Size</th>
  <th>Algo BW GB/s</th>
  <th>Bus BW GB/s</th>
  <th>Latency us</th>
  </tr>
  </thead>'''
        fp.write(html_lines)
        for key_nam in res_dict.keys():
            collective = key_nam
            last_bw = 0.0
            last_time = 0
            for msg_size in res_dict[key_nam].keys():
                bus_bw = res_dict[key_nam][msg_size]['bus_bw']
                time = res_dict[key_nam][msg_size]['time']
                html_lines = f'''
     <tr>
     <td>{collective}</td>
     <td>{msg_size}</td>
     <td>{res_dict[key_nam][msg_size]['alg_bw']}</td>
                 '''
                fp.write(html_lines)

                # For dip_bw_check and dip_lat_check - mark red only if it is greater than some
                # threshold - by default 10.0 % from earlier message size.
                bw_change_pct = 0.0
                time_change_pct = 0.0
                # percent increase for BW
                if float(last_bw) > 0.0:
                    bw_change_pct = ((float(bus_bw) - float(last_bw)) / float(last_bw)) * 100.0

                # percent decrease for latency
                if float(last_time) > 0.0:
                    time_change_pct = ((float(time) - float(last_time)) / float(last_time)) * 100.0

                log.info("%s %s %s %s", bus_bw, last_bw, bw_change_pct, bw_dip_threshold)
                log.info("%s %s %s %s", time, last_time, time_change_pct, time_dip_threshold)

                if bw_change_pct < -(bw_dip_threshold):
                    html_lines = '''<td><span class="label label-danger">''' + str(bus_bw) + '''</td>\n'''
                else:
                    html_lines = '''<td>''' + str(bus_bw) + '''</td>\n'''
                fp.write(html_lines)

                # latency dip check add later
                html_lines = '''<td>''' + str(time) + '''</td>\n'''
                fp.write(html_lines)
                last_bw = bus_bw
                last_time = time

        html_lines = '''
         </table>
         <br><br>
         '''
        fp.write(html_lines)


def build_rccl_result_table(filename, res_dict):
    log.info('Build HTML RCCL Result table')
    with open(filename, 'a') as fp:
        html_lines = '''
<h2 style="background-color: lightblue">RCCL Results Table</h2>
<table id="rccltable" class="display cell-border">
  <thead>
  <tr>
  <th>Collective</th>
  <th>Algo</th>
  <th>Protocol</th>
  <th>QP_count</th>
  <th>PXN_DISABLE</th>
  <th>Msg Size</th>
  <th>Algo BW GB/s</th>
  <th>Bus BW GB/s</th>
  <th>Latency us</th>
  </tr>
  </thead>'''
        fp.write(html_lines)
        for key_nam in res_dict.keys():
            # Parse the new key format: collective-NCCL_ALGO=value NCCL_PROTO=value NCCL_IB_QPS_PER_CONNECTION=value NCCL_PXN_DISABLE=value ...
            parts = key_nam.split("-", 1)  # Split only on first hyphen
            collective = parts[0]

            if len(parts) == 2:
                # Parse the params string to extract individual values
                params_str = parts[1]
                params_dict = {}

                # Split by spaces and parse key=value pairs
                for param in params_str.split():
                    if '=' in param:
                        key, value = param.split('=', 1)
                        params_dict[key] = value

                # Extract the specific values we need for the table
                algo = params_dict.get('NCCL_ALGO', 'Unknown')
                protocol = params_dict.get('NCCL_PROTO', 'Unknown')
                qp_count = params_dict.get('NCCL_IB_QPS_PER_CONNECTION', 'Unknown')
                pxn_disable = params_dict.get('NCCL_PXN_DISABLE', 'Unknown')
            else:
                # This should not happen with current key construction (collective-params_str)
                # but provides safety against malformed keys
                log.error(f"Invalid key format - missing hyphen separator: {key_nam}")
                algo = protocol = qp_count = pxn_disable = 'Unknown'
            last_bw = 0.0
            last_time = 0
            for msg_size in res_dict[key_nam].keys():
                bus_bw = res_dict[key_nam][msg_size]['bus_bw']
                time = res_dict[key_nam][msg_size]['time']
                html_lines = f'''
     <tr>
     <td>{collective}</td>
     <td>{algo}</td>
     <td>{protocol}</td>
     <td>{qp_count}</td>
     <td>{pxn_disable}</td>
     <td>{msg_size}</td>
     <td>{res_dict[key_nam][msg_size]['alg_bw']}</td>
                 '''
                fp.write(html_lines)
                if float(bus_bw) < float(last_bw):
                    html_lines = '''<td><span class="label label-danger">''' + str(bus_bw) + '''</td>\n'''
                else:
                    html_lines = '''<td>''' + str(bus_bw) + '''</td>\n'''
                fp.write(html_lines)
                if float(time) < float(last_time):
                    html_lines = '''<td><span class="label label-danger">''' + str(time) + '''</td>\n'''
                else:
                    html_lines = '''<td>''' + str(time) + '''</td>\n'''
                fp.write(html_lines)
                last_bw = bus_bw
                last_time = time
        html_lines = '''
         </table>
         <br><br>
         '''
        fp.write(html_lines)


def build_rccl_heatmap_metadata_table(filename, act_data_json, ref_data_json):
    try:
        with open(act_data_json, 'r') as fp1:
            act_data_dict = json.load(fp1)
    except Exception as e:
        log.error(f'Error reading file {act_data_json} - {e}')

    try:
        with open(ref_data_json, 'r') as fp2:
            ref_data_dict = json.load(fp2)
    except Exception as e:
        log.error(f'Error reading file {ref_data_json} - {e}')

    log.info('Build HTML RCCL heatmap Metadata table')
    with open(filename, 'a') as fp:
        html_lines = '''
         <br><br>
<table id="metatable" class="display cell-border">
  <thead>
  <tr>
  <th>Item</th>
  <th>GPU Model</th>
  <th>NIC Model</th>
  <th>Collection Date</th>
  <th>BKC Version</th>
  <th>ROCM/CUDA Version</th>
  <th>RCCL/NCCL Commit/Date</th>
  </tr>
  </thead>'''
        fp.write(html_lines)
        html_lines = '<tr>'
        fp.write(html_lines)
        html_lines = '<td>Current</td>'
        fp.write(html_lines)
        for key_nam in act_data_dict.keys():
            if 'metadata' in key_nam:
                log.info("%s", list(act_data_dict['metadata'].keys()))
                if 'gpu_model' in act_data_dict[key_nam].keys():
                    html_lines = f"<td>{act_data_dict['metadata']['gpu_model']}</td>"
                    fp.write(html_lines)
                else:
                    fp.write('<td>-</td>')

                if 'nic_model' in act_data_dict[key_nam].keys():
                    html_lines = f"<td>{act_data_dict['metadata']['nic_model']}</td>"
                    fp.write(html_lines)
                else:
                    fp.write('<td>-</td>')

                if 'date' in act_data_dict[key_nam].keys():
                    html_lines = f"<td>{act_data_dict['metadata']['date']}</td>"
                    fp.write(html_lines)
                else:
                    fp.write('<td>-</td>')

                if 'bkc_version' in act_data_dict[key_nam].keys():
                    html_lines = f"<td>{act_data_dict['metadata']['bkc_version']}</td>"
                    fp.write(html_lines)
                else:
                    fp.write('<td>-</td>')

                if 'rocm_version' in act_data_dict[key_nam].keys():
                    html_lines = f"<td>{act_data_dict['metadata']['rocm_version']}</td>"
                    fp.write(html_lines)
                elif 'cuda_version' in act_data_dict[key_nam].keys():
                    html_lines = f"<td>{act_data_dict['metadata']['cuda_version']}</td>"
                    fp.write(html_lines)
                else:
                    fp.write('<td>-</td>')

                if 'rccl_commit' in act_data_dict[key_nam].keys():
                    html_lines = f"<td>{act_data_dict['metadata']['rccl_commit']}</td>"
                    fp.write(html_lines)
                else:
                    fp.write('<td>-</td>')
        html_lines = '</tr>'
        fp.write(html_lines)

        for key_nam in ref_data_dict.keys():
            if 'metadata' in key_nam:
                html_lines = '<tr>'
                fp.write(html_lines)
                html_lines = '<td>Golden</td>'
                fp.write(html_lines)
                log.info("%s", list(ref_data_dict[key_nam].keys()))
                if 'gpu_model' in ref_data_dict[key_nam].keys():
                    html_lines = f"<td>{ref_data_dict['metadata']['gpu_model']}</td>"
                    fp.write(html_lines)
                else:
                    fp.write('<td>-</td>')
                if 'nic_model' in ref_data_dict[key_nam].keys():
                    html_lines = f"<td>{ref_data_dict['metadata']['nic_model']}</td>"
                    fp.write(html_lines)
                else:
                    fp.write('<td>-</td>')
                if 'date' in ref_data_dict[key_nam].keys():
                    html_lines = f"<td>{ref_data_dict['metadata']['nic_model']}</td>"
                    fp.write(html_lines)
                else:
                    fp.write('<td>-</td>')
                if 'bkc_version' in ref_data_dict[key_nam].keys():
                    html_lines = f"<td>{ref_data_dict['metadata']['bkc_version']}</td>"
                    fp.write(html_lines)
                else:
                    fp.write('<td>-</td>')
                if 'rocm_version' in ref_data_dict[key_nam].keys():
                    html_lines = f"<td>{ref_data_dict['metadata']['rocm_version']}</td>"
                    fp.write(html_lines)
                elif 'cuda_version' in ref_data_dict[key_nam].keys():
                    html_lines = f"<td>{ref_data_dict['metadata']['cuda_version']}</td>"
                    fp.write(html_lines)
                else:
                    fp.write('<td>-</td>')
                if 'rccl_commit' in ref_data_dict[key_nam].keys():
                    html_lines = f"<td>{ref_data_dict['metadata']['rccl_commit']}</td>"
                    fp.write(html_lines)
                else:
                    fp.write('<td>-</td>')
                html_lines = '</tr>'
                fp.write(html_lines)

        html_lines = '''
         </table>'''
        fp.write(html_lines)


def build_rccl_heatmap_table(filename, title, act_data_json, ref_data_json):
    try:
        with open(act_data_json, 'r') as fp1:
            act_data_dict = json.load(fp1)
    except Exception as e:
        log.error(f'Error reading file {act_data_json} - {e}')

    try:
        with open(ref_data_json, 'r') as fp2:
            ref_data_dict_t = json.load(fp2)
    except Exception as e:
        log.error(f'Error reading file {ref_data_json} - {e}')

    # Handle both wrapped (metadata/result) and unwrapped reference formats
    if 'result' in ref_data_dict_t.keys():
        ref_data_dict = ref_data_dict_t['result']
    else:
        # Assume unwrapped format - use directly
        ref_data_dict = ref_data_dict_t

    log.info('Build HTML RCCL heatmap table')
    missing_ref_keys = []
    missing_ref_msg_sizes = 0
    with open(filename, 'a') as fp:
        html_lines = (
            '''
<h2 style="background-color: lightblue">'''
            + str(title)
            + '''</h2>
<table id="rccltable" class="display cell-border">
  <thead>
  <tr>
  <th>Collective</th>
  <th>DataType</th>
  <th>Number of GPUs</th>
  <th>Msg Size</th>
  <th>Result Algo BW GB/s</th>
  <th>Golden Algo BW GB/s</th>
  <th>Result Bus BW GB/s</th>
  <th>Golden Bus BW GB/s</th>
  <th>Result Latency us</th>
  <th>Golden Latency us</th>
  </tr>
  </thead>'''
        )
        fp.write(html_lines)
        for key_nam in act_data_dict.keys():
            if key_nam not in ref_data_dict:
                hint = _stale_rccl_reference_hint(key_nam, ref_data_dict)
                if hint:
                    log.warning(f"Warning: {hint}, skipped from table")
                    continue
            # Handle both 3-part (collective-datatype-gpucount) and 4-part (collective-datatype-gpucount-chconfig) formats
            parts = key_nam.split("-")
            if len(parts) >= 3:
                collective = parts[0]
                data_type = parts[1]
                gpu_count = parts[2]
            else:
                log.warning(f"Warning: Invalid key format {key_nam}, skipping")
                continue
            # Skip if reference data doesn't have this test configuration.
            # Some golden reference files use keys without channel suffix (e.g. "all_reduce_perf-float-8")
            # while the active run may include "-chdefault" (e.g. "all_reduce_perf-float-8-chdefault").
            ref_key = key_nam
            if ref_key not in ref_data_dict:
                # Try fallback by stripping channel suffix (anything after the 3rd dash-separated field)
                # e.g. "collective-dtype-gpucount-chdefault" -> "collective-dtype-gpucount"
                if len(parts) >= 4:
                    ref_key = "-".join(parts[:3])
            if ref_key not in ref_data_dict:
                missing_ref_keys.append(key_nam)
                continue
            for msg_size in act_data_dict[key_nam].keys():
                act_bus_bw = act_data_dict[key_nam][msg_size]['bus_bw']
                act_alg_bw = act_data_dict[key_nam][msg_size]['alg_bw']
                act_time = act_data_dict[key_nam][msg_size]['time']
                # Skip if reference doesn't have this message size
                if msg_size not in ref_data_dict[ref_key]:
                    missing_ref_msg_sizes += 1
                    continue
                ref_bus_bw = ref_data_dict[ref_key][msg_size]['bus_bw']
                ref_alg_bw = ref_data_dict[ref_key][msg_size]['alg_bw']
                act_time = act_data_dict[key_nam][msg_size]['time']
                ref_time = ref_data_dict[ref_key][msg_size]['time']

                html_lines = f'''
     <tr>
     <td>{collective}</td>
     <td>{data_type}</td>
     <td>{gpu_count}</td>
     <td>{msg_size}</td>
                 '''
                fp.write(html_lines)

                if float(act_alg_bw) < float(ref_alg_bw):
                    html_lines = '''<td><span class="label label-danger">''' + str(act_alg_bw) + '''</td>\n'''
                else:
                    html_lines = '''<td>''' + str(act_alg_bw) + '''</td>\n'''
                fp.write(html_lines)
                html_lines = '''<td>''' + str(ref_alg_bw) + '''</td>\n'''
                fp.write(html_lines)

                if float(act_bus_bw) < float(ref_bus_bw):
                    html_lines = '''<td><span class="label label-danger">''' + str(act_bus_bw) + '''</td>\n'''
                else:
                    html_lines = '''<td>''' + str(act_bus_bw) + '''</td>\n'''
                fp.write(html_lines)
                html_lines = '''<td>''' + str(ref_bus_bw) + '''</td>\n'''
                fp.write(html_lines)

                if float(act_time) > float(ref_time):
                    html_lines = '''<td><span class="label label-danger">''' + str(act_time) + '''</td>\n'''
                else:
                    html_lines = '''<td>''' + str(act_time) + '''</td>\n'''
                fp.write(html_lines)
                html_lines = '''<td>''' + str(ref_time) + '''</td>\n'''
                fp.write(html_lines)

        html_lines = '''
         </table>
         <br><br>
         '''
        fp.write(html_lines)

    # Print a concise summary instead of spamming per-key warnings
    if missing_ref_keys:
        sample = ", ".join(missing_ref_keys[:5])
        log.warning(
            f"Warning: {len(missing_ref_keys)} test keys not found in reference data; "
            f"skipped from table. Sample: {sample}"
        )
    if missing_ref_msg_sizes:
        log.warning(
            f"Warning: {missing_ref_msg_sizes} msg_size entries not found in reference data; skipped from table."
        )


def insert_chart(filename, chart_name):
    with open(filename, 'a') as fp:
        html_lines = f'''<div id="{chart_name}"></div>'''
        fp.write(html_lines)


def build_rdma_stats_table(
    filename,
    rdma_dict,
):
    """
    Append an HTML table summarizing RDMA statistics to the given file.

    Args:
      filename (str): Path to the HTML file to append to. The file is opened in append mode.
      rdma_dict (dict): Nested structure of RDMA stats per node and device. Expected format:
        {
          "nodeA": {
            "mlx5_0": {
              "ifname": "rdma0",
              "<counter_name>": "<numeric string or int>", ...
            },
            "mlx5_1": { ... }
          },
          "nodeB": { ... }
        }

    Behavior:
      - Determines the RDMA device columns based on the first node?s device list.
      - Writes a <h2> header and a DataTables-ready <table> with id="rdmastats".
      - Creates a column for each RDMA device index (rdma_device_0, rdma_device_1, ...).
      - For each node (row), inserts a nested table per RDMA device showing non-zero counters.
      - Highlights counters whose names match error-like patterns in red (label-danger).
      - Skips the "ifname" key from stats display.

    Notes:
      - Assumes every node has the same set of RDMA devices as the first node (node_0).
        If the set differs by node, columns may not align as expected.
      - Casting stats values to int may raise if the value is not numeric. Ensure inputs are clean.
      - Consider HTML-escaping node names, device names, and values to avoid HTML injection issues.
      - Closing the file explicitly (fp.close()) is unnecessary; the context manager handles it.
      - The nested table uses border=1 inline; consider CSS classes/styles for consistency.
      - The table relies on external JS/CSS (e.g., DataTables) initialized elsewhere in your page.
    """

    node_0 = list(rdma_dict.keys())[0]

    # Regex pattern for highlighting counters considered "error-like"
    err_pattern = 'err|retransmit|drop|discard|naks|invalid|oflow|out_of_buffer'

    # Determine device count from the first node?s keys
    rdma_device_list = rdma_dict[node_0].keys()
    device_count = len(rdma_device_list)

    # Open the HTML file in append mode; we assume header/body already started elsewhere
    with open(filename, 'a') as fp:
        html_lines = '''
<h2 id="rdmastatsid"></h2><br>
<h2 style="background-color: lightblue">RDMA Statistics Table</h2>
<table id="rdmastatstable" class="display cell-border">
  <thead>
  <tr>
  <th>Node</th>'''
        fp.write(html_lines)

        # Create one column header per RDMA device index from the first node
        for j in range(0, device_count):
            fp.write(f'<th>rdma_device_{j}</th>\n')
        fp.write('</tr></thead>\n')
        # End or header, let us start data rows ..

        # Begin writing data rows, one per node
        for node in rdma_dict.keys():
            # begin each node row
            fp.write(f'<tr><td>{node}</td>\n')
            # For each RDMA device on this node, render a nested table of non-zero counters
            for rdma_device in rdma_dict[node].keys():
                fp.write('<td><table border=1>\n')
                stats_dict = rdma_dict[node][rdma_device]
                fp.write(f'<tr><td>rdma_device</td><td>{rdma_device}</td></tr>\n')
                for stats_key in stats_dict.keys():
                    if stats_key != "ifname":
                        # Convert the value to int to check if it is non-zero
                        # NOTE: Will raise if value is not numeric; sanitize inputs as needed.
                        if int(stats_dict[stats_key]) > 0:
                            # If the counter name looks like an error, apply a red label for emphasis
                            if re.search(f'{err_pattern}', stats_key, re.I):
                                fp.write(
                                    f'<tr><td>{stats_key}</td><td><span class="label label-danger">{stats_dict[stats_key]}</td></tr>\n'
                                )
                            else:
                                fp.write(f'<tr><td>{stats_key}</td><td>{stats_dict[stats_key]}</td></tr>\n')
                fp.write('</table></td>\n')
            # End of each node row
            fp.write('</tr>\n')
        html_lines = '''
         </table>
         <br><br>
         '''
        fp.write(html_lines)
        fp.close()


def build_ethtool_stats_table(
    filename,
    d_dict,
):
    """
    Append an HTML table summarizing per-node ethtool statistics.

    Args:
      filename (str): Path to the HTML file to append to. Opened in append mode ('a').
      d_dict (dict): Nested dict of NIC stats per node and per device.
                     Expected structure:
                       {
                         "nodeA": {
                           "eth0": { "<counter_name>": "<numeric_str_or_int>", ... },
                           "eth1": { ... }
                         },
                         "nodeB": { ... }
                       }

    Behavior:
      - Determines device columns based on the device keys of the first node.
      - Writes a section header and a DataTables-compatible table (id="ethtoolstats").
      - For each node, creates a row with one cell per NIC containing a nested table of stats.
      - Only non-zero counters are displayed to keep the table concise.
      - Highlights counters whose names match an error-like regex (err_pattern) using a "label-danger" span.

    Notes and assumptions:
      - Assumes all nodes have (roughly) the same number/order of NICs as the first node.
      - Values are cast to int to determine > 0; if any value is non-numeric, int(...) will raise.
      - Consider HTML-escaping node/device names and values to avoid injecting HTML.
      - The explicit fp.close() at the end is not needed because the with-context closes the file.
      - The nested table styling uses border=1; consider CSS classes for consistent styling.
      - Ensure matching CSS (e.g., .label-danger) exists in your page?s styles for red highlighting.
    """

    # Use the first node to derive the table column layout and device count (one column per device index)
    node_0 = list(d_dict.keys())[0]
    eth_device_list = d_dict[node_0].keys()
    device_count = len(eth_device_list)

    # Regex to detect "error-like" counters for highlighting
    err_pattern = 'err|retransmit|drop|discard|naks|invalid|oflow|out_of_buffer|collision|reset|uncorrect'

    # Append to the HTML file (assumes an HTML <body> is already open)
    with open(filename, 'a') as fp:
        html_lines = '''
<h2 id="ethtoolstatsid"></h2><br>
<h2 style="background-color: lightblue">Ethtool Statistics Table</h2>
<table id="ethtoolstatstable" class="display cell-border">
  <thead>
  <tr>
  <th>Node</th>'''
        fp.write(html_lines)

        # Create device columns based on the first node's device count
        for j in range(0, device_count):
            fp.write(f'<th>eth_device_{j}</th>\n')
        fp.write('</tr></thead>\n')
        # End or header, let us start data rows ..

        # Start writing table rows, one per node
        for node in d_dict.keys():
            # begin each node row
            fp.write(f'<tr><td>{node}</td>\n')

            # For each NIC on this node, render a nested table of non-zero counters
            for eth_device in d_dict[node].keys():
                fp.write('<td><table border=1>\n')
                fp.write(f'<tr><td>eth_device</td><td>{eth_device}</td></tr>\n')
                stats_dict = d_dict[node][eth_device]
                for stats_key in stats_dict.keys():
                    # Show only counters with non-zero values to reduce noise
                    if int(stats_dict[stats_key]) > 0:
                        # Error-like counters: highlight in red
                        # NOTE: consider closing </span> to keep HTML well-formed.
                        if re.search(f'{err_pattern}', stats_key, re.I):
                            fp.write(
                                f'<tr><td>{stats_key}</td><td><span class="label label-danger">{stats_dict[stats_key]}</td></tr>\n'
                            )
                        else:
                            fp.write(f'<tr><td>{stats_key}</td><td>{stats_dict[stats_key]}</td></tr>\n')
                fp.write('</table></td>\n')
            # End of each node row
            fp.write('</tr>\n')
        html_lines = '''
         </table>
         <br><br>
         '''
        fp.write(html_lines)
        fp.close()


def build_snapshot_stats_diff_table(filename, d_dict, title, table_name, id_name):
    # Use the first node to derive the table column layout and device count (one column per device index)
    node_0 = list(d_dict.keys())[0]
    eth_device_list = d_dict[node_0].keys()
    device_count = len(eth_device_list)

    # Append to the HTML file (assumes an HTML <body> is already open)
    with open(filename, 'a') as fp:
        html_lines = f'''
<h2 id="{id_name}"></h2><br>
<h2 style="background-color: lightblue">{title}</h2>
<table id="{table_name}" class="display cell-border">
  <thead>
  <tr>
  <th>Node</th>'''
        fp.write(html_lines)

        # Create device columns based on the first node's device count
        for j in range(0, device_count):
            fp.write(f'<th>device_{j}</th>\n')
        fp.write('</tr></thead>\n')
        # End or header, let us start data rows ..

        # Start writing table rows, one per node
        for node in d_dict.keys():
            # begin each node row
            fp.write(f'<tr><td>{node}</td>\n')

            # For each NIC on this node, render a nested table of non-zero counters
            for eth_device in d_dict[node].keys():
                # Write the inside stats table only if err diff is seen
                if d_dict[node][eth_device]:
                    stats_dict = d_dict[node][eth_device]
                    non_zero_diff = False
                    for stats_key in stats_dict.keys():
                        if int(stats_dict[stats_key]['diff']) > 0:
                            non_zero_diff = True
                    if non_zero_diff:
                        fp.write('<td><table border=1>\n')
                        fp.write(f'<tr><td>{eth_device}</td><td>Before</td><td>After</td><td>Diff</td></tr>\n')
                        for stats_key in stats_dict.keys():
                            if int(stats_dict[stats_key]['diff']) > 0:
                                fp.write(
                                    f'<tr><td>{stats_key}</td><td>{stats_dict[stats_key]["before"]}</td><td>{stats_dict[stats_key]["after"]}</td><td><span class="label label-danger">{stats_dict[stats_key]["diff"]}</td></tr>\n'
                                )
                        fp.write('</table></td>\n')
                    else:
                        fp.write('<td></td>')
                else:
                    fp.write('<td> </td>')
            # End of each node row
            fp.write('</tr>\n')
        html_lines = '''
         </table>
         <br><br>
         '''
        fp.write(html_lines)
        fp.close()


def build_lldp_table(filename, lldp_dict):
    log.info('Build HTML training table')
    with open(filename, 'a') as fp:
        html_lines = '''
<h2 id="lldpid"></h2><br>
<h2 style="background-color: lightblue">Cluster LLDP Table</h2>
<table id="lldptable" class="display cell-border">
  <thead>
  <tr>
  <th>Node</th>
  <th>local Intf</th>
  <th>Neighbor Name</th>
  <th>Neighbor Descr</th>
  <th>Neighbor GMAC</th>
  <th>Neighbor Mgmt-IP</th>
  <th>Neighbor Port</th>
  </tr>
  </thead>
  '''
        fp.write(html_lines)
        for node in lldp_dict.keys():
            if 'lldp' in lldp_dict[node].keys():
                l_dict_list = lldp_dict[node]['lldp']['interface']
                for l_dict in l_dict_list:
                    intf = list(l_dict.keys())[0]
                    log.info("%s", intf)
                    log.info("%s", l_dict)
                    chassis_key = list(l_dict[intf]['chassis'].keys())[0]
                    chassis_dict = l_dict[intf]['chassis'][chassis_key]
                    if 'descr' in chassis_dict:
                        html_lines = f'''
                         <tr>
                         <td>{node}</td>
                         <td>{intf}</td>
                         <td>{chassis_key}</td>
                         <td>{chassis_dict['descr']}</td>
                         <td>{chassis_dict['id']['value']}</td>
                         <td>{chassis_dict['mgmt-ip']}</td>
                         <td>{l_dict[intf]['port']['id']['value']}</td>
                         </tr>'''
                        fp.write(html_lines)
        html_lines = '''
         </table>
         <br><br>
         '''
        fp.write(html_lines)
        fp.close()


def build_training_results_table(filename, out_dict, title):
    """
    Append an HTML table summarizing per-node training results.

    Args:
      filename (str): Path to the HTML file to append to. Opened in append mode ('a').
      out_dict (dict): Mapping of node -> metrics dict. Expected keys in each metrics dict:
                       {
                         "throughput_per_gpu": <value or list/str>,
                         "tokens_per_gpu": <value or list/str>,
                         "elapsed_time_per_iteration": <value or list/str>,
                         "nan_iterations": <value or list/str>,
                         "mem_usages": <value or list/str>,
                         ...
                       }
                       Values are written as-is; ensure they are strings or renderable.
      title (str): Section title to render above the table.

    Behavior:
      - Appends a section header (<h2>) using the provided title.
      - Appends a DataTables-compatible table (id="training") with a header row and one row per node.
      - For each node, writes a row containing:
          Node, Throughput per GPU, Tokens per GPU, Elapsed time per iteration, Nan iterations, Mem usage.

    Notes:
      - HTML escaping: This function does not escape title/node/values; if inputs can contain
        special HTML characters, consider escaping to avoid broken markup or injection issues.
      - Column header row: Thead includes an opening <tr> but should also include a closing </tr>;
        this function closes the row implicitly by starting the first data row. Consider adding </tr>.
      - Key names: The code expects 'mem_usages' in out_dict, but the column is labeled 'Mem usage'.
        Ensure key names match your data producers (or normalize here).
      - File handling: Explicit fp.close() is not necessary; the 'with' context manages closure.
      - Encoding: For non-ASCII content, consider using encoding='utf-8' when opening the file.
    """

    log.info('Build HTML training table')
    with open(filename, 'a') as fp:
        html_lines = (
            '''
<h2 id="trainingid"></h2><br>
<h2 style="background-color: lightblue">'''
            + title
            + '''</h2>
<table id="trainingtable" class="display cell-border">
  <thead>
  <tr>
  <th>Node</th>
  <th>Throughput per GPU</th>
  <th>Tokens per GPU</th>
  <th>Elapsed time per iteration</th>
  <th>Nan iterations</th>
  <th>Mem usage</th>
  <tr>
  </thead>'''
        )
        fp.write(html_lines)
        for node in out_dict.keys():
            d_dict = out_dict[node]
            html_lines = '''
  <tr>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  </tr>'''.format(
                node,
                d_dict['throughput_per_gpu'],
                d_dict['tokens_per_gpu'],
                d_dict['elapsed_time_per_iteration'],
                d_dict['nan_iterations'],
                d_dict['mem_usages'],
            )
            fp.write(html_lines)
        html_lines = '''
         </table>
         <br><br>
         '''
        fp.write(html_lines)
        fp.close()


def build_err_log_table(filename, d_dict, title, table_name, id_name):
    log.info(f'Build HTML Historic error table {title}')
    with open(filename, 'a') as fp:
        html_lines = f'''
<h2 id="{id_name}"></h2><br>
<h2 style="background-color: lightblue">{title}</h2>
<table id="{table_name}" class="display cell-border">
  <thead>
  <tr>
  <th>Node</th>
  <th>Error Logs</th>
  </tr>
  </thead>'''
        fp.write(html_lines)
        for node in d_dict.keys():
            html_lines = f'''
             <tr>
             <td>{node}</td>
             '''
            fp.write(html_lines)
            html_lines = '<td><span class="label label-danger">'
            for err_line in d_dict[node]:
                html_lines = html_lines + err_line + '<br>'
            html_lines = html_lines + '</td></tr>'
            fp.write(html_lines)
        html_lines = '''
         </table>
         <br><br>
         '''
        fp.write(html_lines)
        fp.close()


def build_html_nic_table(filename, rdma_dict, lshw_dict, ip_dict):
    log.info('Build HTML product table')
    """
    Append a Network Info HTML table to the given file, summarizing NIC and RDMA device state per node.

    Args:
      filename (str): Path to the HTML file to append to. File is opened in append mode ('a').
      rdma_dict (dict): Per-node RDMA/NIC mapping and status. Expected structure (example):
                        {
                          "node1": {
                            "rdma0": {
                              "eth_device": "ensX",
                              "device_status": "ACTIVE|DOWN|...",
                              "link_status": "UP|DOWN|..."
                            },
                            ...
                          },
                          ...
                        }
      lshw_dict (dict): Per-node LSHW details keyed by eth_device. Example:
                        { "node1": { "ensX": { "pci_bus": "0000:01:00.0" }, ... }, ... }
      ip_dict (dict): Per-node IP/MTU details keyed by eth_device. Example:
                      { "node1": { "ensX": { "mtu": 9000, "ipv4_addr_list": ["10.0.0.1/24"] }, ... }, ... }

    Behavior:
      - Writes a "Network Info" section and a DataTables-compatible table (id="nic").
      - For each node:
        * Iterates through its RDMA devices to build lists of:
          - Ethernet device names
          - RDMA device names
          - Link and device status
          - PCIe bus addresses (from lshw_dict)
          - MTU values (from ip_dict)
          - IPv4 address lists (from ip_dict)
        * Renders a single table row for the node, with each cell containing the collected list.
      - Leaves closing of the HTML document to the caller. Ensures only this section/table is closed.

    Notes:
      - This function writes Python lists directly into HTML cells (e.g., ['ensX', 'ensY']).
        Consider formatting as a comma-separated string or an unordered list (<ul><li>...).
      - No HTML escaping is performed; if device names or addresses can contain special characters,
        consider escaping content to prevent broken markup or injection.
      - The code assumes that for every RDMA device:
        - rdma_dict[node][rdma_dev]['eth_device'] exists
        - lshw_dict[node][eth_dev]['pci_bus'] exists
        - ip_dict[node][eth_dev]['mtu'] and ['ipv4_addr_list'] exist
        Missing keys will raise KeyError.
      - The explicit fp.close() is unnecessary in a with-context; it will be closed automatically.
      - Consider opening the file with encoding='utf-8' for portability.
    """

    with open(filename, 'a') as fp:
        html_lines = '''

<h2 id="nicid"></h2><br>
<h2 style="background-color: lightblue">Network Info</h2>
<table id="nictable" class="display cell-border">
  <thead>
  <tr>
  <th>Node</th>
  <th>Eth Device</th>
  <th>RDMA Device</th>
  <th>Link Status</th>
  <th>RDMA Status</th>
  <th>PCIe Bus</th>
  <th>MTU</th>
  <th>IPv4 Addr List</th>
  <th>IPv6 Addr List</th>
  </tr>
  </thead>'''
        fp.write(html_lines)

        # For each node, collect NIC/RDMA properties into lists (one row per node)
        for node in rdma_dict.keys():
            # Native InfiniBand ports have no netdev to look up in lshw_dict/ip_dict.
            rdma_dev_list = [dev for dev in rdma_dict[node] if rdma_dict[node][dev]['eth_device']]
            eth_dev_list = []
            dev_status_list = []
            link_status_list = []
            pcie_bus_list = []
            mtu_list = []
            ip_list = []
            ip6_list = []

            # Build each list by walking RDMA devices mapped to Ethernet interfaces
            for rdma_dev in rdma_dev_list:
                eth_dev = rdma_dict[node][rdma_dev]['eth_device']
                eth_dev_list.append(eth_dev)
                dev_status_list.append(rdma_dict[node][rdma_dev]['device_status'])
                link_status_list.append(rdma_dict[node][rdma_dev]['link_status'])
                pcie_bus_list.append(lshw_dict[node][eth_dev]['pci_bus'])
                mtu_list.append(ip_dict[node][eth_dev]['mtu'])
                ip_list.append(ip_dict[node][eth_dev]['ipv4_addr_list'])
                ip6_list.append(ip_dict[node][eth_dev]['ipv6_addr_list'])
            html_lines = '''
  <tr>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  </tr>
  '''.format(
                node,
                eth_dev_list,
                rdma_dev_list,
                link_status_list,
                dev_status_list,
                pcie_bus_list,
                mtu_list,
                ip_list,
                ip6_list,
            )
            fp.write(html_lines)
        html_lines = '''
         </table>
         <br><br>
         '''
        fp.write(html_lines)
        fp.close()


def build_html_cluster_product_table(filename, model_dict, fw_dict):
    """
    Append a 'Product Info' HTML table to the given file, summarizing GPU model and firmware info per node.

    Args:
      filename (str): Path to the HTML file to append to. Opened in append mode ('a').
      model_dict (dict): Per-node model metadata, expected structure:
                         {
                           "<node>": {
                             "card0": {
                               "Card Series": "...",
                               "GFX Version": "...",
                               "Card SKU": "..."
                             },
                             ...
                           },
                           ...
                         }
      fw_dict (dict): Per-node firmware metadata, expected structure:
                      {
                        "<node>": {
                          "card0": {
                            "MEC firmware version": "...",
                            "RLC firmware version": "...",
                            "RLC SRLC firmware version": "...",
                            "RLC SRLG firmware version": "...",
                            "RLC SRLS firmware version": "...",
                            "SDMA firmware version": "...",
                            "SMC firmware version": "...",
                            "SOS firmware version": "...",  # may be missing
                            "TA RAS firmware version": "...",
                            "TA XGMI firmware version": "...",
                            "VCN firmware version": "..."
                          },
                          ...
                        },
                        ...
                      }

    Behavior:
      - Writes a <h2> section title and an HTML table header (id="prod") compatible with DataTables.
      - Iterates nodes and reads model_dict[node]["card0"] and fw_dict[node]["card0"].
      - Ensures a value (dash) for 'SOS firmware version' if missing.
      - Appends one row per node with model/firmware columns.

    Important notes:
      - Column/header mismatch:
          The <thead> defines 12 columns, but the row formatter provides 1 (node)
          + 3 (model fields) + 11 (firmware fields) = 15 cells. DataTables/HTML
          requires the number of <th> (headers) to match the number of <td> cells
          per row. Either add headers for the extra RLC SRLC/SRLG/SRLS columns or
          remove them from rows to match headers.
      - Robustness:
          Accesses many dictionary keys directly; KeyError will occur if any key is missing
          (except 'SOS firmware version', which is defaulted to "-").
      - Security/HTML:
          Values are inserted verbatim without HTML-escaping. If values can contain
          special characters, consider escaping to avoid broken markup or injection.
      - File handling:
          The explicit fp.close() is unnecessary when using a 'with' context manager.
      - Encoding:
          Consider using encoding='utf-8' when opening files to ensure portability.
      - Single-GPU assumption:
          Function hardcodes "card0". If multiple cards per node need to be shown, extend
          to iterate cards.

    """

    log.info('Build HTML product table')

    # Append to the existing HTML file (assumes <body> already opened elsewhere)
    with open(filename, 'a') as fp:
        html_lines = '''

<h2 id="prodid"></h2><br>
<h2 style="background-color: lightblue">Firmware Info</h2>
<table id="prodtable" class="display cell-border">
  <thead>
  <tr>
  <th>Node</th>
  <th>Model</th>
  <th>GFX Ver</th>
  <th>SKU</th>
  <th>MEC Fw</th>
  <th>RLC Fw</th>
  <th>SDMA Fw</th>
  <th>SMC Fw</th>
  <th>SOS Fw</th>
  <th>TA RAS Fw</th>
  <th>TA XGMI Fw</th>
  <th>VCN Fw</th>
  </tr>
  </thead>'''
        fp.write(html_lines)
        # Emit one row per node
        # Only printing for card0, gpu0 .. assuming all GPUs running same version, otherwise the table size is so huge
        for node in model_dict.keys():
            m_dict = model_dict[node]["card0"]
            f_dict = fw_dict[node]["card0"]
            if 'SOS firmware version' not in f_dict:
                f_dict['SOS firmware version'] = "-"
            html_lines = '''
  <tr>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  </tr>
  '''.format(
                node,
                m_dict['Card Series'],
                m_dict['GFX Version'],
                m_dict['Card SKU'],
                f_dict['MEC firmware version'],
                f_dict['RLC firmware version'],
                f_dict['SDMA firmware version'],
                f_dict['SMC firmware version'],
                f_dict['SOS firmware version'],
                f_dict['TA RAS firmware version'],
                f_dict['TA XGMI firmware version'],
                f_dict['VCN firmware version'],
            )
            fp.write(html_lines)
        html_lines = '''
         </table>
         <br><br>
         '''
        fp.write(html_lines)
        fp.close()


def build_html_gpu_utilization_table(filename, use_dict):
    """
    Append a 'GPU Utilization' HTML table to the given file, summarizing per-GPU usage per node.

    Args:
      filename (str): Path to the HTML file to append to. The file is opened in append mode ('a').
      use_dict (dict): Nested per-node GPU utilization dictionary. Expected structure:
        {
          "<node>": {
            "card0": { "GPU use (%)": "<str|int>", "GFX Activity": "<str|int>" },
            "card1": { "GPU use (%)": "...", "GFX Activity": "..." },
            ...
            "card7": { ... }
          },
          ...
        }

    Behavior:
      - Writes a section header and a DataTables-compatible table (id="gpuuse").
      - Table is hardcoded for 8 GPUs (card0..card7), with two columns per card:
        * GPU use (%)
        * GFX Activity
      - Iterates nodes and emits one row per node, reading values from use_dict.
      - Leaves closing of the overall HTML document to the caller (only closes this table section).

    Notes/Assumptions:
      - The function assumes each node has keys 'card0' through 'card7' and the two metrics per card;
        missing keys will raise KeyError. If GPU counts vary, consider generating columns dynamically.
      - Values are written verbatim; if content can include special characters, consider HTML-escaping.
      - DataTables JS/CSS and initialization for table id "gpuusetable" must be included elsewhere in the page.
      - The explicit fp.close() is unnecessary in a with-context (file is closed automatically).
      - Consider opening the file with encoding='utf-8' for portability.
      - The thead contains an opening <tr> but no explicit closing </tr> before rows; browsers tolerate this,
        but you may want to add the closing tag for strict HTML validity.
    """

    log.info('Build HTML utilization table')
    with open(filename, 'a') as fp:
        html_lines = '''
<h2 id="gpuuseid"></h2><br>
<h2 style="background-color: lightblue">GPU Utilization</h2>
<table id="gpuusetable" class="display cell-border">
  <thead>
  <tr>
  <th>Node</th>
  <th>GPU0 use</th>
  <th>GPU0 GFX Activ</th>
  <th>GPU1 use</th>
  <th>GPU1 GFX Activ</th>
  <th>GPU2 use</th>
  <th>GPU2 GFX Activ</th>
  <th>GPU3 use</th>
  <th>GPU3 GFX Activ</th>
  <th>GPU4 use</th>
  <th>GPU4 GFX Activ</th>
  <th>GPU5 use</th>
  <th>GPU5 GFX Activ</th>
  <th>GPU6 use</th>
  <th>GPU6 GFX Activ</th>
  <th>GPU7 use</th>
  <th>GPU7 GFX Activ</th>
  </tr>
  </thead>'''
        fp.write(html_lines)

        # Emit one data row per node with utilization and activity for GPUs 0..7.
        for node in use_dict.keys():
            u_dict = use_dict[node]  # Per-node GPU metrics (expects card0..card7 with required keys)
            html_lines = '''
  <tr>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  </tr>'''.format(
                node,
                u_dict['card0']['GPU use (%)'],
                u_dict['card0']['GFX Activity'],
                u_dict['card1']['GPU use (%)'],
                u_dict['card1']['GFX Activity'],
                u_dict['card2']['GPU use (%)'],
                u_dict['card2']['GFX Activity'],
                u_dict['card3']['GPU use (%)'],
                u_dict['card3']['GFX Activity'],
                u_dict['card4']['GPU use (%)'],
                u_dict['card4']['GFX Activity'],
                u_dict['card5']['GPU use (%)'],
                u_dict['card5']['GFX Activity'],
                u_dict['card6']['GPU use (%)'],
                u_dict['card6']['GFX Activity'],
                u_dict['card7']['GPU use (%)'],
                u_dict['card7']['GFX Activity'],
            )
            fp.write(html_lines)
        html_lines = '''
         </table>
         <br><br>
         '''
        fp.write(html_lines)
        fp.close()


def build_html_mem_utilization_table(filename, use_dict, amd_dict):
    """
    Append a 'GPU Memory Utilization' HTML table to the given file, summarizing per-GPU
    memory usage and activity metrics per node.

    Args:
      filename (str): Path to the HTML file to append to. Opened in append mode ('a').
      use_dict (dict): Per-node, per-GPU utilization metrics (string/number values). Expected shape:
                       {
                         "node1": {
                           "card0": {
                             "GPU Memory Allocated (VRAM%)": "...",
                             "GPU Memory Read/Write Activity (%)": "...",
                             "Memory Activity": "...",
                             "Avg. Memory Bandwidth": "..."
                           },
                           ...
                           "card7": {...}
                         },
                         ...
                       }
      amd_dict (dict | list): Per-node, per-GPU VRAM usage from AMD SMI (rocm-smi/json).
                              May be a dict (ROCm 7.x style) or list (ROCm 6.x style):
                               - Dict: { "node1": { "gpu_data": [ { "mem_usage": { ... } }, ... ] }, ... }
                               - List: { "node1": [ { "mem_usage": { ... } }, ... ], ... }
                              Each GPU entry should contain:
                                mem_usage.total_vram.value
                                mem_usage.used_vram.value
                                mem_usage.free_vram.value

    Behavior:
      - Writes a section header and a DataTables-compatible table (id="memuse").
      - Header row includes 7 columns per GPU for 8 GPUs (G0..G7), plus the Node column.
      - For each node, outputs a row with:
          - total/used/free VRAM (MB) from amd_dict,
          - memory allocation %, read/write activity %, memory activity, and average memory bandwidth
            from use_dict for each of 8 GPUs (card0..card7).
      - Leaves closing of the overall HTML document to the caller (only closes this table section).

    Assumptions and caveats:
      - Assumes exactly 8 GPUs per node (card0..card7). Will raise KeyError/IndexError if missing.
      - Expects specific keys to exist in both use_dict and amd_dict; missing keys will raise.
      - Writes raw values directly into HTML (no escaping). If values may contain special characters,
        consider HTML-escaping to avoid broken markup/injection.
      - The thead closes correctly; ensure your page includes DataTables assets and initialization for #memuse.
      - For portability, consider opening the file with encoding='utf-8'.

    """

    log.info('Build HTML mem utilization table')
    with open(filename, 'a') as fp:
        html_lines = '''
<h2 id="memuseid"></h2><br>
<h2 style="background-color: lightblue">GPU Memory Utilization</h2>
<table id="memusetable" class="display cell-border">
  <thead>
  <tr>
  <th>Node</th>
         '''
        fp.write(html_lines)
        for i in range(0, 8):
            html_lines = f'''
  <th>G{i} Tot VRAM MB</th>
  <th>G{i} Used VRAM MB</th>
  <th>G{i} Free VRAM MB</th>
  <th>G{i} Mem Allocated</th>
  <th>G{i} Read/Write Acti</th>
  <th>G{i} Mem Acti</th>
  <th>G{i} Mem BW</th>
             '''
            fp.write(html_lines)
        html_lines = '''
  </tr>
  </thead>'''
        fp.write(html_lines)
        for node in use_dict.keys():
            html_lines = '''
  <tr>
  <td>{}</td>
             '''.format(node)
            fp.write(html_lines)
            u_dict = use_dict[node]
            # a_list = amd_dict[node]
            # Normalize amd_dict shape across ROCm versions:
            # - For ROCm 7.x: expect dict with "gpu_data" list
            # - For ROCm 6.x: expect list directly
            if isinstance(amd_dict[node], dict):
                # handling different between rocm6.x and 7.x
                if 'gpu_data' in amd_dict[node].keys():
                    a_list = amd_dict[node]['gpu_data']
            else:
                a_list = amd_dict[node]
            for j in range(0, 8):
                card = 'card' + str(j)
                a_dict = a_list[j]
                html_lines = '''
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  '''.format(
                    a_dict['mem_usage']['total_vram']['value'],
                    a_dict['mem_usage']['used_vram']['value'],
                    a_dict['mem_usage']['free_vram']['value'],
                    u_dict[card]['GPU Memory Allocated (VRAM%)'],
                    u_dict[card]['GPU Memory Read/Write Activity (%)'],
                    u_dict[card]['Memory Activity'],
                    u_dict[card]['Avg. Memory Bandwidth'],
                )
                fp.write(html_lines)
            html_lines = '''
  </tr>
             '''
            fp.write(html_lines)
        html_lines = '''
         </table>
         <br><br>
         '''
        fp.write(html_lines)
        fp.close()


def build_html_pcie_xgmi_metrics_table(filename, metrics_dict, amd_dict):
    """
    Append a 'GPU PCIe XGMI Metrics' HTML table to the given file.

    Args:
      filename (str): Path to the HTML file to append to. File is opened in append mode ('a').
      metrics_dict (dict): Per-node PCIe/XGMI counter metrics (typically parsed from tooling).
                           Expected structure (example):
                             {
                               "node1": {
                                 "card0": {
                                   "pcie_l0_to_recov_count_acc (Count)": "...",
                                   "pcie_replay_count_acc (Count)": "...",
                                   "pcie_replay_rover_count_acc (Count)": "...",
                                   "pcie_nak_sent_count_acc (Count)": "...",
                                   "pcie_nak_rcvd_count_acc (Count)": "...",
                                   "xgmi_link_width": "...",
                                   "xgmi_link_speed_gbps": "...",
                                   "xgmi_link_status_toggle_up_down": "...",
                                   "vram_max_bw_gbs": "..."
                                 },
                                 ...
                               },
                               ...
                             }
      amd_dict (dict | list): Per-node PCIe properties as reported by amd-smi (ROCm 6.x or 7.x JSON).
                              Two possible shapes:
                                - ROCm 7.x style: amd_dict[node] is dict with key 'gpu_data' -> list of per-GPU dicts
                                - ROCm 6.x style: amd_dict[node] is directly a list of per-GPU dicts
                              Each GPU entry should include:
                                {
                                  "pcie": {
                                    "width": "<int or str>",
                                    "speed": { "value": "<GT/s>" },
                                    "bandwidth": { "value": "<Mb/s>" }
                                  },
                                  ...
                                }

    Behavior:
      - Writes a section header and a DataTables-compatible table (id="pciexgmimetrics").
      - Table header includes 12 columns per GPU (G0..G7), plus the Node column.
      - For each node, emits a row with 8× per-GPU metrics:
          * PCIe: lane count, lane speed (GT/s), instantaneous bandwidth (Mb/s)
          * PCIe error counters: L0->Recovery, replay, replay rover, NAK sent/received
          * XGMI: link width, link speed (Gbps), link status toggles up/down
          * VRAM: max bandwidth (GB/s)
      - Uses amd_dict for PCIe properties and metrics_dict for counter/XGMI/VRAM fields.

    Assumptions and caveats:
      - Assumes 8 GPUs per node, referencing keys 'card0'..'card7' and list indices 0..7.
        This will raise if fewer GPUs exist.
      - Dict lookups are direct; missing keys will raise KeyError. Consider adding guards.
      - Values are written verbatim to HTML with no escaping. If values can include special chars,
        HTML-escape them to avoid broken markup/injection.
      - The function prints a status and relies on DataTables CSS/JS and initialization elsewhere.
      - Uses 'with open(..., "a")' context; no need to call fp.close() explicitly.

    """

    log.info('Build HTML PCIe metrics table')
    with open(filename, 'a') as fp:
        html_lines = '''
<h2 id="pciexgmimetid"></h2><br>
<h2 style="background-color: lightblue">GPU PCIe XGMI Metrics Table</h2>
<table id="pciexgmimettable" class="display cell-border">
  <thead>
  <tr>
  <th>Node</th>
         '''
        fp.write(html_lines)
        for i in range(0, 8):
            html_lines = f'''
  <th>G{i} pcie lanes</th>
  <th>G{i} pcie lane speed GT/s:</th>
  <th>G{i} PCIe BW inst Mb/s</th>
  <th>G{i} PCIe l0 to recov count acc</th>
  <th>G{i} PCIe replay count acc</th>
  <th>G{i} PCIe replay rover count acc</th>
  <th>G{i} PCIe nak sent count acc</th>
  <th>G{i} PCIe nak rcvd count acc</th>
  <th>G{i} XGMI link width</th>
  <th>G{i} XGMI link Speed Gbps</th>
  <th>G{i} XGMI link status toggle up/down</th>
  <th>G{i} VRAM max BW GB/s</th>
             '''
            fp.write(html_lines)
        html_lines = '''
  </tr>
  </thead>'''
        fp.write(html_lines)
        for node in metrics_dict.keys():
            html_lines = '''
  <tr>
  <td>{}</td>
             '''.format(node)
            fp.write(html_lines)
            d_dict = metrics_dict[node]
            if isinstance(amd_dict[node], dict):
                # handling different between rocm6.x and 7.x
                if 'gpu_data' in amd_dict[node].keys():
                    a_list = amd_dict[node]['gpu_data']
            else:
                a_list = amd_dict[node]
            for j in range(0, 8):
                card = 'card' + str(j)
                a_dict = a_list[j]
                html_lines = '''
  <td>{}</td>
  <td>{}</td>
  <td>{}</td>
  '''.format(a_dict['pcie']['width'], a_dict['pcie']['speed']['value'], a_dict['pcie']['bandwidth']['value'])
                fp.write(html_lines)

                if int(d_dict[card]['pcie_l0_to_recov_count_acc (Count)']) > 100:
                    html_lines = '''<td><span class="label label-danger">{}</td>'''.format(
                        d_dict[card]['pcie_l0_to_recov_count_acc (Count)']
                    )
                else:
                    html_lines = '''<td>{}</td>'''.format(d_dict[card]['pcie_l0_to_recov_count_acc (Count)'])
                fp.write(html_lines)

                html_lines = '''
  <td>{}</td>
  <td>{}</td>'''.format(
                    d_dict[card]['pcie_replay_count_acc (Count)'], d_dict[card]['pcie_replay_rover_count_acc (Count)']
                )
                fp.write(html_lines)

                if int(d_dict[card]['pcie_nak_sent_count_acc (Count)']) > 5:
                    html_lines = '''<td><span class="label label-danger">{}</td>'''.format(
                        d_dict[card]['pcie_nak_sent_count_acc (Count)']
                    )
                else:
                    html_lines = '''<td>{}</td>'''.format(d_dict[card]['pcie_nak_sent_count_acc (Count)'])
                fp.write(html_lines)

                if int(d_dict[card]['pcie_nak_rcvd_count_acc (Count)']) > 5:
                    html_lines = '''<td><span class="label label-danger">{}</td>'''.format(
                        d_dict[card]['pcie_nak_rcvd_count_acc (Count)']
                    )
                else:
                    html_lines = '''<td>{}</td>'''.format(d_dict[card]['pcie_nak_rcvd_count_acc (Count)'])
                fp.write(html_lines)

                if int(d_dict[card]['xgmi_link_width']) < 16:
                    html_lines = '''<td><span class="label label-danger">{}</td>'''.format(
                        d_dict[card]['xgmi_link_width']
                    )
                else:
                    html_lines = '''<td>{}</td>'''.format(d_dict[card]['xgmi_link_width'])
                fp.write(html_lines)

                if int(d_dict[card]['xgmi_link_speed (Gbps)']) < 32:
                    html_lines = '''<td><span class="label label-danger">{}</td>'''.format(
                        d_dict[card]['xgmi_link_speed (Gbps)']
                    )
                else:
                    html_lines = '''<td>{}</td>'''.format(d_dict[card]['xgmi_link_speed (Gbps)'])
                fp.write(html_lines)

                html_lines = '''
  <td>{}</td>
  <td>{}</td>
  '''.format(
                    d_dict[card]['xgmi_link_status (Up/Down)'],
                    d_dict[card]['vram_max_bandwidth (GB/s)'],
                )
                fp.write(html_lines)
            html_lines = '''
  </tr>
             '''
            fp.write(html_lines)
        html_lines = '''
         </table>
         <br><br>
         '''
        fp.write(html_lines)
        fp.close()


def build_html_error_table(filename, metrics_dict, amd_dict):
    """
    Append a 'GPU Error Metrics' HTML table to the given file, summarizing ECC and PCIe error counters
    per GPU (assumes 8 GPUs per node).

    Args:
      filename (str): Path to the HTML file to append to. File is opened in append mode ('a').
      metrics_dict (dict): Per-node error counters for each GPU (e.g., PCIe counters).
                           Expected structure (example):
                             {
                               "node1": {
                                 "card0": {
                                   "pcie_l0_to_recov_count_acc (Count)": "...",
                                   "pcie_replay_count_acc (Count)": "...",
                                   "pcie_replay_rover_count_acc (Count)": "...",
                                   "pcie_nak_sent_count_acc (Count)": "...",
                                   "pcie_nak_rcvd_count_acc (Count)": "...",
                                   ...
                                 },
                                 "card1": { ... },
                                 ...
                               },
                               ...
                             }
      amd_dict (dict | list): Per-node ROCm SMI data including ECC counts.
                              - ROCm 7.x style: amd_dict[node] is a dict with 'gpu_data' -> list of GPU dicts
                              - ROCm 6.x style: amd_dict[node] is a list of GPU dicts
                              Each GPU entry should include:
                                {
                                  "ecc": {
                                    "total_correctable_count": "<int-like>",
                                    "total_uncorrectable_count": "<int-like>",
                                    "total_deferred_count": "<int-like>",
                                    "cache_correctable_count": "<int-like>",
                                    "cache_uncorrectable_count": "<int-like>"
                                  },
                                  ...
                                }

    Behavior:
      - Writes a section title and a DataTables-compatible table (id="error") with:
        Node + 10 columns per GPU (G0..G7):
          ECC: correct, uncorrect, deferred, cache correct, cache uncorrect
          PCIe: L0->Recovery, replay, replay rover, NAK sent, NAK received
      - For each node, emits one row with 8× per-GPU values.
      - Highlights ECC values > 0 in red using a 'label label-danger' span.

    Assumptions and caveats:
      - Assumes 8 GPUs per node (card0..card7) and that amd_dict and metrics_dict contain aligned data.
      - Direct dict/index access will raise KeyError/IndexError if shapes/keys differ.
      - Values are written verbatim to HTML (no escaping). Escape if inputs may contain special chars.
      - The snippet provided appears truncated (e.g., missing code for remaining ECC fields and PCIe counters,
        missing closing tags for some spans/td/tr/table). Ensure the full implementation closes all tags
        and writes all declared columns in the header.
      - Consider opening files with encoding='utf-8' for portability.

    """

    log.info('Build HTML Error table')
    with open(filename, 'a') as fp:
        html_lines = '''
<h2 id="gpuerrorid"></h2><br>
<h2 style="background-color: lightblue">GPU Error Metrics Table</h2>
<table id="gpuerrortable" class="display cell-border">
  <thead>
  <tr>
  <th>Node</th>
         '''
        fp.write(html_lines)
        for i in range(0, 8):
            html_lines = f'''
  <th>G{i} ECC correct</th>
  <th>G{i} ECC uncorrect</th>
  <th>G{i} ECC deferred</th>
  <th>G{i} ECC cache correct</th>
  <th>G{i} ECC cache uncorrect</th>
  <th>G{i} PCIe l0 to recov count acc</th>
  <th>G{i} PCIe replay count acc</th>
  <th>G{i} PCIe replay rover count acc</th>
  <th>G{i} PCIe nak sent count acc</th>
  <th>G{i} PCIe nak rcvd count acc</th>
             '''
            fp.write(html_lines)
        html_lines = '''
  </tr>
  </thead>'''
        fp.write(html_lines)
        for node in metrics_dict.keys():
            html_lines = '''
  <tr>
  <td>{}</td>
             '''.format(node)
            fp.write(html_lines)
            d_dict = metrics_dict[node]
            if isinstance(amd_dict[node], dict):
                # handling different between rocm6.x and 7.x
                if 'gpu_data' in amd_dict[node].keys():
                    a_list = amd_dict[node]['gpu_data']
            else:
                a_list = amd_dict[node]
            for j in range(0, 8):
                card = 'card' + str(j)
                a_dict = a_list[j]
                if int(a_dict['ecc']['total_correctable_count']) > 0:
                    html_lines = '''
                     <td><span class="label label-danger">{}</td>
                     '''.format(a_dict['ecc']['total_correctable_count'])
                else:
                    html_lines = '''
                     <td>{}</td>
                     '''.format(a_dict['ecc']['total_correctable_count'])
                fp.write(html_lines)

                if int(a_dict['ecc']['total_uncorrectable_count']) > 0:
                    html_lines = '''
                     <td><span class="label label-danger">{}</td>
                     '''.format(a_dict['ecc']['total_uncorrectable_count'])
                else:
                    html_lines = '''
                     <td>{}</td>
                     '''.format(a_dict['ecc']['total_uncorrectable_count'])
                fp.write(html_lines)

                if int(a_dict['ecc']['total_deferred_count']) > 0:
                    html_lines = '''
                     <td><span class="label label-danger">{}</td>
                     '''.format(a_dict['ecc']['total_deferred_count'])
                else:
                    html_lines = '''
                     <td>{}</td>
                     '''.format(a_dict['ecc']['total_deferred_count'])
                fp.write(html_lines)

                if int(a_dict['ecc']['cache_correctable_count']) > 0:
                    html_lines = '''
                     <td><span class="label label-danger">{}</td>
                     '''.format(a_dict['ecc']['cache_correctable_count'])
                else:
                    html_lines = '''
                     <td>{}</td>
                     '''.format(a_dict['ecc']['cache_correctable_count'])
                fp.write(html_lines)

                if int(a_dict['ecc']['cache_uncorrectable_count']) > 0:
                    html_lines = '''
                     <td><span class="label label-danger">{}</td>
                     '''.format(a_dict['ecc']['cache_uncorrectable_count'])
                else:
                    html_lines = '''
                     <td>{}</td>
                     '''.format(a_dict['ecc']['cache_uncorrectable_count'])
                fp.write(html_lines)

                if int(d_dict[card]['pcie_l0_to_recov_count_acc (Count)']) > 4:
                    html_lines = '''
                     <td><span class="label label-danger">{}</td>
                     '''.format(d_dict[card]['pcie_l0_to_recov_count_acc (Count)'])
                else:
                    html_lines = '''
                     <td>{}</td>
                     '''.format(d_dict[card]['pcie_l0_to_recov_count_acc (Count)'])
                fp.write(html_lines)

                if int(d_dict[card]['pcie_replay_count_acc (Count)']) > 0:
                    html_lines = '''
                     <td><span class="label label-danger">{}</td>
                     '''.format(d_dict[card]['pcie_replay_count_acc (Count)'])
                else:
                    html_lines = '''
                     <td>{}</td>
                     '''.format(d_dict[card]['pcie_replay_count_acc (Count)'])
                fp.write(html_lines)

                if int(d_dict[card]['pcie_replay_rover_count_acc (Count)']) > 0:
                    html_lines = '''
                     <td><span class="label label-danger">{}</td>
                     '''.format(d_dict[card]['pcie_replay_rover_count_acc (Count)'])
                else:
                    html_lines = '''
                     <td>{}</td>
                     '''.format(d_dict[card]['pcie_replay_rover_count_acc (Count)'])
                fp.write(html_lines)

                if int(d_dict[card]['pcie_nak_sent_count_acc (Count)']) > 0:
                    html_lines = '''
                     <td><span class="label label-danger">{}</td>
                     '''.format(d_dict[card]['pcie_nak_sent_count_acc (Count)'])
                else:
                    html_lines = '''
                     <td>{}</td>
                     '''.format(d_dict[card]['pcie_nak_sent_count_acc (Count)'])
                fp.write(html_lines)

                if int(d_dict[card]['pcie_nak_rcvd_count_acc (Count)']) > 0:
                    html_lines = '''
                     <td><span class="label label-danger">{}</td>
                     '''.format(d_dict[card]['pcie_nak_rcvd_count_acc (Count)'])
                else:
                    html_lines = '''
                     <td>{}</td>
                     '''.format(d_dict[card]['pcie_nak_rcvd_count_acc (Count)'])
                fp.write(html_lines)

            html_lines = '''
  </tr>
             '''
            fp.write(html_lines)
        html_lines = '''
         </table>
         <br><br>
         '''
        fp.write(html_lines)
        fp.close()


# TO BE DONE
"""
def build_html_env_metrics_table():
    log.info('Build HTML env metrics table')
    with open(filename, 'a') as fp:
        html_lines = '''
<h2 id="envmetricsid"></h2><br>
<h2 style="background-color: lightblue">GPU Environmental Metrics Table</h2>
<table id="envmetricstable" class="display cell-border">
  <thead>
  <tr>
  <th>Node</th>
  <th>G0 Temp hotspot</th>
  <th>G0 Temp Mem</th>
  <th>G0 Temp vrsoc</th>
  <th>G0 gfxclk</th>
  <th>G0 uclk</th>
  <th>G0 pcie lanes</th>
  <th>G0 pcie speed GT/s:</th>
  <th>G0 gfx activi</th>
  <th>G0 mem activi</th>
  <th>G0 PCIe BW acc GB/s</th>
  <th>G0 PCIe BW inst GB/s</th>
  <th>G0 socket power</th>
  <th>G0 XGMI link width</th>
  <th>G0 XGMI link Speed Gbps</th>
  <th>G0 PCIe l0 to recov count acc</th>
  <th>G0 PCIe replay count acc</th>
  <th>G0 PCIe nak sent count acc</th>
  <th>G0 PCIe nak rcvd count acc</th>
  <th>G0 VRAM max BW GB/s</th>
         '''
"""


def build_html_config_table():
    log.info('Build config table')


_TOPO_PCI_SEP = ":"


def _topo_pci_domain(bdf):
    '''PCI domain of a BDF (e.g. "0001:01:00.0" -> "0001").

    The domain is what pairs a GPU with its backend NICs: a GPU at 0001:01:00.0
    and NICs at 0001:44:00.0 / 0001:48:00.0 share domain "0001".
    '''
    if not bdf:
        return ''
    return bdf.split(_TOPO_PCI_SEP)[0]


def _topo_nics_by_domain(node):
    '''Group a node's NICs (those with a BDF) by PCI domain -> [nic, ...].'''
    by_domain = {}
    for nic in node.get('nics', []):
        if nic.get('bdf'):
            by_domain.setdefault(_topo_pci_domain(nic['bdf']), []).append(nic)
    return by_domain


def _topo_gpu_peer_links(node):
    '''Per-GPU peer link map for a node: {src_bdf: {peer_bdf: (link_type, weight)}}.

    SELF links are dropped; only cross-GPU links remain.
    '''
    links = {}
    for gpu in node.get('gpus', []):
        src = gpu.get('bdf')
        links[src] = {}
        for link in gpu.get('links', []):
            if (link.get('link_type') or '').upper() == 'SELF':
                continue
            links[src][link.get('peer_bdf')] = (link.get('link_type'), link.get('weight'))
    return links


# Entity + link palette. Colours identify entity kind (CPU/GPU/NIC); link TYPE is
# carried by line style (solid red = XGMI, dashed grey = PCIe) plus a legend, so
# the XGMI-vs-PCIe distinction never relies on colour alone. Validated for CVD.
_TOPO_CSS = '''
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e2e1dc;
--gpu:#2a78d6;--nic:#1baf7a;--cpu:#4a3aa7;--xgmi:#e34948;--pcie:#8a8a8a;}
*{box-sizing:border-box}
body{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:var(--surface);color:var(--ink);}
.wrap{max-width:1180px;margin:0 auto;padding:24px;}
h2{font-size:20px;margin:0 0 4px}.sub{color:var(--ink2);font-size:13px;margin:0 0 16px}
.sum{background:#eef6ff;padding:8px 12px;border-radius:6px;font-size:13px;margin:0 0 16px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin:10px 0 18px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.dot{width:10px;height:10px;border-radius:3px;display:inline-block}
.node-card{border:1px solid var(--line);border-radius:10px;margin:0 0 20px;overflow:hidden}
.node-head{background:#f3f4f6;padding:10px 14px;font-weight:600;display:flex;justify-content:space-between;align-items:center}
.node-head .meta{font-weight:400;color:var(--ink2);font-size:12px}
.node-body{padding:14px;display:grid;grid-template-columns:1fr;gap:16px}
@media(min-width:880px){.node-body{grid-template-columns:290px 1fr}}
.cpu-card{border:1px solid var(--line);border-radius:8px;padding:12px;height:fit-content}
.cpu-title{font-weight:600;color:var(--cpu);font-size:14px;margin:0 0 2px}
.cpu-sub{color:var(--ink2);font-size:11px;font-family:ui-monospace,monospace;margin:0 0 10px}
.numa{border-left:3px solid var(--cpu);background:#f6f5fc;border-radius:4px;padding:7px 10px;margin:8px 0}
.numa b{font-size:12px}.numa .cl{font-family:ui-monospace,monospace;font-size:11px;color:var(--ink2);display:block;margin-top:2px}
.fe{display:flex;align-items:center;gap:5px;margin-top:6px;font-size:11px;color:#0b5}
.fe .dot{background:var(--nic)}
.kv{display:flex;justify-content:space-between;font-size:12px;padding:2px 0;border-bottom:1px dotted var(--line)}
.kv span:last-child{font-family:ui-monospace,monospace;color:var(--ink2)}
.dimm-list{margin-top:10px;font-size:12px}
.dimm-list summary{cursor:pointer;color:var(--mem);font-weight:600}
.dimm{border-left:3px solid var(--mem);background:#fdf3ee;border-radius:4px;padding:5px 9px;margin:6px 0;font-size:11px}
.dimm .cl{font-family:ui-monospace,monospace;color:var(--ink2)}
.muted{color:var(--ink2)}
pre{background:#f6f8fa;padding:12px;overflow:auto;font-size:11px}
'''


def _topo_legend():
    return (
        '<div class="legend">'
        '<span><i class="dot" style="background:var(--cpu)"></i>CPU / NUMA</span>'
        '<span><i class="dot" style="background:var(--gpu)"></i>GPU</span>'
        '<span><i class="dot" style="background:var(--nic)"></i>NIC</span>'
        '<span><svg width="26" height="10"><line x1="0" y1="5" x2="26" y2="5" '
        'stroke="var(--xgmi)" stroke-width="3"/></svg>XGMI (GPU&harr;GPU)</span>'
        '<span><svg width="26" height="10"><line x1="0" y1="5" x2="26" y2="5" '
        'stroke="var(--pcie)" stroke-width="2" stroke-dasharray="4 3"/></svg>PCIe (CPU&harr;GPU)</span>'
        '</div>'
    )


def _topo_node_meta(node):
    cpu = node.get('cpu', {})
    nic_count = len([n for n in node.get('nics', []) if n.get('bdf')])
    name = node.get('hostname') or node.get('host', '')
    return (
        f'{html_escape(name)} &middot; {node.get("gpu_count", 0)} GPUs '
        f'&middot; {nic_count} NICs &middot; {cpu.get("numa_nodes", "?")} NUMA nodes'
    )


def _topo_dimm_summary(node):
    '''(total_gb, count, "<type> <speed>") for the populated DIMMs, or (None, 0, "").

    Falls back to the lsmem total when per-DIMM detail is unavailable (no sudo).
    '''
    dimms = node.get('dimms', [])
    if not dimms:
        total = (node.get('mem_summary') or {}).get('total_online')
        return (total, 0, '') if total else (None, 0, '')
    total_gb = 0
    for dimm in dimms:
        head = (dimm.get('size') or '').split()
        if head and head[0].isdigit():
            total_gb += int(head[0])
    first = dimms[0]
    spec = ' '.join(p for p in (first.get('type'), first.get('speed')) if p)
    return (f'{total_gb} GB' if total_gb else None, len(dimms), spec)


def _topo_scale_out_summary(node):
    '''Compact scale-out fabric label from a node's RDMA records, or None.

    Counts the active data-plane RDMA ports (excluding the front-end/mgmt NIC in
    PCI domain 0000) and names the fabric (RoCE/IB), line rate and generation --
    e.g. "8x 400G RoCE (NDR), all ACTIVE". This is the inter-node scale-out
    capacity, discoverable from the node even though the wiring is not.
    '''
    backend = [r for r in node.get('rdma', []) if (r.get('bdf') or '').split(':')[0] != '0000']
    if not backend:
        return None
    active = [r for r in backend if r.get('active')]
    first = active[0] if active else backend[0]
    gbps = first.get('gbps')
    rate = f'{gbps}G' if gbps else (first.get('rate') or '?')
    fabric = first.get('fabric') or first.get('link_layer') or 'RDMA'
    gen = f' ({first.get("gen")})' if first.get('gen') else ''
    state = 'all ACTIVE' if len(active) == len(backend) else f'{len(active)}/{len(backend)} ACTIVE'
    return f'{len(backend)}&times; {rate} {fabric}{gen}, {state}'


def _topo_cpu_panel(node):
    '''Left panel: CPU identity, memory, NUMA domains (with NVMe), front-end NIC, DIMMs.'''
    cpu = node.get('cpu', {})
    front_end = _topo_nics_by_domain(node).get('0000', [])
    mem_total, dimm_count, dimm_spec = _topo_dimm_summary(node)
    nvme = node.get('nvme', [])
    rows = [
        '<div class="cpu-title">cpu0</div>',
        f'<div class="cpu-sub">{html_escape(cpu.get("model_name", "CPU"))}</div>',
        f'<div class="kv"><span>Sockets</span><span>{cpu.get("sockets", "?")}</span></div>',
        f'<div class="kv"><span>Cores / socket</span><span>{cpu.get("cores_per_socket", "?")}</span></div>',
        f'<div class="kv"><span>Threads total</span><span>{cpu.get("cpus", "?")}</span></div>',
        f'<div class="kv"><span>NUMA nodes</span><span>{cpu.get("numa_nodes", "?")}</span></div>',
    ]
    if mem_total:
        detail = f'{dimm_count}&times; {html_escape(dimm_spec)}' if dimm_count else 'total online'
        rows.append(f'<div class="kv"><span>Memory</span><span>{mem_total} ({detail})</span></div>')
    if nvme:
        rows.append(f'<div class="kv"><span>NVMe drives</span><span>{len(nvme)}</span></div>')
    if node.get('scale_up_mode'):
        rows.append(
            f'<div class="kv"><span>Scale-up (GPU)</span>'
            f'<span>XGMI &middot; {html_escape(node["scale_up_mode"])}</span></div>'
        )
    scale_out = _topo_scale_out_summary(node)
    if scale_out:
        rows.append(f'<div class="kv"><span>Scale-out (NIC)</span><span>{scale_out}</span></div>')

    # NVMe grouped by the NUMA node its controller reports.
    nvme_by_numa = {}
    for drive in nvme:
        nvme_by_numa.setdefault(str(drive.get('numa_node')), []).append(drive)

    for i, numa in enumerate(node.get('numa', [])):
        extra = ''
        if i == 0 and front_end:
            nic = front_end[0]
            extra += (
                f'<span class="fe"><i class="dot"></i>{html_escape(nic["name"])} '
                f'({html_escape(nic["bdf"])}) &middot; front-end</span>'
            )
        # numa["node"] is like "node1"; its index matches the NVMe numa_node.
        node_idx = numa['node'].replace('node', '')
        for drive in nvme_by_numa.get(node_idx, []):
            cap = drive.get('capacity') or ''
            extra += (
                f'<span class="fe" style="color:var(--mem)">'
                f'<i class="dot" style="background:var(--mem)"></i>'
                f'{html_escape(drive["name"])} {html_escape(cap)} &middot; NVMe</span>'
            )
        rows.append(
            f'<div class="numa"><b>{html_escape(numa["node"])}</b>'
            f'<span class="cl">cpus={html_escape(numa["cpulist"])}</span>{extra}</div>'
        )

    # Per-DIMM detail (only when dmidecode succeeded).
    dimms = node.get('dimms', [])
    if dimms:
        dimm_rows = ''.join(
            f'<div class="dimm"><b>{html_escape(d.get("bank") or d.get("locator") or "DIMM")}</b> '
            f'{html_escape(d.get("size") or "")} {html_escape(d.get("type") or "")} '
            f'{html_escape(d.get("speed") or "")} '
            f'<span class="cl">{html_escape(d.get("manufacturer") or "")}</span></div>'
            for d in dimms
        )
        rows.append(
            f'<details class="dimm-list"><summary>{len(dimms)} populated DIMM slots</summary>{dimm_rows}</details>'
        )
    return '<div class="cpu-card">' + ''.join(rows) + '</div>'


def _topo_mesh_svg(node):
    '''Right panel: CPU block, CPU->GPU PCIe host links, GPU XGMI mesh, backend NICs.'''
    gpus = node.get('gpus', [])
    if not gpus:
        return (
            '<div class="muted" style="padding:40px;text-align:center;'
            'border:1px dashed var(--line);border-radius:8px">'
            'No GPUs on this node (amdgpu driver not loaded). '
            'CPU and NICs shown on the left.</div>'
        )
    links = _topo_gpu_peer_links(node)
    nics_by_domain = _topo_nics_by_domain(node)
    width, height = 720, 470
    cpu_x, cpu_y = 70, height / 2
    cx, cy, radius = 430, height / 2, 140

    pos = {}
    count = len(gpus)
    for i, gpu in enumerate(gpus):
        ang = -math.pi / 2 + 2 * math.pi * i / max(count, 1)
        pos[gpu['bdf']] = (cx + radius * math.cos(ang), cy + radius * math.sin(ang))

    svg = [f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px">']

    # CPU -> GPU host PCIe x16 links, behind the nodes.
    for gpu in gpus:
        x, y = pos[gpu['bdf']]
        svg.append(
            f'<line x1="{cpu_x + 34:.0f}" y1="{cpu_y:.0f}" x2="{x:.0f}" y2="{y:.0f}" '
            f'stroke="var(--pcie)" stroke-width="1.6" stroke-dasharray="5 4" opacity="0.65"/>'
        )
    first_x, first_y = pos[gpus[0]['bdf']]
    mid_x, mid_y = (cpu_x + 34 + first_x) / 2, (cpu_y + first_y) / 2
    svg.append(
        f'<text x="{mid_x:.0f}" y="{mid_y - 4:.0f}" fill="var(--pcie)" font-size="9" '
        f'text-anchor="middle">PCIe x16</text>'
    )

    # GPU<->GPU edges: solid red XGMI, dashed grey PCIe. Deduped undirected.
    drawn = set()
    for src, peers in links.items():
        for dst, (link_type, _weight) in peers.items():
            key = tuple(sorted((src, dst)))
            if key in drawn or src not in pos or dst not in pos:
                continue
            drawn.add(key)
            x1, y1 = pos[src]
            x2, y2 = pos[dst]
            if (link_type or '').upper() == 'XGMI':
                svg.append(
                    f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                    f'stroke="var(--xgmi)" stroke-width="2.5" opacity="0.85"/>'
                )
            else:
                svg.append(
                    f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                    f'stroke="var(--pcie)" stroke-width="2" stroke-dasharray="5 4"/>'
                )

    # CPU block.
    cpu = node.get('cpu', {})
    svg.append(f'<rect x="{cpu_x - 34:.0f}" y="{cpu_y - 40:.0f}" width="68" height="80" rx="8" fill="var(--cpu)"/>')
    svg.append(
        f'<text x="{cpu_x:.0f}" y="{cpu_y - 6:.0f}" text-anchor="middle" fill="#fff" '
        f'font-size="12" font-weight="600">cpu0</text>'
    )
    svg.append(
        f'<text x="{cpu_x:.0f}" y="{cpu_y + 10:.0f}" text-anchor="middle" fill="#dcd8f5" '
        f'font-size="8">{cpu.get("cores_per_socket", "?")}C</text>'
    )
    svg.append(
        f'<text x="{cpu_x:.0f}" y="{cpu_y + 22:.0f}" text-anchor="middle" fill="#dcd8f5" '
        f'font-size="8">{cpu.get("numa_nodes", "?")} NUMA</text>'
    )

    # GPU circles + their backend NICs (same PCI domain) radiating outward.
    for gpu in gpus:
        x, y = pos[gpu['bdf']]
        domain = _topo_pci_domain(gpu['bdf'])
        ang = math.atan2(y - cy, x - cx)
        for j, nic in enumerate(nics_by_domain.get(domain, [])):
            nx = x + 66 * math.cos(ang)
            ny = y + 66 * math.sin(ang) + (j - 0.5) * 24
            svg.append(
                f'<line x1="{x:.0f}" y1="{y:.0f}" x2="{nx:.0f}" y2="{ny:.0f}" '
                f'stroke="var(--nic)" stroke-width="1.4" opacity="0.55"/>'
            )
            up = nic.get('operstate') == 'up'
            fill = '#eafaf3' if up else '#f3f4f6'
            stroke = 'var(--nic)' if up else '#bbb'
            text_fill = '#0b5' if up else '#999'
            svg.append(
                f'<rect x="{nx - 26:.0f}" y="{ny - 9:.0f}" width="58" height="18" rx="4" '
                f'fill="{fill}" stroke="{stroke}"/>'
            )
            svg.append(
                f'<text x="{nx + 3:.0f}" y="{ny + 4:.0f}" text-anchor="middle" '
                f'fill="{text_fill}" font-size="8">{html_escape(nic["name"][:9])}</text>'
            )
        svg.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="27" fill="var(--gpu)"/>')
        svg.append(
            f'<text x="{x:.0f}" y="{y - 2:.0f}" text-anchor="middle" fill="#fff" '
            f'font-size="12" font-weight="600">gpu{gpu.get("gpu_index")}</text>'
        )
        svg.append(
            f'<text x="{x:.0f}" y="{y + 11:.0f}" text-anchor="middle" fill="#dbeafe" '
            f'font-size="8">{html_escape(domain)}</text>'
        )

    mode = node.get('scale_up_mode')
    label = f'Scale-up: XGMI mesh ({len(drawn)} links)'
    if mode:
        label += f' &middot; {html_escape(mode)}'
    svg.append(f'<text x="{cx:.0f}" y="{cy:.0f}" text-anchor="middle" fill="var(--ink2)" font-size="10">{label}</text>')
    svg.append('</svg>')
    return ''.join(svg)


def _topo_inter_node_section(cluster_dict):
    '''Render the inter-node (scale-out) fabric section.

    When LLDP resolved node->switch edges, list them grouped by switch. When it
    did not (the common case -- no lldpd daemon), show a muted note explaining
    the wiring needs LLDP or ARC, so the empty state is honest rather than silent.
    '''
    edges = cluster_dict.get('inter_node_edges', [])
    if not edges:
        return (
            '<h3>Inter-node fabric (scale-out)</h3>'
            '<div class="muted" style="border:1px dashed var(--line);border-radius:8px;'
            'padding:14px">Inter-node (scale-out) wiring was not discovered. Node '
            'uplinks and their rates are shown per node above; mapping them to '
            'switches needs LLDP on the nodes or ARC rack discovery.</div>'
        )
    by_switch = {}
    for edge in edges:
        switch = edge.get('via') or edge.get('dst', {}).get('name') or 'unknown'
        by_switch.setdefault(switch, []).append(edge)
    rows = []
    for switch in sorted(by_switch.keys()):
        links = ''.join(
            f'<div class="dimm" style="border-left-color:var(--nic)">'
            f'<b>{html_escape(e.get("src", {}).get("name", ""))}</b> '
            f'<span class="cl">{html_escape(e.get("src", {}).get("port", ""))}</span> '
            f'&rarr; {html_escape(e.get("dst", {}).get("port", ""))} '
            f'<span class="cl">{html_escape(e.get("speed") or "")} '
            f'{html_escape(e.get("status") or "")}</span></div>'
            for e in by_switch[switch]
        )
        rows.append(
            f'<details class="dimm-list" open><summary style="color:var(--nic)">'
            f'ToR switch {html_escape(switch)} &middot; {len(by_switch[switch])} links'
            f'</summary>{links}</details>'
        )
    return '<h3>Inter-node fabric (scale-out)</h3>' + ''.join(rows)


def build_topology_graph(filename, cluster_dict):
    '''Render a cluster topology dict to a standalone, self-contained HTML file.

    cluster_dict is ClusterTopology.to_dict() output. Each node gets a card with
    a CPU/NUMA panel (model, sockets, cores/threads, NUMA domains with cpulists,
    front-end NIC) and an SVG diagram: the CPU block, dashed PCIe host links from
    the CPU to every GPU, the GPU XGMI mesh (solid red XGMI / dashed grey PCIe
    between GPUs), and each GPU's backend NICs grouped by PCI domain. A node whose
    driver is down renders its CPU/NIC panel with a no-GPU note instead of a mesh.

    No external assets: the page is pure HTML+SVG so it travels in the pytest-html
    ZIP bundle and opens offline.
    '''
    log.info('Build HTML cluster topology graph: %s', filename)
    summary = cluster_dict.get('summary', {})

    cards = []
    for node in cluster_dict.get('nodes', []):
        cards.append(
            f'<div class="node-card"><div class="node-head">'
            f'<span>{html_escape(node.get("host", ""))}</span>'
            f'<span class="meta">{_topo_node_meta(node)}</span></div>'
            f'<div class="node-body">{_topo_cpu_panel(node)}{_topo_mesh_svg(node)}</div></div>'
        )

    scale_up_mode = summary.get('scale_up_mode') or 'n/a'
    summary_line = (
        'Nodes: {node_count} (reachable {reachable_node_count}, '
        'unreachable {unreachable_node_count}) &nbsp;|&nbsp; Racks: {rack_count} '
        '&nbsp;|&nbsp; GPUs: {total_gpus} &nbsp;|&nbsp; XGMI links: {xgmi_edge_count} '
        '&nbsp;|&nbsp; DIMMs: {total_dimms} &nbsp;|&nbsp; NVMe: {total_nvme} '
        '&nbsp;|&nbsp; Scale-up: {scale_up_mode} '
        '&nbsp;|&nbsp; Scale-out NICs: {scale_out_nic_count} '
        '&nbsp;|&nbsp; Switches: {switch_count}'
    ).format(
        node_count=summary.get('node_count', 0),
        reachable_node_count=summary.get('reachable_node_count', 0),
        unreachable_node_count=summary.get('unreachable_node_count', 0),
        rack_count=summary.get('rack_count', 0),
        total_gpus=summary.get('total_gpus', 0),
        xgmi_edge_count=summary.get('xgmi_edge_count', 0),
        total_dimms=summary.get('total_dimms', 0),
        total_nvme=summary.get('total_nvme', 0),
        scale_up_mode=html_escape(scale_up_mode),
        scale_out_nic_count=summary.get('scale_out_nic_count', 0),
        switch_count=summary.get('switch_count', 0),
    )

    inter_section = _topo_inter_node_section(cluster_dict)

    with open(filename, 'w') as fp:
        fp.write('<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8"/>\n')
        fp.write('<title>CVS Cluster Topology</title>\n')
        fp.write(f'<style>{_TOPO_CSS}</style>\n')
        fp.write('</head>\n<body>\n<div class="wrap">\n')
        fp.write('<h2>CVS Cluster Topology</h2>\n')
        fp.write(
            '<p class="sub">Per node: CPU/NUMA, the GPU XGMI mesh (scale-up), '
            'CPU&harr;GPU PCIe host links, backend NICs, DIMMs and NVMe. '
            'Inter-node (scale-out) wiring below.</p>\n'
        )
        fp.write(f'<div class="sum">{summary_line}</div>\n')
        fp.write(_topo_legend() + '\n')
        fp.write(''.join(cards))
        fp.write(inter_section)
        fp.write('<h3>Raw topology JSON</h3>\n<pre id="json-display"></pre>\n')
        fp.write(
            '<script>document.getElementById("json-display").textContent = '
            + 'JSON.stringify('
            + json.dumps(cluster_dict)
            + ', null, 2);</script>\n'
        )
        fp.write('</div>\n</body>\n</html>\n')
