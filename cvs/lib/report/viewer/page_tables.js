function pageLongTables(root) {
  var pageSize = __TABLE_PAGE_SIZE__;
  var scope = root && root.querySelectorAll ? root : document;
  var seq = 0;
  Array.prototype.forEach.call(scope.querySelectorAll("table"), function(table) {
    var body = table.tBodies[0];
    if (!body) {
      var rows = Array.prototype.slice.call(table.rows);
      if (!rows.length) return;
      var start = table.tHead ? table.tHead.rows.length : (rows[0].querySelector("th") ? 1 : 0);
      if (!table.tHead && start === 1) table.createTHead().appendChild(rows[0]);
      body = table.createTBody();
      for (var i = start; i < rows.length; i++) body.appendChild(rows[i]);
    }
    var data = body.rows;
    var next = table.nextElementSibling;
    if (next && next.classList.contains("rd-pager")) next.remove();
    if (data.length <= pageSize) {
      for (var j = 0; j < data.length; j++) data[j].hidden = false;
      return;
    }
    if (!table.id) {
      seq += 1;
      table.id = "rd-paged-" + seq;
    }
    var pager = document.createElement("div");
    pager.className = "rd-pager";
    pager.innerHTML = "<button type='button' data-dir='-1'>Previous</button>"
      + "<span class='rd-pager-count'></span><button type='button' data-dir='1'>Next</button>";
    table.insertAdjacentElement("afterend", pager);
    var page = 0;
    var label = pager.querySelector(".rd-pager-count");
    function show() {
      var startRow = page * pageSize;
      var endRow = Math.min(startRow + pageSize, data.length);
      for (var n = 0; n < data.length; n++) data[n].hidden = n < startRow || n >= endRow;
      label.textContent = (startRow + 1) + "\u2013" + endRow + " of " + data.length;
    }
    pager.addEventListener("click", function(event) {
      var button = event.target.closest("button");
      if (!button) return;
      page = Math.max(0, Math.min(Math.ceil(data.length / pageSize) - 1, page + Number(button.getAttribute("data-dir"))));
      show();
    });
    show();
  });
}
