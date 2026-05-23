(function () {
  let host = null;
  let titleEl = null;
  let bodyEl = null;
  let closeBtn = null;

  function ensureHost() {
    if (host) return host;
    host = document.createElement("div");
    host.className = "op-modal-host";
    host.innerHTML = [
      '<div class="op-modal-backdrop" data-modal-close></div>',
      '<section class="op-modal" role="dialog" aria-modal="true" aria-labelledby="opModalTitle">',
      '  <header class="op-modal-header">',
      '    <h2 id="opModalTitle"></h2>',
      '    <button class="ghost op-modal-close" type="button" aria-label="Close">Close</button>',
      "  </header>",
      '  <div class="op-modal-body"></div>',
      "</section>",
    ].join("");
    document.body.appendChild(host);
    titleEl = host.querySelector("#opModalTitle");
    bodyEl = host.querySelector(".op-modal-body");
    closeBtn = host.querySelector(".op-modal-close");
    closeBtn.addEventListener("click", close);
    host.querySelector("[data-modal-close]").addEventListener("click", close);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && host.classList.contains("open")) close();
    });
    return host;
  }

  function open(options) {
    const opts = options || {};
    ensureHost();
    titleEl.textContent = opts.title || "Tool";
    bodyEl.replaceChildren();
    if (opts.className) {
      bodyEl.className = `op-modal-body ${opts.className}`;
    } else {
      bodyEl.className = "op-modal-body";
    }
    if (opts.content instanceof Node) {
      bodyEl.appendChild(opts.content);
    } else if (typeof opts.html === "string") {
      bodyEl.innerHTML = opts.html;
    }
    host.classList.add("open");
    document.body.classList.add("op-modal-open");
    if (typeof opts.onOpen === "function") opts.onOpen(bodyEl);
    return bodyEl;
  }

  function close() {
    if (!host) return;
    host.classList.remove("open");
    document.body.classList.remove("op-modal-open");
    if (bodyEl) bodyEl.replaceChildren();
  }

  window.operatorModal = { open, close };
})();
