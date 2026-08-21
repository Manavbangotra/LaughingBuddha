/* =============================================================================
   book.js — reading-shell behaviour: theme toggle, mobile nav, scroll-spy,
   and full-text search over a prebuilt index.

   No dependencies. The search index is a single JSON file produced by the
   build; scoring is a simple field-weighted prefix match, which is more than
   enough for a few hundred documents and keeps the site fully offline.
   ========================================================================== */

(function () {
  "use strict";

  /* --- theme -------------------------------------------------------------- */

  var root = document.documentElement;
  var toggle = document.getElementById("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var explicit = root.getAttribute("data-theme");
      var dark = explicit
        ? explicit === "dark"
        : window.matchMedia("(prefers-color-scheme: dark)").matches;
      var next = dark ? "light" : "dark";
      root.setAttribute("data-theme", next);
      localStorage.setItem("book-theme", next);
    });
  }

  /* --- mobile navigation --------------------------------------------------- */

  var menu = document.getElementById("menu-toggle");
  if (menu) {
    menu.addEventListener("click", function () {
      document.body.classList.toggle("nav-open");
    });
    document.addEventListener("click", function (e) {
      if (!document.body.classList.contains("nav-open")) return;
      var side = document.querySelector(".sidebar");
      if (side && !side.contains(e.target) && e.target !== menu) {
        document.body.classList.remove("nav-open");
      }
    });
  }

  // Keep the active chapter visible in a sidebar that is thousands of rows long.
  var current = document.querySelector(".sidebar a.current");
  if (current) {
    var side = document.querySelector(".sidebar");
    var top = current.offsetTop - side.clientHeight / 2;
    if (top > 0) side.scrollTop = top;
  }

  /* --- scroll-spy on the right rail ---------------------------------------- */

  var railLinks = Array.prototype.slice.call(
    document.querySelectorAll(".rail a[href^='#']")
  );
  if (railLinks.length && "IntersectionObserver" in window) {
    var byId = {};
    var targets = [];
    railLinks.forEach(function (a) {
      var el = document.getElementById(decodeURIComponent(a.hash.slice(1)));
      if (el) {
        byId[el.id] = a;
        targets.push(el);
      }
    });
    var visible = new Set();
    var obs = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (en) {
          if (en.isIntersecting) visible.add(en.target.id);
          else visible.delete(en.target.id);
        });
        var first = targets.filter(function (t) { return visible.has(t.id); })[0];
        railLinks.forEach(function (a) { a.classList.remove("active"); });
        if (first && byId[first.id]) byId[first.id].classList.add("active");
      },
      { rootMargin: "-72px 0px -70% 0px", threshold: 0 }
    );
    targets.forEach(function (t) { obs.observe(t); });
  }

  /* --- search --------------------------------------------------------------- */

  var input = document.getElementById("search");
  var panel = document.getElementById("search-results");
  if (!input || !panel) return;

  var index = null;
  var loading = false;
  var selection = -1;

  function load() {
    if (index || loading) return Promise.resolve(index);
    loading = true;
    return fetch("search-index.json")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        index = data.map(function (d) {
          return {
            u: d.u, t: d.t, l: d.l, h: d.h || [],
            b: d.b || "",
            lc: (d.t + " " + (d.h || []).join(" ") + " " + (d.b || "")).toLowerCase(),
            tlc: d.t.toLowerCase(),
            hlc: (d.h || []).join(" ").toLowerCase()
          };
        });
        loading = false;
        return index;
      })
      .catch(function () { loading = false; return null; });
  }

  input.addEventListener("focus", load);

  document.addEventListener("keydown", function (e) {
    if (e.key === "/" && document.activeElement !== input) {
      e.preventDefault();
      input.focus();
      input.select();
    } else if (e.key === "Escape") {
      close();
      input.blur();
    }
  });

  function close() {
    panel.classList.remove("open");
    panel.innerHTML = "";
    selection = -1;
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  function snippet(doc, terms) {
    var lower = doc.b.toLowerCase();
    var at = -1;
    for (var i = 0; i < terms.length && at < 0; i++) at = lower.indexOf(terms[i]);
    if (at < 0) at = 0;
    var start = Math.max(0, at - 60);
    var text = doc.b.slice(start, start + 190);
    var out = escapeHtml((start > 0 ? "…" : "") + text + "…");
    terms.forEach(function (t) {
      if (t.length < 2) return;
      out = out.replace(
        new RegExp("(" + t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig"),
        "<mark>$1</mark>"
      );
    });
    return out;
  }

  function search(q) {
    var terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length || !index) return [];
    var hits = [];
    index.forEach(function (d) {
      var score = 0;
      var ok = true;
      terms.forEach(function (t) {
        if (d.lc.indexOf(t) < 0) { ok = false; return; }
        if (d.tlc.indexOf(t) >= 0) score += 40;
        if (d.hlc.indexOf(t) >= 0) score += 12;
        var n = d.b.toLowerCase().split(t).length - 1;
        score += Math.min(n, 12);
      });
      if (ok) hits.push({ d: d, s: score });
    });
    hits.sort(function (a, b) { return b.s - a.s; });
    return hits.slice(0, 12).map(function (h) { return h.d; });
  }

  function render(results, terms) {
    if (!results.length) {
      panel.innerHTML = '<div class="empty">No matches.</div>';
      panel.classList.add("open");
      return;
    }
    panel.innerHTML = results
      .map(function (d) {
        return (
          '<a href="' + d.u + '" role="option">' +
          '<div class="r-label">' + escapeHtml(d.l || "") + "</div>" +
          "<div>" + escapeHtml(d.t) + "</div>" +
          '<div class="r-snippet">' + snippet(d, terms) + "</div>" +
          "</a>"
        );
      })
      .join("");
    panel.classList.add("open");
    selection = -1;
  }

  var timer = null;
  input.addEventListener("input", function () {
    clearTimeout(timer);
    var q = input.value.trim();
    if (q.length < 2) { close(); return; }
    timer = setTimeout(function () {
      load().then(function () {
        render(search(q), q.toLowerCase().split(/\s+/).filter(Boolean));
      });
    }, 110);
  });

  input.addEventListener("keydown", function (e) {
    var items = panel.querySelectorAll("a");
    if (!items.length) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      items[Math.max(selection, 0)].classList.remove("sel");
      selection += e.key === "ArrowDown" ? 1 : -1;
      if (selection < 0) selection = items.length - 1;
      if (selection >= items.length) selection = 0;
      items[selection].classList.add("sel");
      items[selection].scrollIntoView({ block: "nearest" });
    } else if (e.key === "Enter" && selection >= 0) {
      e.preventDefault();
      window.location.href = items[selection].getAttribute("href");
    }
  });

  document.addEventListener("click", function (e) {
    if (!panel.contains(e.target) && e.target !== input) close();
  });
})();
