/* Small helpers shared by the dashboard: data loading, chart styling and an
   in-browser scorer for the gradient boosted trees exported by the notebook. */
const DS = (() => {
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const series = () => [1, 2, 3, 4, 5, 6, 7, 8].map((i) => css(`--series-${i}`));

  const load = async (path) => {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`Could not load ${path}`);
    return res.json();
  };

  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  /* model.trees[k] holds flat arrays: f (feature index, -1 for a leaf),
     t (threshold), y / n / m (child when x < t, x >= t, x missing), v (leaf value).
     XGBoost compares 32-bit floats, so inputs are rounded the same way here. */
  const margin = (model, x) => {
    let total = model.base;
    for (const tree of model.trees) {
      let i = 0;
      while (tree.f[i] !== -1) {
        const value = x[tree.f[i]];
        if (value === null || value === undefined || Number.isNaN(value)) i = tree.m[i];
        else i = Math.fround(value) < tree.t[i] ? tree.y[i] : tree.n[i];
      }
      total += tree.v[i];
    }
    return total;
  };

  const predict = (model, x) => {
    const z = margin(model, x);
    return model.objective === "logistic" ? sigmoid(z) : z;
  };

  /* Build a feature vector from a plain object keyed by feature name. */
  const vector = (model, values) => model.features.map((name) => (name in values ? values[name] : 0));

  const layout = (extra = {}) => {
    const base = {
      font: { family: css("--font"), size: 13, color: css("--text-secondary") },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      margin: { l: 56, r: 16, t: 8, b: 44 },
      colorway: series(),
      hoverlabel: {
        bgcolor: css("--surface-2"),
        bordercolor: css("--border"),
        font: { family: css("--font"), size: 13, color: css("--text-primary") },
      },
      legend: { orientation: "h", y: 1.12, x: 0, font: { color: css("--text-secondary") } },
      xaxis: { gridcolor: css("--grid"), linecolor: css("--border"), zeroline: false, automargin: true },
      yaxis: { gridcolor: css("--grid"), linecolor: css("--border"), zeroline: false, automargin: true },
    };
    const merged = { ...base, ...extra };
    merged.xaxis = { ...base.xaxis, ...(extra.xaxis || {}) };
    merged.yaxis = { ...base.yaxis, ...(extra.yaxis || {}) };
    return merged;
  };

  const tableFor = (traces) => {
    const usable = traces.filter((t) => Array.isArray(t.x) && Array.isArray(t.y) && t.x.length <= 400);
    if (!usable.length) return null;
    const details = document.createElement("details");
    details.className = "table-view";
    const summary = document.createElement("summary");
    summary.textContent = "View as table";
    details.appendChild(summary);
    const wrap = document.createElement("div");
    wrap.className = "table-wrap";
    const table = document.createElement("table");
    const head = table.insertRow();
    ["Series", "x", "y"].forEach((h) => {
      const th = document.createElement("th");
      th.textContent = h;
      head.appendChild(th);
    });
    usable.forEach((t, k) => {
      t.x.forEach((xv, i) => {
        const row = table.insertRow();
        row.insertCell().textContent = t.name || `Series ${k + 1}`;
        row.insertCell().textContent = xv;
        const yv = t.y[i];
        row.insertCell().textContent = typeof yv === "number" ? fmt(yv, 3) : yv;
      });
    });
    wrap.appendChild(table);
    details.appendChild(wrap);
    return details;
  };

  const plot = (id, traces, extra = {}, opts = {}) => {
    const el = document.getElementById(id);
    Plotly.react(el, traces, layout(extra), { displayModeBar: false, responsive: true });
    if (opts.table !== false) {
      const old = el.parentElement.querySelector(".table-view");
      if (old) old.remove();
      const t = tableFor(traces);
      if (t) el.after(t);
    }
  };

  const fmt = (v, digits = 1) =>
    v === null || v === undefined || Number.isNaN(v)
      ? "n/a"
      : Number(v).toLocaleString("en-US", { maximumFractionDigits: digits });

  const pct = (v, digits = 1) => `${(100 * v).toFixed(digits)}%`;

  const tiles = (id, items) => {
    const el = document.getElementById(id);
    el.innerHTML = "";
    items.forEach(({ label, value, note }) => {
      const div = document.createElement("div");
      div.className = "tile";
      div.innerHTML = `<div class="tile-label"></div><div class="tile-value"></div><div class="tile-note"></div>`;
      div.children[0].textContent = label;
      div.children[1].textContent = value;
      div.children[2].textContent = note || "";
      el.appendChild(div);
    });
  };

  const fill = (select, options, selected) => {
    select.innerHTML = "";
    options.forEach((o) => {
      const opt = document.createElement("option");
      const [value, label] = Array.isArray(o) ? o : [o, o];
      opt.value = value;
      opt.textContent = label;
      if (String(value) === String(selected)) opt.selected = true;
      select.appendChild(opt);
    });
  };

  return { css, series, load, sigmoid, margin, predict, vector, layout, plot, fmt, pct, tiles, fill };
})();
