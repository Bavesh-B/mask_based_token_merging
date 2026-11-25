"use strict";

const scenes = {
  pannuke: {
    name: "PanNuke",
    domain: "H&E histology",
    description: "Dense fields of nuclei make visually similar instances easy to fuse. The white outlines show how each method groups patch tokens.",
    width: 556,
    height: 550,
    iou: { algm: "0.47", tome: "0.45", maskmerge: "0.70" },
  },
  isaid: {
    name: "iSAID",
    domain: "Aerial imagery",
    description: "Aerial views put small objects and similar-looking surfaces close together. MaskMerge uses predicted instance identity to guide global token matches.",
    width: 556,
    height: 550,
    iou: { algm: "0.67", tome: "0.65", maskmerge: "0.70" },
  },
  cityscapes: {
    name: "Cityscapes",
    domain: "Street scenes",
    description: "Pedestrians cross close to one another while vehicles occupy the background. Compare the inferred masks and the regions over which tokens were merged.",
    width: 1099,
    height: 550,
    iou: { algm: "0.28", tome: "0.30", maskmerge: "0.61" },
  },
};

// Camera-ready Table 1. Rows are Dense EoMT, ToMe, ALGM, and MaskMerge.
const benchmarks = {
  "pannuke-s": { dataset: "PanNuke", scale: "S", rows: [[35.1, 95.1, 140.8], [27.1, 220.1, 76.1], [25.4, 224.2, 41.6], [29.5, 229.9, 35.3]] },
  "pannuke-b": { dataset: "PanNuke", scale: "B", rows: [[39.4, 49.1, 623], [29.5, 114.0, 369], [30.1, 121.5, 229], [32.1, 142.8, 153]] },
  "pannuke-l": { dataset: "PanNuke", scale: "L", rows: [[41.3, 26.4, 1654], [32.8, 49.8, 1093], [34.7, 55.2, 546], [35.2, 58.9, 482]] },
  "isaid-b": { dataset: "iSAID", scale: "B", rows: [[41.6, 83.9, 383], [31.5, 160.5, 207], [33.1, 168.8, 198], [36.2, 206.9, 182]] },
  "cityscapes-b": { dataset: "Cityscapes", scale: "B", rows: [[21.6, 39.5, 827], [10.6, 109.7, 330], [19.8, 107.4, 347], [19.8, 105.2, 298]] },
};

const methodNames = { algm: "ALGM", tome: "ToMe", input: "Ground truth" };
const byId = (id) => document.getElementById(id);
const displayNumber = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });

function setupComparison() {
  const datasetButtons = [...document.querySelectorAll("#dataset-controls button")];
  const method = byId("compare-method");
  const comparison = byId("comparison");
  const slider = byId("compare-slider");
  let activeDataset = "pannuke";

  function setSplit(value) {
    const amount = Math.max(0, Math.min(100, Math.round(value)));
    slider.value = amount;
    const baseline = methodNames[method.value];
    slider.setAttribute("aria-valuetext", `${amount}% ${baseline}, ${100 - amount}% MaskMerge`);
    comparison.style.setProperty("--split", `${amount}%`);
  }

  function setImage(id, filename, alt, scene) {
    const img = byId(id);
    img.width = scene.width;
    img.height = scene.height;
    img.alt = alt;
    img.src = `assets/${activeDataset}-${filename}.webp`;
  }

  function renderScene() {
    const scene = scenes[activeDataset];
    const baseline = methodNames[method.value];
    const ratio = scene.width / scene.height;
    comparison.style.setProperty("--scene-ratio", ratio);
    comparison.style.setProperty("--scene-width", `calc(var(--gallery-size) * ${ratio})`);
    setImage("comparison-maskmerge", "maskmerge", `MaskMerge masks and token boundaries on a ${scene.name} image`, scene);
    setImage("comparison-baseline", method.value, `${baseline} instance masks on the same ${scene.name} image`, scene);
    byId("compare-left-label").textContent = baseline;
    byId("scene-title").textContent = scene.name;
    byId("scene-domain").textContent = scene.domain;
    byId("scene-description").textContent = scene.description;
    byId("scene-baseline-name").textContent = baseline;
    byId("scene-baseline-iou").textContent = scene.iou[method.value] ?? "Reference";
    byId("scene-maskmerge-iou").textContent = scene.iou.maskmerge;
    setSplit(slider.value);
  }

  for (const button of datasetButtons) {
    button.addEventListener("click", () => {
      activeDataset = button.dataset.dataset;
      for (const item of datasetButtons) {
        item.setAttribute("aria-pressed", String(item === button));
      }
      renderScene();
    });
  }
  method.addEventListener("change", renderScene);
  slider.addEventListener("input", () => setSplit(slider.value));

  function splitAtPointer(event) {
    const bounds = comparison.getBoundingClientRect();
    setSplit((event.clientX - bounds.left) / bounds.width * 100);
  }
  comparison.dataset.interactive = "";
  comparison.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || !event.isPrimary) return;
    comparison.setPointerCapture(event.pointerId);
    splitAtPointer(event);
  });
  comparison.addEventListener("pointermove", (event) => {
    if (comparison.hasPointerCapture(event.pointerId)) splitAtPointer(event);
  });
  renderScene();
}

function setupBenchmarks() {
  const selector = byId("benchmark-select");
  const names = ["Dense EoMT", "ToMe", "ALGM", "MaskMerge"];

  function renderBenchmark() {
    const { dataset, scale, rows } = benchmarks[selector.value];
    const title = `${dataset} with ViT-${scale}`;
    byId("benchmark-caption").textContent = `${dataset} / DINOv2 ViT-${scale}`;
    byId("benchmark-speed").textContent = dataset === "PanNuke" ? "Images/s" : "Tiles/s";
    const tableRows = rows.map(([map, speed, flops], index) => {
      const row = document.createElement("tr");
      const name = document.createElement("th");
      name.scope = "row";
      name.textContent = names[index];
      row.append(name);
      for (const value of [map.toFixed(1), speed.toFixed(1), displayNumber.format(flops)]) {
        row.insertCell().textContent = value;
      }
      row.classList.toggle("ours", index === 3);
      return row;
    });
    byId("benchmark-rows").replaceChildren(...tableRows);
    const [algmAP, , algmFlops] = rows[2];
    const [maskmergeAP, , maskmergeFlops] = rows[3];
    const accuracy = maskmergeAP === algmAP
      ? `matches ALGM at ${maskmergeAP.toFixed(1)} mAP`
      : `gains ${(maskmergeAP - algmAP).toFixed(1)} mAP over ALGM`;
    const unit = dataset === "PanNuke" ? "image" : "tile";
    byId("benchmark-summary").textContent = `On ${title}, MaskMerge ${accuracy} and uses ${displayNumber.format(algmFlops - maskmergeFlops)} fewer GFLOPs per ${unit}.`;
  }

  selector.addEventListener("change", renderBenchmark);
}

function setupTheme() {
  const button = byId("theme-toggle");
  const systemTheme = window.matchMedia("(prefers-color-scheme: dark)");
  let preference;
  try { preference = localStorage.getItem("maskmerge-theme"); } catch {}

  function applyTheme() {
    const dark = preference ? preference === "dark" : systemTheme.matches;
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    button.setAttribute("aria-pressed", String(dark));
  }

  button.addEventListener("click", () => {
    preference = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    try { localStorage.setItem("maskmerge-theme", preference); } catch {}
    applyTheme();
  });
  systemTheme.addEventListener("change", applyTheme);
  applyTheme();
  button.hidden = false;
}

function setupCitation() {
  const button = byId("copy-citation");
  const code = byId("bibtex");
  const status = byId("copy-status");
  let resetTitle;
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(code.textContent);
      status.textContent = "BibTeX copied.";
      button.title = "Copied BibTeX";
      button.classList.add("copied");
      clearTimeout(resetTitle);
      resetTitle = setTimeout(() => {
        button.title = "Copy BibTeX";
        button.classList.remove("copied");
        status.textContent = "";
      }, 1800);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = "BibTeX selected.";
    }
  });
  button.hidden = false;
}

setupComparison();
setupBenchmarks();
setupTheme();
setupCitation();
for (const control of document.querySelectorAll("[data-js]")) control.hidden = false;
