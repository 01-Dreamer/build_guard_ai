TEST_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BuildGuard Lab</title>
  <style>
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #18212f;
      --muted: #657386;
      --line: #d9e1ea;
      --accent: #0f766e;
      --danger: #b42318;
      --ok: #087443;
      --soft: #eef3f7;
      --shadow: 0 12px 28px rgba(24, 33, 47, 0.12);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      min-height: 76px;
      padding: 18px clamp(16px, 4vw, 44px);
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }

    h1 {
      margin: 0;
      font-size: clamp(22px, 3vw, 34px);
      line-height: 1.05;
      letter-spacing: 0;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      min-height: 34px;
      padding: 0 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      background: #f8fafc;
      font-size: 14px;
      white-space: nowrap;
    }

    main {
      display: grid;
      grid-template-columns: minmax(300px, 390px) minmax(0, 1fr);
      gap: 18px;
      max-width: 1440px;
      margin: 0 auto;
      padding: 18px clamp(16px, 4vw, 44px) 36px;
    }

    section {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }

    .stack {
      display: grid;
      gap: 14px;
    }

    .panel {
      padding: 16px;
    }

    .panel h2,
    .toolbar h2 {
      margin: 0 0 12px;
      font-size: 16px;
      letter-spacing: 0;
    }

    form {
      display: grid;
      gap: 12px;
    }

    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }

    input[type="text"],
    input[type="file"] {
      width: 100%;
      min-height: 40px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      color: var(--ink);
      font: inherit;
    }

    input[type="file"] {
      padding: 7px;
    }

    button {
      min-height: 40px;
      border: 0;
      border-radius: 6px;
      padding: 0 14px;
      background: var(--accent);
      color: #ffffff;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }

    button.secondary {
      color: var(--ink);
      background: #e8edf4;
    }

    button.danger {
      background: var(--danger);
    }

    button:disabled {
      cursor: wait;
      opacity: 0.68;
    }

    .button-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    .message {
      min-height: 22px;
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }

    .message.ok { color: var(--ok); }
    .message.err { color: var(--danger); }

    .summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }

    .metric {
      min-height: 70px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: var(--soft);
    }

    .metric strong {
      display: block;
      font-size: 24px;
      line-height: 1;
      margin-bottom: 8px;
    }

    .metric span {
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }

    .workspace {
      min-height: 680px;
      display: grid;
      grid-template-rows: auto minmax(360px, 1fr) auto;
      overflow: hidden;
    }

    .toolbar {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }

    .toolbar form {
      width: min(100%, 560px);
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: end;
    }

    .canvas-wrap {
      display: grid;
      place-items: center;
      min-height: 360px;
      padding: 16px;
      background:
        linear-gradient(90deg, rgba(24, 33, 47, 0.05) 1px, transparent 1px),
        linear-gradient(rgba(24, 33, 47, 0.05) 1px, transparent 1px);
      background-size: 24px 24px;
    }

    .empty {
      display: grid;
      place-items: center;
      min-height: 360px;
      width: 100%;
      border: 1px dashed #b7c2d2;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.74);
      color: var(--muted);
      text-align: center;
      font-weight: 800;
    }

    canvas {
      max-width: 100%;
      max-height: 72vh;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
    }

    .results {
      padding: 14px 16px 16px;
      border-top: 1px solid var(--line);
      background: #ffffff;
    }

    .result-list {
      display: grid;
      gap: 10px;
    }

    .result-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #f8fafc;
    }

    .result-item strong {
      display: block;
      margin-bottom: 5px;
      font-size: 15px;
    }

    .result-item span {
      display: block;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }

    pre {
      min-height: 220px;
      max-height: 460px;
      overflow: auto;
      margin: 0;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #111827;
      color: #d1fae5;
      font-size: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
    }

    @media (max-width: 900px) {
      header {
        align-items: flex-start;
        flex-direction: column;
      }

      main {
        grid-template-columns: 1fr;
      }

      .toolbar {
        align-items: stretch;
        flex-direction: column;
      }

      .toolbar form {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>BuildGuard Lab</h1>
    <div id="serviceStatus" class="status-pill">Checking service</div>
  </header>

  <main>
    <div class="stack">
      <section class="panel">
        <h2>Register Face</h2>
        <form id="registerForm">
          <label>ID
            <input id="registerId" name="id" type="text" autocomplete="off" required>
          </label>
          <label>Name
            <input id="registerName" name="name" type="text" autocomplete="off">
          </label>
          <label>Face Image
            <input id="registerImage" name="img" type="file" accept="image/*" required>
          </label>
          <button type="submit">Register</button>
          <div id="registerMessage" class="message"></div>
        </form>
      </section>

      <section class="panel">
        <h2>Delete Face</h2>
        <form id="deleteForm">
          <label>ID
            <input id="deleteId" type="text" autocomplete="off" required>
          </label>
          <div class="button-row">
            <button class="danger" type="submit">Delete</button>
            <button class="secondary" id="refreshStatus" type="button">Refresh</button>
          </div>
          <div id="deleteMessage" class="message"></div>
        </form>
      </section>

      <section class="panel">
        <h2>Result</h2>
        <div class="summary">
          <div class="metric">
            <strong id="violationCount">0</strong>
            <span>violations</span>
          </div>
          <div class="metric">
            <strong id="personCount">0</strong>
            <span>persons</span>
          </div>
        </div>
        <pre id="responseBox">{}</pre>
      </section>
    </div>

    <section class="workspace">
      <div class="toolbar">
        <div>
          <h2>PPE Detection</h2>
          <div id="detectMessage" class="message"></div>
        </div>
        <form id="detectForm">
          <label>Site Image
            <input id="detectImage" name="img" type="file" accept="image/*" required>
          </label>
          <button type="submit">Detect</button>
        </form>
      </div>

      <div class="canvas-wrap">
        <div id="emptyState" class="empty">Select a construction-site image</div>
        <canvas id="imageCanvas" hidden></canvas>
      </div>

      <div class="results">
        <div id="resultList" class="result-list"></div>
      </div>
    </section>
  </main>

  <script>
    const serviceStatus = document.getElementById("serviceStatus");
    const responseBox = document.getElementById("responseBox");
    const registerForm = document.getElementById("registerForm");
    const deleteForm = document.getElementById("deleteForm");
    const detectForm = document.getElementById("detectForm");
    const registerMessage = document.getElementById("registerMessage");
    const deleteMessage = document.getElementById("deleteMessage");
    const detectMessage = document.getElementById("detectMessage");
    const refreshStatus = document.getElementById("refreshStatus");
    const fileInput = document.getElementById("detectImage");
    const canvas = document.getElementById("imageCanvas");
    const ctx = canvas.getContext("2d");
    const emptyState = document.getElementById("emptyState");
    const resultList = document.getElementById("resultList");
    const violationCount = document.getElementById("violationCount");
    const personCount = document.getElementById("personCount");

    function setMessage(el, text, kind) {
      el.textContent = text || "";
      el.className = kind ? `message ${kind}` : "message";
    }

    function setBusy(form, busy) {
      for (const button of form.querySelectorAll("button")) {
        button.disabled = busy;
      }
    }

    function showResponse(payload) {
      responseBox.textContent = JSON.stringify(payload, null, 2);
    }

    async function requestJson(url, options) {
      const response = await fetch(url, options);
      const contentType = response.headers.get("content-type") || "";
      const payload = contentType.includes("application/json")
        ? await response.json()
        : { detail: await response.text() };
      if (!response.ok) {
        throw Object.assign(new Error(payload.detail || response.statusText), {
          payload,
          status: response.status,
        });
      }
      return payload;
    }

    async function loadHealth() {
      try {
        const payload = await requestJson("/health");
        serviceStatus.textContent = `Ready · ${payload.registered_faces} registered`;
      } catch (error) {
        serviceStatus.textContent = "Service unavailable";
      }
    }

    function getFile(input) {
      return input.files && input.files.length ? input.files[0] : null;
    }

    function buildImage(file) {
      return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error("image preview failed"));
        image.src = URL.createObjectURL(file);
      });
    }

    function drawBase(image) {
      const maxWidth = 1280;
      const scale = Math.min(1, maxWidth / image.naturalWidth);
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      canvas.hidden = false;
      emptyState.hidden = true;
      return scale;
    }

    function drawBox(bbox, scale, color, label) {
      const [x1, y1, x2, y2] = bbox.map((value) => value * scale);
      ctx.save();
      ctx.lineWidth = Math.max(2, Math.round(canvas.width / 430));
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      if (label) {
        ctx.font = "800 16px system-ui, sans-serif";
        ctx.textBaseline = "top";
        const width = ctx.measureText(label).width;
        const labelX = Math.max(0, Math.min(x1, canvas.width - width - 14));
        const labelY = Math.max(0, y1 - 28);
        ctx.fillStyle = color;
        ctx.fillRect(labelX, labelY, width + 14, 24);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, labelX + 7, labelY + 3);
      }
      ctx.restore();
    }

    function drawDetections(image, payload) {
      const scale = drawBase(image);
      const personsByBox = new Map((payload.persons || []).map((person) => [JSON.stringify(person.bbox), person]));
      for (const violation of payload.violations || []) {
        const person = personsByBox.get(JSON.stringify(violation.bbox)) || violation;
        const identity = person.identity || violation.identity || {};
        const name = identity.name || identity.id || "unknown";
        drawBox(violation.bbox, scale, "#b42318", name);
      }
    }

    function renderResults(payload) {
      resultList.innerHTML = "";
      violationCount.textContent = String(payload.count || 0);
      personCount.textContent = String((payload.persons || []).length);

      const violations = payload.violations || [];
      if (!violations.length) {
        const item = document.createElement("div");
        item.className = "result-item";
        const title = document.createElement("strong");
        const text = document.createElement("span");
        title.textContent = "No PPE violations";
        text.textContent = "No person missing required safety equipment.";
        item.append(title, text);
        resultList.appendChild(item);
        return;
      }

      const personsByBox = new Map((payload.persons || []).map((person) => [JSON.stringify(person.bbox), person]));
      for (const [index, violation] of violations.entries()) {
        const person = personsByBox.get(JSON.stringify(violation.bbox)) || violation;
        const item = document.createElement("div");
        item.className = "result-item";
        const title = document.createElement("strong");
        const identity = document.createElement("span");
        const helmet = document.createElement("span");
        const vest = document.createElement("span");
        const score = document.createElement("span");
        const missing = document.createElement("span");
        const personIdentity = person.identity || {};
        title.textContent = `#${index + 1} ${personIdentity.name || personIdentity.id || "unknown"}`;
        identity.textContent = `identity: ${personIdentity.recognized ? "recognized" : "unknown"}`;
        missing.textContent = `missing: ${(violation.missing || []).join(", ") || "unknown"}`;
        helmet.textContent = `helmet: ${violation.helmet_status || person.helmet_status || "unknown"}`;
        vest.textContent = `vest: ${violation.vest_status || person.vest_status || "unknown"}`;
        score.textContent = `person confidence: ${Number(person.confidence || 0).toFixed(4)}`;
        item.append(title, identity, missing, helmet, vest, score);
        resultList.appendChild(item);
      }
    }

    registerForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      setBusy(registerForm, true);
      setMessage(registerMessage, "Registering", "");
      try {
        const formData = new FormData();
        formData.append("id", document.getElementById("registerId").value.trim());
        formData.append("name", document.getElementById("registerName").value.trim());
        formData.append("img", getFile(document.getElementById("registerImage")));
        const payload = await requestJson("/faces/register", { method: "POST", body: formData });
        setMessage(registerMessage, payload.replaced ? "Face replaced" : "Face registered", "ok");
        showResponse(payload);
        await loadHealth();
      } catch (error) {
        setMessage(registerMessage, error.message, "err");
        showResponse(error.payload || { detail: error.message });
      } finally {
        setBusy(registerForm, false);
      }
    });

    deleteForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      setBusy(deleteForm, true);
      setMessage(deleteMessage, "Deleting", "");
      try {
        const id = encodeURIComponent(document.getElementById("deleteId").value.trim());
        const payload = await requestJson(`/faces/${id}`, { method: "DELETE" });
        setMessage(deleteMessage, payload.deleted ? "Face deleted" : "ID not found", payload.deleted ? "ok" : "");
        showResponse(payload);
        await loadHealth();
      } catch (error) {
        setMessage(deleteMessage, error.message, "err");
        showResponse(error.payload || { detail: error.message });
      } finally {
        setBusy(deleteForm, false);
      }
    });

    detectForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      setBusy(detectForm, true);
      setMessage(detectMessage, "Detecting", "");
      try {
        const file = getFile(fileInput);
        const image = await buildImage(file);
        const formData = new FormData();
        formData.append("img", file);
        const payload = await requestJson("/safety/detect", { method: "POST", body: formData });
        drawDetections(image, payload);
        renderResults(payload);
        setMessage(detectMessage, `${(payload.persons || []).length} person(s), ${payload.count} PPE violation(s)`, payload.count ? "err" : "ok");
        showResponse(payload);
      } catch (error) {
        setMessage(detectMessage, error.message, "err");
        showResponse(error.payload || { detail: error.message });
      } finally {
        setBusy(detectForm, false);
      }
    });

    refreshStatus.addEventListener("click", loadHealth);
    loadHealth();
  </script>
</body>
</html>
"""
