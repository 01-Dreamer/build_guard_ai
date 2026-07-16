TEST_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BuildGuard Face Lab</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #607086;
      --line: #d9e0ea;
      --accent: #0f766e;
      --accent-ink: #ffffff;
      --warn: #b42318;
      --ok: #087443;
      --shadow: 0 14px 34px rgba(31, 42, 68, 0.12);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
    }

    header {
      min-height: 76px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      padding: 18px clamp(16px, 4vw, 44px);
      border-bottom: 1px solid var(--line);
      background: #ffffff;
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
      grid-template-columns: minmax(300px, 380px) minmax(0, 1fr);
      gap: 18px;
      padding: 18px clamp(16px, 4vw, 44px) 36px;
      max-width: 1440px;
      margin: 0 auto;
    }

    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .stack {
      display: grid;
      gap: 14px;
    }

    .panel {
      padding: 16px;
    }

    .panel h2 {
      margin: 0 0 14px;
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
      font-weight: 600;
    }

    input[type="text"],
    input[type="file"] {
      width: 100%;
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      color: var(--ink);
      background: #ffffff;
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
      color: var(--accent-ink);
      background: var(--accent);
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }

    button.secondary {
      color: var(--ink);
      background: #e8edf4;
    }

    button.danger {
      background: var(--warn);
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

    .workspace {
      min-height: 620px;
      display: grid;
      grid-template-rows: auto minmax(320px, 1fr) auto;
      overflow: hidden;
    }

    .toolbar {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 14px;
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }

    .identify-form {
      width: min(100%, 520px);
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: end;
    }

    .canvas-wrap {
      display: grid;
      place-items: center;
      min-height: 320px;
      padding: 16px;
      background:
        linear-gradient(90deg, rgba(23, 32, 42, 0.05) 1px, transparent 1px),
        linear-gradient(rgba(23, 32, 42, 0.05) 1px, transparent 1px);
      background-size: 24px 24px;
    }

    canvas {
      max-width: 100%;
      max-height: 68vh;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
    }

    .empty {
      display: grid;
      place-items: center;
      width: 100%;
      min-height: 320px;
      border: 1px dashed #b7c2d2;
      border-radius: 8px;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.74);
      text-align: center;
      font-weight: 700;
    }

    .results {
      display: grid;
      gap: 10px;
      padding: 14px 16px 16px;
      border-top: 1px solid var(--line);
      background: #ffffff;
    }

    .result-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }

    .result-item {
      min-height: 78px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #f8fafc;
    }

    .result-item strong {
      display: block;
      margin-bottom: 4px;
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
      min-height: 160px;
      max-height: 360px;
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

    .message {
      min-height: 24px;
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }

    .message.ok {
      color: var(--ok);
    }

    .message.err {
      color: var(--warn);
    }

    @media (max-width: 860px) {
      header {
        align-items: flex-start;
        flex-direction: column;
      }

      main {
        grid-template-columns: 1fr;
      }

      .identify-form {
        grid-template-columns: 1fr;
      }

      .workspace {
        min-height: 520px;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>BuildGuard Face Lab</h1>
    <div id="serviceStatus" class="status-pill">Checking service</div>
  </header>

  <main>
    <div class="stack">
      <section class="panel">
        <h2>Register</h2>
        <form id="registerForm">
          <label>ID
            <input id="registerId" name="id" type="text" autocomplete="off" required>
          </label>
          <label>Name
            <input id="registerName" name="name" type="text" autocomplete="off">
          </label>
          <label>Image
            <input id="registerImage" name="img" type="file" accept="image/*" required>
          </label>
          <button type="submit">Register Face</button>
          <div id="registerMessage" class="message"></div>
        </form>
      </section>

      <section class="panel">
        <h2>Delete</h2>
        <form id="deleteForm">
          <label>ID
            <input id="deleteId" type="text" autocomplete="off" required>
          </label>
          <div class="button-row">
            <button class="danger" type="submit">Delete Face</button>
            <button class="secondary" id="refreshStatus" type="button">Refresh</button>
          </div>
          <div id="deleteMessage" class="message"></div>
        </form>
      </section>

      <section class="panel">
        <h2>Response</h2>
        <pre id="responseBox">{}</pre>
      </section>
    </div>

    <section class="workspace">
      <div class="toolbar">
        <div>
          <h2>Identify</h2>
          <div id="identifyMessage" class="message"></div>
        </div>
        <form id="identifyForm" class="identify-form">
          <label>Image
            <input id="identifyImage" name="img" type="file" accept="image/*" required>
          </label>
          <button type="submit">Identify</button>
        </form>
      </div>

      <div class="canvas-wrap">
        <div id="emptyState" class="empty">Select an image</div>
        <canvas id="imageCanvas" hidden></canvas>
      </div>

      <div class="results">
        <div id="resultGrid" class="result-grid"></div>
      </div>
    </section>
  </main>

  <script>
    const statusEl = document.getElementById("serviceStatus");
    const responseBox = document.getElementById("responseBox");
    const registerForm = document.getElementById("registerForm");
    const deleteForm = document.getElementById("deleteForm");
    const identifyForm = document.getElementById("identifyForm");
    const registerMessage = document.getElementById("registerMessage");
    const deleteMessage = document.getElementById("deleteMessage");
    const identifyMessage = document.getElementById("identifyMessage");
    const refreshStatus = document.getElementById("refreshStatus");
    const canvas = document.getElementById("imageCanvas");
    const ctx = canvas.getContext("2d");
    const emptyState = document.getElementById("emptyState");
    const resultGrid = document.getElementById("resultGrid");

    let lastIdentifyImage = null;

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
        const detail = payload.detail || response.statusText;
        throw Object.assign(new Error(detail), { payload, status: response.status });
      }
      return payload;
    }

    async function loadHealth() {
      try {
        const payload = await requestJson("/health");
        statusEl.textContent = `Ready · ${payload.registered_faces} registered`;
        showResponse(payload);
      } catch (error) {
        statusEl.textContent = "Service unavailable";
        showResponse(error.payload || { detail: error.message });
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

    function drawImage(image) {
      const maxWidth = 1200;
      const scale = Math.min(1, maxWidth / image.naturalWidth);
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      canvas.hidden = false;
      emptyState.hidden = true;
      return scale;
    }

    function drawFaces(image, faces) {
      const scale = drawImage(image);
      ctx.lineWidth = Math.max(2, Math.round(canvas.width / 420));
      ctx.font = "700 16px system-ui, sans-serif";
      ctx.textBaseline = "top";

      for (const face of faces) {
        const [x1, y1, x2, y2] = face.bbox.map((value) => value * scale);
        const recognized = Boolean(face.recognized);
        const color = recognized ? "#087443" : "#b42318";
        const label = recognized
          ? `${face.name || face.id} ${Number(face.match_score || 0).toFixed(3)}`
          : "unknown";
        const textWidth = ctx.measureText(label).width;
        const labelX = Math.max(0, Math.min(x1, canvas.width - textWidth - 14));
        const labelY = Math.max(0, y1 - 28);

        ctx.strokeStyle = color;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = color;
        ctx.fillRect(labelX, labelY, textWidth + 14, 24);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, labelX + 7, labelY + 3);
      }
    }

    function renderResults(faces) {
      resultGrid.innerHTML = "";
      if (!faces.length) {
        const item = document.createElement("div");
        item.className = "result-item";
        const title = document.createElement("strong");
        title.textContent = "No faces";
        const text = document.createElement("span");
        text.textContent = "No face detected in this image.";
        item.append(title, text);
        resultGrid.appendChild(item);
        return;
      }

      for (const [index, face] of faces.entries()) {
        const item = document.createElement("div");
        item.className = "result-item";
        const title = face.recognized ? (face.name || face.id) : "Unknown";
        const matchScore = face.match_score == null ? "-" : Number(face.match_score).toFixed(4);
        const detScore = Number(face.detection_score || 0).toFixed(4);
        const heading = document.createElement("strong");
        const idLine = document.createElement("span");
        const matchLine = document.createElement("span");
        const detectLine = document.createElement("span");
        heading.textContent = `#${index + 1} ${title}`;
        idLine.textContent = `ID: ${face.id || "-"}`;
        matchLine.textContent = `match: ${matchScore}`;
        detectLine.textContent = `detect: ${detScore}`;
        item.append(heading, idLine, matchLine, detectLine);
        resultGrid.appendChild(item);
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

    identifyForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      setBusy(identifyForm, true);
      setMessage(identifyMessage, "Identifying", "");
      try {
        const file = getFile(document.getElementById("identifyImage"));
        const image = await buildImage(file);
        lastIdentifyImage = image;

        const formData = new FormData();
        formData.append("img", file);
        const payload = await requestJson("/faces/identify", { method: "POST", body: formData });
        drawFaces(image, payload.faces || []);
        renderResults(payload.faces || []);
        setMessage(identifyMessage, `${payload.count} face(s) detected`, "ok");
        showResponse(payload);
      } catch (error) {
        setMessage(identifyMessage, error.message, "err");
        showResponse(error.payload || { detail: error.message });
        if (lastIdentifyImage) {
          drawImage(lastIdentifyImage);
        }
      } finally {
        setBusy(identifyForm, false);
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

    refreshStatus.addEventListener("click", loadHealth);

    loadHealth();
  </script>
</body>
</html>
"""
