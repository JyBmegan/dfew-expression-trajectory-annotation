(() => {
  const shell = document.querySelector("[data-assignment]");
  if (!shell) return;
  const assignment = shell.dataset.assignment;
  const taskType = shell.dataset.taskType;
  const form = document.querySelector("#rating-form");
  const errorBox = document.querySelector("#form-error");
  const saveStatus = document.querySelector("#save-status");
  const saveStatusText = document.querySelector("#save-status-text");
  const retrySave = document.querySelector("#retry-save");
  let pendingDraft = null;
  let draftTimer = null;
  let draftSaving = false;
  let draftFailed = false;
  let leavingAfterSubmit = false;

  const setSaveStatus = (kind, message) => {
    if (!saveStatus) return;
    saveStatus.dataset.state = kind;
    saveStatusText.textContent = message;
    retrySave?.classList.toggle("hidden", kind !== "failed");
  };

  const persistDraft = async () => {
    clearTimeout(draftTimer);
    if (draftSaving || !pendingDraft) return;
    const payload = pendingDraft;
    pendingDraft = null;
    draftSaving = true;
    draftFailed = false;
    setSaveStatus("saving", "正在自动保存…");
    try {
      const response = await fetch(`/api/draft/${assignment}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error("自动保存失败");
      await response.json();
      const stamp = new Intl.DateTimeFormat("zh-CN", {
        hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
      }).format(new Date());
      setSaveStatus("saved", `已保存到本机 · ${stamp}`);
    } catch (_error) {
      pendingDraft = payload;
      draftFailed = true;
      setSaveStatus("failed", "自动保存失败，请勿关闭页面");
    } finally {
      draftSaving = false;
      if (pendingDraft && !draftFailed) {
        draftTimer = setTimeout(persistDraft, 120);
      }
    }
  };

  const queueDraft = (payload, delay = 300) => {
    pendingDraft = payload;
    draftFailed = false;
    setSaveStatus("dirty", "内容已改变，等待保存…");
    clearTimeout(draftTimer);
    draftTimer = setTimeout(persistDraft, delay);
  };

  retrySave?.addEventListener("click", persistDraft);
  window.addEventListener("beforeunload", (event) => {
    if (!leavingAfterSubmit && (pendingDraft || draftSaving || draftFailed)) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
  let activeMilliseconds = 0;
  let lastActivity = performance.now();
  let pageWasVisible = !document.hidden;
  const markActivity = () => {
    const now = performance.now();
    if (pageWasVisible) activeMilliseconds += Math.min(now - lastActivity, 60000);
    lastActivity = now;
    pageWasVisible = !document.hidden;
  };
  ["pointerdown", "keydown", "input", "change"].forEach(name => {
    window.addEventListener(name, markActivity, {passive: true});
  });
  document.addEventListener("visibilitychange", markActivity);
  const categoryKeys = {"1":"Happiness", "2":"Sadness", "3":"Neutral", "4":"Anger", "5":"Surprise", "6":"Disgust", "7":"Fear", "m":"Mixed", "u":"Unclear", "x":"Face not visible"};
  const chooseCategory = (name, category) => {
    const radio = form.querySelector(`[name="${name}"][value="${CSS.escape(category)}"]`);
    if (radio) {
      radio.checked = true;
      radio.dispatchEvent(new Event("change", {bubbles: true}));
    }
  };

  const showError = (message) => {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
  };

  const submit = async (payload) => {
    const button = form.querySelector(".submit-rating");
    button.disabled = true;
    errorBox.classList.add("hidden");
    setSaveStatus("saving", "正在提交本题…");
    markActivity();
    payload.duration_ms = Math.round(activeMilliseconds);
    try {
      const response = await fetch(`/api/rating/${assignment}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "保存失败，请重试。");
      leavingAfterSubmit = true;
      window.location.assign(result.next);
    } catch (error) {
      button.disabled = false;
      setSaveStatus("failed", "提交失败，答案仍保留在当前页面");
      showError(error.message);
    }
  };

  if (taskType === "frame") {
    const slider = form.elements.intensity;
    const output = document.querySelector("#intensity-output");
    const saveFrameDraft = () => {
      const data = new FormData(form);
      queueDraft({
        visible_category: data.get("visible_category"),
        intensity: Number(slider.value),
      });
    };
    try {
      const draft = JSON.parse(document.querySelector("#draft-data").textContent || "{}");
      if (draft.visible_category) chooseCategory("visible_category", draft.visible_category);
      if (Number.isInteger(Number(draft.intensity))) {
        slider.value = Math.max(0, Math.min(6, Number(draft.intensity)));
        output.value = slider.value;
      }
      if (draft.visible_category || draft.intensity !== undefined) {
        setSaveStatus("saved", "已恢复上次自动保存的内容");
      }
    } catch (_) { /* Ignore malformed drafts; submitted ratings remain authoritative. */ }
    slider.addEventListener("input", () => { output.value = slider.value; });
    slider.addEventListener("input", saveFrameDraft);
    form.querySelectorAll('[name="visible_category"]').forEach(radio => radio.addEventListener("change", () => {
      if (radio.checked && (radio.value === "Neutral" || radio.value === "Face not visible")) {
        slider.value = 0;
        output.value = 0;
      }
      saveFrameDraft();
    }));
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const category = new FormData(form).get("visible_category");
      if (!category) return showError("请先选择可见表情类别。");
      submit({visible_category: category, intensity: Number(slider.value)});
    });
    window.addEventListener("keydown", (event) => {
      const key = event.key.toLowerCase();
      if (categoryKeys[key]) {
        event.preventDefault();
        chooseCategory("visible_category", categoryKeys[key]);
      } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        const change = event.key === "ArrowRight" ? 1 : -1;
        slider.value = Math.max(0, Math.min(6, Number(slider.value) + change));
        output.value = slider.value;
        saveFrameDraft();
      } else if (event.key === "Enter") {
        event.preventDefault();
        form.requestSubmit();
      }
    });
    return;
  }

  const urls = JSON.parse(document.querySelector("#frame-urls").textContent);
  const image = document.querySelector("#sequence-frame");
  const chip = document.querySelector("#frame-chip");
  const playButton = document.querySelector("#play-pause");
  const speed = document.querySelector("#playback-speed");
  const cells = [...document.querySelectorAll(".trajectory-cell")];
  const sliders = cells.map(cell => cell.querySelector("input"));
  let frame = 0;
  let playing = true;
  let timer = null;

  urls.forEach(url => { const preload = new Image(); preload.src = url; });

  const renderFrame = (next) => {
    frame = (next + 16) % 16;
    image.src = urls[frame];
    image.alt = `Sequence frame ${frame + 1}`;
    chip.textContent = `${frame + 1} / 16`;
    cells.forEach((cell, index) => cell.classList.toggle("active", index === frame));
  };

  const schedule = () => {
    clearTimeout(timer);
    if (!playing) return;
    timer = setTimeout(() => { renderFrame(frame + 1); schedule(); }, Number(speed.value));
  };

  const setPlaying = (value) => {
    playing = value;
    playButton.textContent = playing ? "暂停" : "播放";
    schedule();
  };

  window.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if (categoryKeys[key]) {
      event.preventDefault();
      chooseCategory("dominant_category", categoryKeys[key]);
    } else if (event.code === "Space") {
      event.preventDefault();
      setPlaying(!playing);
    } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      setPlaying(false);
      renderFrame(frame + (event.key === "ArrowRight" ? 1 : -1));
    } else if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      event.preventDefault();
      const slider = sliders[frame];
      const change = event.key === "ArrowUp" ? 1 : -1;
      slider.value = Math.max(0, Math.min(6, Number(slider.value) + change));
      cells[frame].querySelector("output").value = slider.value;
      saveSequenceDraft();
    }
  });

  playButton.addEventListener("click", () => setPlaying(!playing));
  speed.addEventListener("change", schedule);
  document.querySelector("#previous-frame").addEventListener("click", () => { setPlaying(false); renderFrame(frame - 1); });
  document.querySelector("#next-frame").addEventListener("click", () => { setPlaying(false); renderFrame(frame + 1); });
  cells.forEach((cell, index) => cell.addEventListener("click", (event) => {
    if (event.target.tagName !== "INPUT") { setPlaying(false); renderFrame(index); }
  }));

  const collect = () => {
    const data = new FormData(form);
    return {
      dominant_category: data.get("dominant_category"),
      intensities: sliders.map(slider => Number(slider.value)),
      occlusion: data.has("occlusion"),
      speaking: data.has("speaking"),
      abrupt_change: data.has("abrupt_change"),
      subject_switch: data.has("subject_switch"),
    };
  };

  form.querySelectorAll('[name="dominant_category"]').forEach(radio => radio.addEventListener("change", () => {
    if (radio.checked && (radio.value === "Neutral" || radio.value === "Face not visible")) {
      sliders.forEach((slider, index) => {
        slider.value = 0;
        cells[index].querySelector("output").value = 0;
      });
    }
  }));

  const saveSequenceDraft = () => queueDraft(collect(), 400);

  sliders.forEach((slider, index) => slider.addEventListener("input", () => {
    cells[index].querySelector("output").value = slider.value;
    renderFrame(index);
    setPlaying(false);
    saveSequenceDraft();
  }));
  form.addEventListener("change", saveSequenceDraft);

  try {
    const draft = JSON.parse(document.querySelector("#draft-data").textContent || "{}");
    if (draft.dominant_category) {
      const radio = form.querySelector(`[name="dominant_category"][value="${CSS.escape(draft.dominant_category)}"]`);
      if (radio) radio.checked = true;
    }
    if (Array.isArray(draft.intensities) && draft.intensities.length === 16) {
      draft.intensities.forEach((value, index) => {
        sliders[index].value = value;
        cells[index].querySelector("output").value = value;
      });
    }
    ["occlusion", "speaking", "abrupt_change", "subject_switch"].forEach(name => {
      if (draft[name]) form.elements[name].checked = true;
    });
    if (draft.dominant_category || Array.isArray(draft.intensities)) {
      setSaveStatus("saved", "已恢复上次自动保存的内容");
    }
  } catch (_) { /* A malformed draft is ignored; submitted ratings remain authoritative. */ }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const payload = collect();
    if (!payload.dominant_category) return showError("请先选择整个序列中的主要可见表情。");
    submit(payload);
  });
  renderFrame(0);
  schedule();
})();
