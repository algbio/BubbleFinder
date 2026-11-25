"use strict";

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    const block = document.getElementById(button.dataset.copy);
    if (!block) return;
    const text = block.textContent.replace(/\n$/, "");
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        const field = document.createElement("textarea");
        field.value = text;
        field.style.position = "fixed";
        field.style.opacity = "0";
        document.body.appendChild(field);
        field.select();
        const copied = document.execCommand("copy");
        field.remove();
        button.focus();
        if (!copied) throw new Error("Clipboard unavailable");
      }
      button.textContent = "Copied";
    } catch {
      button.textContent = "Select text to copy";
    }
    window.setTimeout(() => { button.textContent = "Copy"; }, 2200);
  });
});

document.querySelectorAll("[data-tab-group]").forEach((group) => {
  const tabs = Array.from(group.querySelectorAll('[role="tab"]'));
  const activate = (selected, focus = false) => {
    for (const tab of tabs) {
      const isSelected = tab === selected;
      tab.setAttribute("aria-selected", String(isSelected));
      tab.tabIndex = isSelected ? 0 : -1;
      const panel = document.getElementById(tab.getAttribute("aria-controls"));
      if (panel) panel.hidden = !isSelected;
    }
    if (focus) selected.focus();
  };
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activate(tab));
    tab.addEventListener("keydown", (event) => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      if (next !== undefined) {
        event.preventDefault();
        activate(tabs[next], true);
      }
    });
  });
  if (tabs.length) activate(tabs.find(tab => tab.getAttribute("aria-selected") === "true") || tabs[0]);
});
