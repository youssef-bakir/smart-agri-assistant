// web/common.js - Universal bilingual i18n & navigation logic
// No external dependencies, strictly vanilla JavaScript

let currentLang = localStorage.getItem("smart_agri_lang") || "ar";
let translations = {};

async function loadTranslations(lang) {
  try {
    const res = await fetch(`/i18n/${lang}.json`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    translations = await res.json();
    currentLang = lang;
    localStorage.setItem("smart_agri_lang", lang);
    applyTranslations();
  } catch (err) {
    console.error("Failed to load i18n dictionary:", err);
  }
}

function applyTranslations() {
  document.documentElement.lang = currentLang;
  document.documentElement.dir = currentLang === "ar" ? "rtl" : "ltr";

  // Translate all marked text nodes
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (translations[key]) {
      el.textContent = translations[key];
    }
  });

  // Translate placeholders if any
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    const key = el.getAttribute("data-i18n-placeholder");
    if (translations[key]) {
      el.placeholder = translations[key];
    }
  });

  // Update language toggle button label
  const langBtn = document.getElementById("lang-toggle-btn");
  if (langBtn) {
    langBtn.textContent = translations["lang_switch_label"] || (currentLang === "ar" ? "English" : "العربية");
  }

  // Dispatch custom event for page listeners
  window.dispatchEvent(new CustomEvent("languageChanged", { detail: { lang: currentLang, translations } }));
}

function switchLanguage() {
  const nextLang = currentLang === "ar" ? "en" : "ar";
  loadTranslations(nextLang);
}

function t(key, fallback = "") {
  return translations[key] || fallback || key;
}

// Expose globally
window.t = t;
window.getCurrentLang = () => currentLang;
window.switchLanguage = switchLanguage;

// Initial bootstrap on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  const langBtn = document.getElementById("lang-toggle-btn");
  if (langBtn) {
    langBtn.addEventListener("click", switchLanguage);
  }
  loadTranslations(currentLang);
});
