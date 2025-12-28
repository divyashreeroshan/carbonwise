// static/js/home.js
document.addEventListener("DOMContentLoaded", function () {
  // quick estimate modal
  const quickBtn = document.getElementById("quickEstimateBtn");
  const modalEl = document.getElementById("quickEstimateModal");
  const qeForm = document.getElementById("quickEstimateForm");
  const resultEl = document.getElementById("qe_result");
  const qeGrocery = document.getElementById("qe_grocery");
  const qeKm = document.getElementById("qe_km");
  const qeWaste = document.getElementById("qe_waste");
  const qeClothes = document.getElementById("qe_clothes");

  if (quickBtn && modalEl) {
    quickBtn.addEventListener("click", function () {
      const bsModal = new bootstrap.Modal(modalEl);
      bsModal.show();
      compute();
    });
  }

  function compute() {
    // very rough rule-of-thumb estimator for instant feedback (client-side only)
    const grocery = Number(qeGrocery.value || 0);
    const km = Number(qeKm.value || 0);
    const waste = Number(qeWaste.value || 0);
    const clothes = Number(qeClothes.value || 0);

    // simple factors (conservative): grocery 0.05 kg per currency unit; km 0.24 kg/km; waste bag 5 kg each; clothes 2 kg each
    const est = grocery * 0.05 + km * 0.24 + waste * 5 + clothes * 2;
    resultEl.innerText = `${Math.round(est)} kgCO₂e`;
  }

  // recompute when inputs change
  [qeGrocery, qeKm, qeWaste, qeClothes].forEach(function (el) {
    if (!el) return;
    el.addEventListener("input", compute);
  });

  // smooth scroll: when clicking start calculator button, go to /personal
  const startBtns = document.querySelectorAll(".btn-cta");
  startBtns.forEach(btn => {
    btn.addEventListener("click", function (e) {
      // allow normal navigation; add a small animation before redirect
      btn.classList.add("disabled");
      setTimeout(() => { window.location = btn.getAttribute("href"); }, 220);
      e.preventDefault();
    });
  });
});
// appended to static/js/home.js
document.addEventListener("DOMContentLoaded", function () {
  const thumbLinks = document.querySelectorAll(".chart-thumb-link");
  if (thumbLinks && thumbLinks.length) {
    thumbLinks.forEach(function (a) {
      a.addEventListener("click", function (e) {
        e.preventDefault();
        const src = a.getAttribute("data-chart");
        const modalImg = document.getElementById("chartModalImg");
        modalImg.src = src;
        const modalEl = document.getElementById("chartModal");
        const bsModal = new bootstrap.Modal(modalEl);
        bsModal.show();
      });
    });
  }
});
document.addEventListener('click', (e) => {
  const t = e.target.closest('.tab');
  if (!t) return;
  const siblings = t.parentElement?.querySelectorAll('.tab') || [];
  siblings.forEach(s => s.classList.remove('tab-active'));
  t.classList.add('tab-active');
});

