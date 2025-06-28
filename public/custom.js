window.addEventListener("DOMContentLoaded", () => {
  // Ortada büyük logo (sadece ilk açılışta göster, sonra kaybolsun)
  if (!document.querySelector(".center-big-logo")) {
    const logo = document.createElement("img");
    logo.src = "/logo.png"; // Ortadaki büyük logo
    logo.className = "center-big-logo";
    logo.alt = "Logo";
    document.body.appendChild(logo);

    // 2 saniye sonra ortadaki büyük logoyu gizle
    setTimeout(() => {
      logo.style.display = "none";
    }, 2000);
  }

  // Footer üstü küçük logo (her zaman görünür)
  if (!document.querySelector(".footer-logo")) {
    const footerLogo = document.createElement("img");
    footerLogo.src = "/footerlogo.png"; // Footer üstündeki küçük logo
    footerLogo.className = "footer-logo";
    footerLogo.alt = "Footer Logo";
    document.body.appendChild(footerLogo);
  }

  // Footer imza
  if (!document.querySelector(".footer-signature")) {
    const footer = document.createElement("div");
    footer.className = "footer-signature";
    footer.innerText = "Prepared by Arif Aşıroğlu for the Jr. AI Specialist position upon Obilet's request.";
    document.body.appendChild(footer);
  }
});