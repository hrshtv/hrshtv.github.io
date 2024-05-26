// Based on: https://derekkedziora.com/blog/dark-mode-revisited
  
// this checks whether system dark mode is set 
let systemInitiatedDark = window.matchMedia("(prefers-color-scheme: dark)"); // detects system preference
// // this checks for session storage telling to override the system preferences 
let theme = sessionStorage.getItem('theme');
// console.log(theme);
// let theme = sessionStorage.getItem('theme');
// document.getElementById("theme-toggle").className = "bi bi-moon-fill";

if (systemInitiatedDark.matches) {
  // i.e., if dark theme
  // document.getElementById("theme-toggle").className = "bi-brightness-high-fill";
  document.getElementById("theme-toggle").className = "bi bi-moon-fill";
} 
else {
  document.getElementById("theme-toggle").className = "bi bi-moon-fill";
}

function prefersColorTest(systemInitiatedDark) {
  if (systemInitiatedDark.matches) {
    document.documentElement.setAttribute('data-theme', 'light');
    document.getElementById("theme-toggle").className = "bi bi-moon-fill";
    // document.documentElement.setAttribute('data-theme', 'dark');    
    // document.getElementById("theme-toggle").className = "bi bi-brightness-high-fill";
  } 
  else {
    document.documentElement.setAttribute('data-theme', 'light');
    document.getElementById("theme-toggle").className = "bi bi-moon-fill";
  }
  sessionStorage.setItem('theme', ''); // clear session storage
}
systemInitiatedDark.addListener(prefersColorTest);

function modeSwitcher() {
  // it’s important to check for overrides again 
  let theme = sessionStorage.getItem('theme');
  // checks if reader selected dark mode 
  if (theme === "dark") {
    document.documentElement.setAttribute('data-theme', 'light');
    sessionStorage.setItem('theme', 'light');
    document.getElementById("theme-toggle").className = "bi bi-moon-fill";
  } 
  else if (theme === "light") {
    document.documentElement.setAttribute('data-theme', 'dark');
    sessionStorage.setItem('theme', 'dark');
    document.getElementById("theme-toggle").className = "bi bi-brightness-high-fill"; 
  } 
  else if (systemInitiatedDark.matches) { 
    document.documentElement.setAttribute('data-theme', 'light');
    sessionStorage.setItem('theme', 'light');
    document.getElementById("theme-toggle").className = "bi bi-moon-fill";
  } 
  else {
    document.documentElement.setAttribute('data-theme', 'dark');
    sessionStorage.setItem('theme', 'dark');
    document.getElementById("theme-toggle").className = "bi bi-brightness-high-fill";
  }
}

if (theme === "dark") {
  document.documentElement.setAttribute('data-theme', 'dark');
  sessionStorage.setItem('theme', 'dark');
  document.getElementById("theme-toggle").className = "bi bi-brightness-high-fill";
} 
else if (theme === "light") {
  document.documentElement.setAttribute('data-theme', 'light');
  sessionStorage.setItem('theme', 'light');
  document.getElementById("theme-toggle").className = "bi bi-moon-fill";
}
else {
  document.documentElement.setAttribute('data-theme', 'light');
  sessionStorage.setItem('theme', 'light');
  document.getElementById("theme-toggle").className = "bi bi-moon-fill";
}
