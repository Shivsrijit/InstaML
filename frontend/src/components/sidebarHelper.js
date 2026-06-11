export const toggleSidebar = () => {
  const isMobile = window.innerWidth <= 768;
  if (isMobile) {
    const isOpen = document.documentElement.classList.contains('sidebar-mobile-open');
    if (isOpen) {
      document.documentElement.classList.remove('sidebar-mobile-open');
    } else {
      document.documentElement.classList.add('sidebar-mobile-open');
    }
  } else {
    const isHidden = document.documentElement.classList.contains('sidebar-hidden');
    if (isHidden) {
      document.documentElement.classList.remove('sidebar-hidden');
      localStorage.setItem('sidebar-hidden', 'false');
    } else {
      document.documentElement.classList.add('sidebar-hidden');
      localStorage.setItem('sidebar-hidden', 'true');
    }
  }
};

export const closeMobileSidebar = () => {
  document.documentElement.classList.remove('sidebar-mobile-open');
};
