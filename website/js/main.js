/* ============================================
   BitbyBit — Website Interactions & Animations
   Premium particle effects, tilt cards, typing
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {
  initParticleCanvas();
  initScrollReveal();
  initAnimatedCounters();
  initTechCards();
  initModuleTabs();
  initSmoothScroll();
  initNavbar();
  initCardTilt();
  initTypingEffect();
  initSectionDividers();
  initFlowPulse();
});

/* ─────────────────────────────────────────────
   PARTICLE CANVAS — Floating connected dots
   ───────────────────────────────────────────── */
function initParticleCanvas() {
  const canvas = document.getElementById('particle-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let w, h, particles, mouse;
  const PARTICLE_COUNT = 80;
  const CONNECT_DIST = 140;
  const MOUSE_DIST = 180;

  function resize() {
    w = canvas.width = canvas.offsetWidth;
    h = canvas.height = canvas.offsetHeight;
  }

  mouse = { x: -1000, y: -1000 };
  canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    mouse.x = e.clientX - rect.left;
    mouse.y = e.clientY - rect.top;
  });
  canvas.addEventListener('mouseleave', () => {
    mouse.x = -1000;
    mouse.y = -1000;
  });

  class Particle {
    constructor() {
      this.reset();
    }
    reset() {
      this.x = Math.random() * w;
      this.y = Math.random() * h;
      this.vx = (Math.random() - 0.5) * 0.5;
      this.vy = (Math.random() - 0.5) * 0.5;
      this.r = Math.random() * 2 + 0.5;
      this.alpha = Math.random() * 0.5 + 0.2;
      // Color: blue or cyan
      this.color = Math.random() > 0.6 ? '6,182,212' : '59,130,246';
    }
    update() {
      this.x += this.vx;
      this.y += this.vy;
      if (this.x < 0 || this.x > w) this.vx *= -1;
      if (this.y < 0 || this.y > h) this.vy *= -1;
      // Gentle mouse repel
      const dx = this.x - mouse.x;
      const dy = this.y - mouse.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < MOUSE_DIST) {
        const force = (MOUSE_DIST - dist) / MOUSE_DIST * 0.02;
        this.vx += dx * force;
        this.vy += dy * force;
      }
      // Dampen velocity
      this.vx *= 0.99;
      this.vy *= 0.99;
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${this.color},${this.alpha})`;
      ctx.fill();
    }
  }

  function init() {
    resize();
    particles = Array.from({ length: PARTICLE_COUNT }, () => new Particle());
  }

  function drawConnections() {
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < CONNECT_DIST) {
          const alpha = (1 - dist / CONNECT_DIST) * 0.15;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(59,130,246,${alpha})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
  }

  function animate() {
    ctx.clearRect(0, 0, w, h);
    particles.forEach(p => { p.update(); p.draw(); });
    drawConnections();
    requestAnimationFrame(animate);
  }

  window.addEventListener('resize', () => { resize(); });
  init();
  animate();
}

/* ─────────────────────────────────────────────
   SCROLL REVEAL — Intersection Observer
   ───────────────────────────────────────────── */
function initScrollReveal() {
  const reveals = document.querySelectorAll('.reveal');
  if (!reveals.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('revealed');
      }
    });
  }, {
    threshold: 0.12,
    rootMargin: '0px 0px -60px 0px'
  });

  reveals.forEach(el => observer.observe(el));
}

/* ─────────────────────────────────────────────
   ANIMATED COUNTERS — Count up with easing
   ───────────────────────────────────────────── */
function initAnimatedCounters() {
  const counters = document.querySelectorAll('[data-count]');
  if (!counters.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting && !entry.target.dataset.counted) {
        entry.target.dataset.counted = 'true';
        animateCounter(entry.target);
      }
    });
  }, { threshold: 0.5 });

  counters.forEach(el => observer.observe(el));
}

function animateCounter(el) {
  const target = parseInt(el.dataset.count, 10);
  const suffix = el.dataset.suffix || '';
  const prefix = el.dataset.prefix || '';
  const duration = 2200;
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // Ease-out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(eased * target);
    el.textContent = prefix + current.toLocaleString() + suffix;
    if (progress < 1) requestAnimationFrame(update);
  }

  requestAnimationFrame(update);
}

/* ─────────────────────────────────────────────
   TECH CARD EXPAND/COLLAPSE
   ───────────────────────────────────────────── */
function initTechCards() {
  const cards = document.querySelectorAll('.tech-card');
  cards.forEach(card => {
    const header = card.querySelector('.tech-card-header');
    if (!header) return;
    header.addEventListener('click', () => {
      const wasExpanded = card.classList.contains('expanded');
      cards.forEach(c => c.classList.remove('expanded'));
      if (!wasExpanded) card.classList.add('expanded');
    });
  });
}

/* ─────────────────────────────────────────────
   MODULE CATEGORY TABS
   ───────────────────────────────────────────── */
function initModuleTabs() {
  const tabs = document.querySelectorAll('.module-tab');
  const panels = document.querySelectorAll('.module-panel');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      tabs.forEach(t => t.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      const panel = document.getElementById(target);
      if (panel) panel.classList.add('active');
    });
  });
}

/* ─────────────────────────────────────────────
   SMOOTH SCROLL for Nav Links
   ───────────────────────────────────────────── */
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      const targetId = link.getAttribute('href').slice(1);
      const target = document.getElementById(targetId);
      if (target) {
        const navHeight = document.querySelector('.navbar')?.offsetHeight || 70;
        const top = target.getBoundingClientRect().top + window.scrollY - navHeight;
        window.scrollTo({ top, behavior: 'smooth' });
        document.querySelector('.nav-links')?.classList.remove('open');
      }
    });
  });
}

/* ─────────────────────────────────────────────
   NAVBAR — Compact on scroll, mobile toggle
   ───────────────────────────────────────────── */
function initNavbar() {
  const navbar = document.querySelector('.navbar');
  const toggle = document.querySelector('.nav-toggle');
  const links = document.querySelector('.nav-links');
  if (!navbar) return;

  window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  }, { passive: true });

  if (toggle && links) {
    toggle.addEventListener('click', () => links.classList.toggle('open'));
  }
}

/* ─────────────────────────────────────────────
   CARD TILT — Subtle 3D tilt on mouse move
   ───────────────────────────────────────────── */
function initCardTilt() {
  const cards = document.querySelectorAll('.arch-card, .result-card');
  cards.forEach(card => {
    card.addEventListener('mousemove', e => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const cx = rect.width / 2;
      const cy = rect.height / 2;
      const rotateX = (y - cy) / cy * -4; // max ±4deg
      const rotateY = (x - cx) / cx * 4;
      card.style.transform = `perspective(800px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-4px)`;
    });

    card.addEventListener('mouseleave', () => {
      card.style.transform = 'perspective(800px) rotateX(0) rotateY(0) translateY(0)';
    });
  });
}

/* ─────────────────────────────────────────────
   TYPING EFFECT — Hero tagline types out
   ───────────────────────────────────────────── */
function initTypingEffect() {
  const el = document.querySelector('.hero-tagline');
  if (!el) return;

  const text = el.textContent;
  el.textContent = '';
  el.style.borderRight = '2px solid var(--accent-blue)';
  el.style.opacity = '1';

  let i = 0;
  const speed = 28; // ms per char

  function type() {
    if (i < text.length) {
      el.textContent += text.charAt(i);
      i++;
      setTimeout(type, speed);
    } else {
      // Blinking cursor for a moment, then remove
      setTimeout(() => {
        el.style.borderRight = 'none';
      }, 1500);
    }
  }

  // Start after hero fade-in
  setTimeout(type, 800);
}

/* ─────────────────────────────────────────────
   SECTION DIVIDERS — Animated gradient lines
   ───────────────────────────────────────────── */
function initSectionDividers() {
  const sections = document.querySelectorAll('section');
  sections.forEach(section => {
    // Add glow line between sections via observer
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('section-active');
        }
      });
    }, { threshold: 0.05 });
    observer.observe(section);
  });
}

/* ─────────────────────────────────────────────
   FLOW PULSE — Animate the pipeline connector
   ───────────────────────────────────────────── */
function initFlowPulse() {
  const pipeline = document.querySelector('.flow-pipeline');
  if (!pipeline) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        pipeline.classList.add('flow-active');
      }
    });
  }, { threshold: 0.2 });

  observer.observe(pipeline);
}
