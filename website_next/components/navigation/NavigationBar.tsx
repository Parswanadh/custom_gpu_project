'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import { Cpu } from 'lucide-react';

const navItems = [
  { name: 'Architecture', href: '#architecture' },
  { name: 'Performance', href: '#metrics' },
  { name: 'Comparison', href: '#comparison' },
  { name: 'Live Demo', href: '#demo' },
  { name: 'Roadmap', href: '#roadmap' },
];

export default function NavigationBar() {
  const [isScrolled, setIsScrolled] = useState(false);
  const [activeSection, setActiveSection] = useState('');

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);

      // Check which section is strictly active
      const sections = navItems.map((item) => item.href.substring(1));
      for (const section of sections.reverse()) {
        const el = document.getElementById(section);
        if (el && window.scrollY >= el.offsetTop - 150) {
          setActiveSection(section);
          break;
        }
      }
    };

    window.addEventListener('scroll', handleScroll);
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ type: 'spring', stiffness: 100, damping: 20 }}
      className={clsx(
        'fixed top-0 inset-x-0 z-50 transition-all duration-300',
        isScrolled 
          ? 'py-3 backdrop-blur-2xl bg-silicon-black/70 border-b border-die-copper/40 shadow-[0_4px_30px_rgba(0,0,0,0.5)]' 
          : 'py-6 bg-transparent border-transparent'
      )}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between">
        <div className="flex items-center gap-2 text-white font-space font-bold text-xl glow-on-hover">
          <Cpu className="w-6 h-6 text-neon-cyan" />
          <span>BITBYBIT</span>
        </div>

        <div className="hidden md:flex items-center gap-8">
          {navItems.map((item) => (
            <a
              key={item.name}
              href={item.href}
              className={clsx(
                "font-inter text-sm font-medium transition-colors hover:text-white relative",
                activeSection === item.href.substring(1) ? "text-white" : "text-metal-silver"
              )}
            >
              {item.name}
              {activeSection === item.href.substring(1) && (
                <motion.div
                  layoutId="nav-indicator"
                  className="absolute -bottom-2 left-0 right-0 h-0.5 bg-neon-cyan shadow-[0_0_10px_rgba(0,245,255,0.8)]"
                />
              )}
            </a>
          ))}
        </div>

        <button className="hidden md:block py-2 px-6 rounded-full font-dm text-sm font-bold text-silicon-black bg-white hover:bg-oxide-green transition-colors shadow-[0_0_15px_rgba(255,255,255,0.3)] hover:shadow-[0_0_20px_rgba(0,255,136,0.6)]">
          View GitHub
        </button>
      </div>
    </motion.nav>
  );
}
