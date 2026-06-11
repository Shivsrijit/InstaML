import React, { useEffect, useRef } from 'react';

const InteractiveBackground = () => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId;
    let particles = [];
    const particleCount = 65;
    const connectionDistance = 110;
    const mouseRadius = 150;
    let mouse = { x: null, y: null };

    // Setup dimensions
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Particle class definition
    class Particle {
      constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.vx = (Math.random() - 0.5) * 0.4; // very slow drift
        this.vy = (Math.random() - 0.5) * 0.4;
        this.radius = Math.random() * 2 + 1.5;
        this.baseRadius = this.radius;
      }

      update() {
        // Move particle
        this.x += this.vx;
        this.y += this.vy;

        // Wrap around boundaries
        if (this.x < 0) this.x = canvas.width;
        if (this.x > canvas.width) this.x = 0;
        if (this.y < 0) this.y = canvas.height;
        if (this.y > canvas.height) this.y = 0;

        // Mouse interaction (repelling force)
        if (mouse.x !== null && mouse.y !== null) {
          const dx = this.x - mouse.x;
          const dy = this.y - mouse.y;
          const dist = Math.hypot(dx, dy);

          if (dist < mouseRadius) {
            // Force coefficient pushing particle away
            const force = (mouseRadius - dist) / mouseRadius;
            const angle = Math.atan2(dy, dx);
            this.x += Math.cos(angle) * force * 1.5;
            this.y += Math.sin(angle) * force * 1.5;
            this.radius = this.baseRadius + force * 2; // grow slightly
          } else {
            if (this.radius > this.baseRadius) {
              this.radius -= 0.1;
            }
          }
        } else {
          if (this.radius > this.baseRadius) {
            this.radius -= 0.1;
          }
        }
      }

      draw(theme) {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        
        let isHovered = false;
        if (mouse.x !== null && mouse.y !== null) {
          const dist = Math.hypot(this.x - mouse.x, this.y - mouse.y);
          if (dist < mouseRadius) isHovered = true;
        }

        if (isHovered) {
          // Glow slightly brighter when close to mouse
          ctx.fillStyle = theme === 'light'
            ? 'rgba(28, 27, 24, 0.35)'
            : 'rgba(255, 255, 255, 0.35)';
        } else {
          ctx.fillStyle = theme === 'light' 
            ? 'rgba(28, 27, 24, 0.12)' 
            : 'rgba(255, 255, 255, 0.08)';
        }
        ctx.fill();
      }
    }

    // Initialize particles
    const init = () => {
      particles = [];
      for (let i = 0; i < particleCount; i++) {
        particles.push(new Particle());
      }
    };
    init();

    // Mouse handlers
    const handleMouseMove = (e) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
    };

    const handleMouseLeave = () => {
      mouse.x = null;
      mouse.y = null;
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseleave', handleMouseLeave);

    // Animation Loop
    const animate = () => {
      const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw connections
      for (let i = 0; i < particles.length; i++) {
        const p1 = particles[i];
        
        // Connect to mouse cursor
        if (mouse.x !== null && mouse.y !== null) {
          const dx = p1.x - mouse.x;
          const dy = p1.y - mouse.y;
          const dist = Math.hypot(dx, dy);
          if (dist < mouseRadius) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(mouse.x, mouse.y);
            const alpha = ((mouseRadius - dist) / mouseRadius) * 0.12;
            ctx.strokeStyle = currentTheme === 'light'
              ? `rgba(28, 27, 24, ${alpha})`
              : `rgba(255, 255, 255, ${alpha})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }

        // Connect to other particles
        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j];
          const dx = p1.x - p2.x;
          const dy = p1.y - p2.y;
          const dist = Math.hypot(dx, dy);

          if (dist < connectionDistance) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            
            const lineAlpha = ((connectionDistance - dist) / connectionDistance) * 0.05;
            
            ctx.strokeStyle = currentTheme === 'light'
              ? `rgba(31, 30, 26, ${lineAlpha})`
              : `rgba(245, 244, 240, ${lineAlpha})`;
            ctx.lineWidth = 0.75;
            ctx.stroke();
          }
        }
      }

      // Update and draw particles
      particles.forEach(p => {
        p.update();
        p.draw(currentTheme);
      });

      animationFrameId = requestAnimationFrame(animate);
    };

    animate();

    // Cleanup listeners
    return () => {
      window.removeEventListener('resize', resizeCanvas);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseleave', handleMouseLeave);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <>
      <div className="radial-glow glow-1"></div>
      <div className="radial-glow glow-2"></div>
      <div className="radial-glow glow-3"></div>
      <canvas 
        ref={canvasRef} 
        className="interactive-bg-canvas"
      />
    </>
  );
};

export default InteractiveBackground;
