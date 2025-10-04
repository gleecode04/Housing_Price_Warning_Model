// Project timeline data and Gantt chart implementation
document.addEventListener('DOMContentLoaded', function () {
  // Gantt Chart Data
  const ganttData = {
    tasks: [
      {
        name: 'Dataset acquistion & API integration',
        assignee: 'Husaan',
        startWeek: 1,
        duration: 2,
        color: '#10b981'
      },
      {
        name: 'Literature review & proposal drafting',
        assignee: 'Sanjay',
        startWeek: 1,
        duration: 2,
        color: '#81141f'
      },
      {
        name: 'Data preprocessing, leakage control',
        assignee: 'Ivan',
        startWeek: 3,
        duration: 2,
        color: '#3b82f6'
      },
      {
        name: 'Feature engineering',
        assignee: 'Husaan',
        startWeek: 3,
        duration: 2,
        color: '#10b981'
      },
      {
        name: 'Baseline models (Elastic Net, Random Forests)',
        assignee: 'Chris',
        startWeek: 5,
        duration: 2,
        color: '#f59e0b'
      },
      {
        name: 'Hyperparameter tuning pipelines',
        assignee: 'Daniel',
        startWeek: 5,
        duration: 2,
        color: '#6366f1'
      },
      {
        name: 'Advanced models (XGBoost, LightGBM, clustering)',
        assignee: 'Chris + Daniel',
        startWeek: 7,
        duration: 2,
        color: '#aaaaaa'
      },
      {
        name: 'Calibration & fairness evaluation',
        assignee: 'Ivan',
        startWeek: 7,
        duration: 2,
        color: '#3b82f6'
      },
      {
        name: 'Backtesting & financial evaluation',
        assignee: 'Husaan + Daniel',
        startWeek: 9,
        duration: 2,
        color: '#aaaaaa'
      },
      {
        name: 'Results write-up',
        assignee: 'Sanjay',
        startWeek: 9,
        duration: 2,
        color: '#81141f'
      },
      {
        name: 'Final report polishing and GitHub Pages deployment',
        assignee: 'Ivan + Sanjay',
        startWeek: 11,
        duration: 2,
        color: '#aaaaaa'
      },
      {
        name: 'Presentation prep and rehearsal',
        assignee: 'All Members',
        startWeek: 11,
        duration: 2,
        color: '#aaaaaa'
      }
    ],
    weeks: 12
  };

  // Create Gantt Chart using Chart.js
  function createGanttChart() {
    const canvas = document.getElementById('ganttChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Chart dimensions
    const chartWidth = canvas.width - 320; // Leave space for task names
    const chartHeight = canvas.height - 80; // Leave space for week labels
    const barHeight = 20;
    const barSpacing = 25;
    const startX = 300;
    const startY = 50;

    // Draw week headers
    ctx.font = '12px Inter, sans-serif';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';

    for (let week = 1; week <= ganttData.weeks; week++) {
      const x = startX + (week - 1) * (chartWidth / ganttData.weeks) + (chartWidth / ganttData.weeks) / 2;
      ctx.fillText(`W${week}`, x, 30);
    }

    // Draw grid lines
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;

    for (let week = 0; week <= ganttData.weeks; week++) {
      const x = startX + week * (chartWidth / ganttData.weeks);
      ctx.beginPath();
      ctx.moveTo(x, 35);
      ctx.lineTo(x, startY + ganttData.tasks.length * barSpacing);
      ctx.stroke();
    }

    // Draw tasks
    ganttData.tasks.forEach((task, index) => {
      const y = startY + index * barSpacing;

      // Draw task name
      ctx.font = '12px Inter, sans-serif';
      ctx.fillStyle = '#1e293b';
      ctx.textAlign = 'left';
      ctx.fillText(task.name, 10, y + 15);

      // Draw assignee
      ctx.font = '10px Inter, sans-serif';
      ctx.fillStyle = '#64748b';
      ctx.fillText(`(${task.assignee})`, 10, y + 28);

      // Draw task bar
      const barX = startX + (task.startWeek - 1) * (chartWidth / ganttData.weeks);
      const barWidth = task.duration * (chartWidth / ganttData.weeks) - 2;

      // Task bar background
      ctx.fillStyle = task.color;
      ctx.fillRect(barX, y, barWidth, barHeight);

      // Task bar border
      ctx.strokeStyle = task.color;
      ctx.lineWidth = 1;
      ctx.strokeRect(barX, y, barWidth, barHeight);

      // Duration text
      ctx.font = '10px Inter, sans-serif';
      ctx.fillStyle = 'white';
      ctx.textAlign = 'center';
      if (barWidth > 30) { // Only show text if bar is wide enough
        ctx.fillText(`${task.duration}w`, barX + barWidth / 2, y + 13);
      }
    });

    // Draw legend
    const legendY = startY + ganttData.tasks.length * barSpacing + 30;
    const legendItems = [
      { color: '#3b82f6', label: 'Ivan' },
      { color: '#10b981', label: 'Husaan' },
      { color: '#f59e0b', label: 'Chris' },
      { color: '#6366f1', label: 'Daniel' },
      { color: '#81141f', label: 'Sanjay'},
      { color: '#aaaaaa', label: 'Multiple Members'}
    ];

    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'left';

    legendItems.forEach((item, index) => {
      const x = startX + index * 120;

      // Legend color box
      ctx.fillStyle = item.color;
      ctx.fillRect(x, legendY, 15, 15);

      // Legend text
      ctx.fillStyle = '#1e293b';
      ctx.fillText(item.label, x + 20, legendY + 12);
    });
  }

  // Smooth scrolling for navigation links
  function initSmoothScrolling() {
    const navLinks = document.querySelectorAll('.nav-menu a[href^="#"]');

    navLinks.forEach(link => {
      link.addEventListener('click', function (e) {
        e.preventDefault();

        const targetId = this.getAttribute('href').substring(1);
        const targetElement = document.getElementById(targetId);

        if (targetElement) {
          const offsetTop = targetElement.offsetTop - 80; // Account for fixed navbar

          window.scrollTo({
            top: offsetTop,
            behavior: 'smooth'
          });

          // Update active nav link
          navLinks.forEach(navLink => navLink.classList.remove('active'));
          this.classList.add('active');
        }
      });
    });
  }

  // Intersection Observer for active navigation highlighting
  function initScrollSpy() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-menu a[href^="#"]');

    const observerOptions = {
      root: null,
      rootMargin: '-100px 0px -50% 0px',
      threshold: 0
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const id = entry.target.getAttribute('id');

          // Remove active class from all nav links
          navLinks.forEach(link => link.classList.remove('active'));

          // Add active class to current section's nav link
          const activeLink = document.querySelector(`.nav-menu a[href="#${id}"]`);
          if (activeLink) {
            activeLink.classList.add('active');
          }
        }
      });
    }, observerOptions);

    sections.forEach(section => observer.observe(section));
  }

  // Animate cards on scroll
  function initScrollAnimations() {
    const cards = document.querySelectorAll('.dataset-card, .method-card, .model-card, .metric-category, .goal-card');

    const animationObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.animation = 'fadeInUp 0.6s ease forwards';
        }
      });
    }, {
      threshold: 0.1,
      rootMargin: '0px 0px -50px 0px'
    });

    cards.forEach(card => {
      card.style.opacity = '0';
      card.style.transform = 'translateY(20px)';
      animationObserver.observe(card);
    });
  }

  // Add CSS animation keyframes
  function addAnimationStyles() {
    const style = document.createElement('style');
    style.textContent = `
            @keyframes fadeInUp {
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .nav-menu a.active {
                color: var(--primary-color);
                background-color: var(--surface-color);
            }
            
            .card-hover-effect {
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            
            .card-hover-effect:hover {
                transform: translateY(-2px);
                box-shadow: var(--shadow-medium);
            }
        `;
    document.head.appendChild(style);
  }

  // Word count checker for proposal requirements
  function checkWordCount() {
    const sections = ['introduction', 'problem', 'methods', 'results'];
    let totalWords = 0;

    sections.forEach(sectionId => {
      const section = document.getElementById(sectionId);
      if (section) {
        const text = section.textContent || section.innerText;
        const words = text.trim().split(/\s+/).filter(word => word.length > 0);
        totalWords += words.length;
      }
    });

    // Create word count indicator
    const wordCountElement = document.createElement('div');
    wordCountElement.id = 'word-count';
    wordCountElement.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: var(--primary-color);
            color: white;
            padding: 10px 15px;
            border-radius: 25px;
            font-size: 12px;
            font-weight: 500;
            box-shadow: var(--shadow-medium);
            z-index: 1000;
        `;
    wordCountElement.textContent = `Word Count: ${totalWords}/800`;

    // Color code based on word count
    if (totalWords > 800) {
      wordCountElement.style.background = '#ef4444'; // Red if over limit
    } else if (totalWords > 700) {
      wordCountElement.style.background = '#f59e0b'; // Orange if close to limit
    }

    document.body.appendChild(wordCountElement);
  }

  // Initialize all features
  function init() {
    addAnimationStyles();
    createGanttChart();
    initSmoothScrolling();
    initScrollSpy();
    initScrollAnimations();
    checkWordCount();

    // Add hover effects to cards
    const cards = document.querySelectorAll('.dataset-card, .method-card, .model-card');
    cards.forEach(card => {
      card.classList.add('card-hover-effect');
    });

    // Console log for debugging
    console.log('CS 4641 Project Website Initialized');
    console.log(`Gantt chart created with ${ganttData.tasks.length} tasks`);
  }

  // Run initialization
  init();

  // Redraw Gantt chart on window resize
  window.addEventListener('resize', function () {
    setTimeout(createGanttChart, 100);
  });
});
