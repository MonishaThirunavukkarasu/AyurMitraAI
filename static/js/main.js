// ==============================
// AYURMEDHA MAIN JAVASCRIPT
// Complete Production Version with All Features
// ==============================

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('AyurMedha JS initialized');
    
    // Initialize all components
    initNavbar();
    initFlipCards();
    initHerbFilters();
    initSmoothScroll();
    initFormValidation();
    initScrollReveal();
    initFlashMessages();
    initCounters();
    initTooltips();
    initSearchSuggestions();
    initParallaxEffect();
    initDarkMode();
    initPredictionHistory();
    initCharts();
    initAccessibility();
    initAnimations();
    initMobileMenu();
});

// ==============================
// GLOBAL VARIABLES
// ==============================

let currentFlipCard = null;
let scrollPosition = 0;
let isMobile = window.innerWidth <= 768;

// ==============================
// NAVBAR FUNCTIONS
// ==============================

function initNavbar() {
    const navbar = document.querySelector('.navbar');
    const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
    const navRight = document.querySelector('.nav-right');
    
    if (!navbar) return;
    
    // Scroll effect
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });
    
    // Mobile menu toggle
    if (mobileMenuBtn && navRight) {
        mobileMenuBtn.addEventListener('click', () => {
            navRight.classList.toggle('active');
            mobileMenuBtn.classList.toggle('active');
            document.body.classList.toggle('menu-open');
        });
    }
    
    // Close mobile menu on link click
    const navLinks = document.querySelectorAll('.nav-right a');
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            if (window.innerWidth <= 768) {
                navRight?.classList.remove('active');
                mobileMenuBtn?.classList.remove('active');
                document.body.classList.remove('menu-open');
            }
        });
    });
    
    // Set active nav link based on current page
    const currentPath = window.location.pathname;
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentPath || (currentPath === '/' && href === '/')) {
            link.classList.add('active');
        } else if (currentPath.includes(href) && href !== '/') {
            link.classList.add('active');
        }
    });
    
    // Add mobile menu button if not exists
    if (!mobileMenuBtn && window.innerWidth <= 768) {
        const btn = document.createElement('button');
        btn.className = 'mobile-menu-btn';
        btn.innerHTML = '<span></span><span></span><span></span>';
        navbar.insertBefore(btn, navRight);
        
        btn.addEventListener('click', () => {
            navRight.classList.toggle('active');
            btn.classList.toggle('active');
            document.body.classList.toggle('menu-open');
        });
    }
}

// ==============================
// FLIP CARD FUNCTIONS
// ==============================

function initFlipCards() {
    const flipCards = document.querySelectorAll('.flip-card');
    
    if (!flipCards.length) return;
    
    flipCards.forEach((card, index) => {
        // Set unique ID for accessibility
        card.id = `herb-card-${index + 1}`;
        
        // Click handler
        card.addEventListener('click', function(e) {
            // Don't flip if clicking on links or buttons inside the card
            if (e.target.tagName === 'A' || e.target.tagName === 'BUTTON' || 
                e.target.closest('a') || e.target.closest('button')) {
                return;
            }
            
            // Close any other flipped cards first
            flipCards.forEach(otherCard => {
                if (otherCard !== this && otherCard.classList.contains('flipped')) {
                    otherCard.classList.remove('flipped');
                    otherCard.setAttribute('aria-pressed', 'false');
                }
            });
            
            // Toggle this card
            this.classList.toggle('flipped');
            
            // Update ARIA state
            const isFlipped = this.classList.contains('flipped');
            this.setAttribute('aria-pressed', isFlipped);
            
            // Track for analytics (optional)
            if (isFlipped) {
                const herbName = this.querySelector('h3')?.textContent || 'Unknown';
                console.log(`Herb flipped: ${herbName}`);
            }
        });
        
        // Keyboard accessibility
        card.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.classList.toggle('flipped');
                this.setAttribute('aria-pressed', this.classList.contains('flipped'));
            }
            
            // Close on Escape
            if (e.key === 'Escape' && this.classList.contains('flipped')) {
                this.classList.remove('flipped');
                this.setAttribute('aria-pressed', 'false');
            }
        });
        
        // Make cards focusable
        card.setAttribute('tabindex', '0');
        card.setAttribute('role', 'button');
        card.setAttribute('aria-pressed', 'false');
        card.setAttribute('aria-label', `Flip ${card.querySelector('h3')?.textContent || 'herb'} card to see details`);
        
        // Touch device optimization
        if ('ontouchstart' in window) {
            card.addEventListener('touchstart', function(e) {
                e.preventDefault();
                this.classList.toggle('flipped');
            }, { passive: true });
        }
    });
    
    // Close all flipped cards when clicking outside
    document.addEventListener('click', function(e) {
        if (!e.target.closest('.flip-card')) {
            flipCards.forEach(card => {
                if (card.classList.contains('flipped')) {
                    card.classList.remove('flipped');
                    card.setAttribute('aria-pressed', 'false');
                }
            });
        }
    });
    
    // Handle escape key globally
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            flipCards.forEach(card => {
                if (card.classList.contains('flipped')) {
                    card.classList.remove('flipped');
                    card.setAttribute('aria-pressed', 'false');
                }
            });
        }
    });
}

// ==============================
// HERB FILTER FUNCTIONS
// ==============================

function initHerbFilters() {
    const filterButtons = document.querySelectorAll('.filter-btn');
    const searchInput = document.getElementById('herbSearch');
    const herbCards = document.querySelectorAll('.flip-card');
    const filterContainer = document.querySelector('.filter-controls');
    const noResultsMessage = document.createElement('div');
    
    if (!herbCards.length) return;
    
    // Create no results message
    noResultsMessage.className = 'no-results';
    noResultsMessage.innerHTML = `
        <div style="text-align: center; padding: 60px 20px;">
            <div style="font-size: 60px; margin-bottom: 20px;">🌿</div>
            <h3 style="font-size: 24px; color: var(--primary-dark); margin-bottom: 10px;">No Herbs Found</h3>
            <p style="color: #666;">Try adjusting your search or filter criteria</p>
            <button class="btn-secondary" onclick="resetFilters()" style="margin-top: 20px;">Reset Filters</button>
        </div>
    `;
    noResultsMessage.style.display = 'none';
    filterContainer?.parentNode.insertBefore(noResultsMessage, filterContainer.nextSibling);
    
    // Filter by dosha
    if (filterButtons.length) {
        filterButtons.forEach(btn => {
            btn.addEventListener('click', function() {
                // Update active state
                filterButtons.forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                
                const filterValue = this.dataset.filter;
                filterHerbs(filterValue, searchInput?.value || '');
            });
        });
    }
    
    // Search functionality
    if (searchInput) {
        let searchTimeout;
        searchInput.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                const activeFilter = document.querySelector('.filter-btn.active')?.dataset.filter || 'all';
                filterHerbs(activeFilter, this.value.toLowerCase().trim());
            }, 300);
        });
        
        // Clear search button
        const clearBtn = document.createElement('span');
        clearBtn.className = 'search-clear';
        clearBtn.innerHTML = '×';
        clearBtn.style.cssText = `
            position: absolute;
            right: 40px;
            top: 50%;
            transform: translateY(-50%);
            cursor: pointer;
            font-size: 20px;
            color: #999;
            display: none;
        `;
        searchInput.parentNode.appendChild(clearBtn);
        
        clearBtn.addEventListener('click', () => {
            searchInput.value = '';
            clearBtn.style.display = 'none';
            const activeFilter = document.querySelector('.filter-btn.active')?.dataset.filter || 'all';
            filterHerbs(activeFilter, '');
            searchInput.focus();
        });
        
        searchInput.addEventListener('input', function() {
            clearBtn.style.display = this.value ? 'block' : 'none';
        });
    }
    
    function filterHerbs(filter, searchTerm) {
        let visibleCount = 0;
        
        herbCards.forEach(card => {
            // Get herb data
            const herbName = card.querySelector('h3')?.textContent.toLowerCase() || '';
            const sanskritName = card.querySelector('.sanskrit-name')?.textContent.toLowerCase() || '';
            const description = card.querySelector('.herb-description')?.textContent.toLowerCase() || '';
            const doshaTags = Array.from(card.querySelectorAll('.dosha-tag')).map(tag => 
                tag.textContent.toLowerCase()
            );
            
            // Check filter match
            let filterMatch = filter === 'all';
            if (!filterMatch) {
                doshaTags.forEach(tag => {
                    if (tag.includes(filter)) filterMatch = true;
                });
            }
            
            // Check search match
            let searchMatch = true;
            if (searchTerm) {
                searchMatch = herbName.includes(searchTerm) || 
                             sanskritName.includes(searchTerm) || 
                             description.includes(searchTerm);
            }
            
            // Apply filters
            if (filterMatch && searchMatch) {
                card.style.display = 'block';
                setTimeout(() => {
                    card.style.opacity = '1';
                    card.style.transform = 'scale(1)';
                }, 10);
                visibleCount++;
            } else {
                card.style.opacity = '0';
                card.style.transform = 'scale(0.8)';
                setTimeout(() => {
                    card.style.display = 'none';
                }, 300);
            }
        });
        
        // Show/hide no results message
        if (noResultsMessage) {
            noResultsMessage.style.display = visibleCount === 0 ? 'block' : 'none';
        }
        
        // Update results count
        updateResultsCount(visibleCount, herbCards.length);
    }
    
    function updateResultsCount(visible, total) {
        let countElement = document.querySelector('.results-count');
        if (!countElement) {
            countElement = document.createElement('div');
            countElement.className = 'results-count';
            countElement.style.cssText = `
                text-align: center;
                margin: 20px 0;
                color: #666;
                font-size: 14px;
            `;
            filterContainer?.parentNode.insertBefore(countElement, filterContainer.nextSibling);
        }
        countElement.textContent = `Showing ${visible} of ${total} herbs`;
    }
    
    // Reset filters function (make globally available)
    window.resetFilters = function() {
        const allBtn = document.querySelector('[data-filter="all"]');
        if (allBtn) allBtn.click();
        
        if (searchInput) {
            searchInput.value = '';
            const clearBtn = document.querySelector('.search-clear');
            if (clearBtn) clearBtn.style.display = 'none';
        }
        
        herbCards.forEach(card => {
            card.style.display = 'block';
            card.style.opacity = '1';
            card.style.transform = 'scale(1)';
        });
        
        if (noResultsMessage) {
            noResultsMessage.style.display = 'none';
        }
    };
}

// ==============================
// SMOOTH SCROLL FUNCTIONS
// ==============================

function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            
            if (href === '#') return;
            
            e.preventDefault();
            
            const target = document.querySelector(href);
            if (target) {
                const headerOffset = 80;
                const elementPosition = target.getBoundingClientRect().top;
                const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
                
                window.scrollTo({
                    top: offsetPosition,
                    behavior: 'smooth'
                });
                
                // Update URL without jumping
                history.pushState(null, null, href);
            }
        });
    });
}

// ==============================
// FORM VALIDATION FUNCTIONS
// ==============================

function initFormValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        if (form.classList.contains('no-validate')) return;
        
        form.addEventListener('submit', function(e) {
            let isValid = true;
            const requiredFields = form.querySelectorAll('[required]');
            
            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    isValid = false;
                    showFieldError(field, 'This field is required');
                } else {
                    clearFieldError(field);
                }
                
                // Email validation
                if (field.type === 'email' && field.value) {
                    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                    if (!emailRegex.test(field.value)) {
                        isValid = false;
                        showFieldError(field, 'Please enter a valid email address');
                    }
                }
                
                // Password validation
                if (field.type === 'password' && field.value) {
                    if (field.value.length < 6) {
                        isValid = false;
                        showFieldError(field, 'Password must be at least 6 characters');
                    }
                }
                
                // Number validation
                if (field.type === 'number' && field.value) {
                    const min = parseFloat(field.min) || -Infinity;
                    const max = parseFloat(field.max) || Infinity;
                    const value = parseFloat(field.value);
                    
                    if (value < min) {
                        isValid = false;
                        showFieldError(field, `Minimum value is ${min}`);
                    }
                    if (value > max) {
                        isValid = false;
                        showFieldError(field, `Maximum value is ${max}`);
                    }
                }
            });
            
            if (!isValid) {
                e.preventDefault();
                showNotification('Please fix the errors in the form', 'error');
                
                // Scroll to first error
                const firstError = form.querySelector('.field-error');
                if (firstError) {
                    firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            } else {
                showLoader();
            }
        });
        
        // Real-time validation
        form.querySelectorAll('input, textarea, select').forEach(field => {
            field.addEventListener('blur', function() {
                validateField(this);
            });
            
            field.addEventListener('input', function() {
                if (this.classList.contains('error')) {
                    validateField(this);
                }
            });
        });
    });
    
    // Auto-expand textarea
    const textareas = document.querySelectorAll('textarea');
    textareas.forEach(textarea => {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    });
}

function validateField(field) {
    if (!field.hasAttribute('required') && !field.value) return true;
    
    let isValid = true;
    let errorMessage = '';
    
    if (field.hasAttribute('required') && !field.value.trim()) {
        isValid = false;
        errorMessage = 'This field is required';
    } else if (field.type === 'email' && field.value) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(field.value)) {
            isValid = false;
            errorMessage = 'Please enter a valid email address';
        }
    } else if (field.type === 'password' && field.value && field.value.length < 6) {
        isValid = false;
        errorMessage = 'Password must be at least 6 characters';
    } else if (field.type === 'number' && field.value) {
        const min = parseFloat(field.min) || -Infinity;
        const max = parseFloat(field.max) || Infinity;
        const value = parseFloat(field.value);
        
        if (value < min) {
            isValid = false;
            errorMessage = `Minimum value is ${min}`;
        }
        if (value > max) {
            isValid = false;
            errorMessage = `Maximum value is ${max}`;
        }
    }
    
    if (isValid) {
        clearFieldError(field);
    } else {
        showFieldError(field, errorMessage);
    }
    
    return isValid;
}

function showFieldError(field, message) {
    clearFieldError(field);
    
    field.classList.add('error');
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'field-error';
    errorDiv.textContent = message;
    errorDiv.style.cssText = `
        color: #dc3545;
        font-size: 12px;
        margin-top: 5px;
        animation: slideDown 0.3s ease;
    `;
    
    field.parentNode.appendChild(errorDiv);
}

function clearFieldError(field) {
    field.classList.remove('error');
    const existingError = field.parentNode.querySelector('.field-error');
    if (existingError) {
        existingError.remove();
    }
}

// ==============================
// SCROLL REVEAL ANIMATIONS
// ==============================

function initScrollReveal() {
    const revealElements = document.querySelectorAll(
        '.feature-card, .flip-card, .dosha-card, .dashboard-card, .info-card, .result-card, .herb-card'
    );
    
    if (!revealElements.length) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
                
                // Add stagger effect for children
                const children = entry.target.children;
                Array.from(children).forEach((child, index) => {
                    child.style.transitionDelay = `${index * 0.1}s`;
                });
                
                // Unobserve after revealed
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });
    
    revealElements.forEach(element => {
        element.classList.add('reveal-init');
        observer.observe(element);
    });
    
    // Add CSS for reveal animations
    const style = document.createElement('style');
    style.textContent = `
        .reveal-init {
            opacity: 0;
            transform: translateY(30px);
            transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .reveal-init.revealed {
            opacity: 1;
            transform: translateY(0);
        }
        
        .reveal-init.revealed > * {
            transition: all 0.3s ease;
        }
    `;
    document.head.appendChild(style);
}

// ==============================
// FLASH MESSAGES FUNCTIONS
// ==============================

function initFlashMessages() {
    const flashMessages = document.querySelectorAll('.flash');
    
    flashMessages.forEach((message, index) => {
        // Set delay based on index
        const delay = 5000 + (index * 1000);
        
        setTimeout(() => {
            message.style.animation = 'slideOutRight 0.5s ease forwards';
            
            setTimeout(() => {
                message.remove();
            }, 500);
        }, delay);
        
        // Add close button
        const closeBtn = document.createElement('span');
        closeBtn.innerHTML = '×';
        closeBtn.style.cssText = `
            position: absolute;
            right: 15px;
            top: 50%;
            transform: translateY(-50%);
            cursor: pointer;
            font-size: 20px;
            opacity: 0.7;
            transition: opacity 0.3s ease;
        `;
        closeBtn.addEventListener('mouseenter', () => closeBtn.style.opacity = '1');
        closeBtn.addEventListener('mouseleave', () => closeBtn.style.opacity = '0.7');
        closeBtn.addEventListener('click', () => {
            message.style.animation = 'slideOutRight 0.3s ease forwards';
            setTimeout(() => message.remove(), 300);
        });
        
        message.style.position = 'relative';
        message.appendChild(closeBtn);
    });
}

// ==============================
// NOTIFICATION SYSTEM
// ==============================

function showNotification(message, type = 'info', duration = 5000) {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-icon">${getNotificationIcon(type)}</div>
        <div class="notification-content">${message}</div>
        <button class="notification-close">×</button>
    `;
    
    // Style the notification
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 30px;
        background: white;
        border-radius: 12px;
        box-shadow: var(--shadow-lg);
        padding: 15px 25px;
        display: flex;
        align-items: center;
        gap: 15px;
        z-index: 10000;
        min-width: 300px;
        max-width: 400px;
        animation: slideInRight 0.5s ease;
        border-left: 4px solid ${getNotificationColor(type)};
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after duration
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.5s ease forwards';
        setTimeout(() => notification.remove(), 500);
    }, duration);
    
    // Close button handler
    notification.querySelector('.notification-close').addEventListener('click', () => {
        notification.style.animation = 'slideOutRight 0.3s ease forwards';
        setTimeout(() => notification.remove(), 300);
    });
}

function getNotificationIcon(type) {
    switch(type) {
        case 'success': return '✓';
        case 'error': return '✗';
        case 'warning': return '⚠';
        default: return 'ℹ';
    }
}

function getNotificationColor(type) {
    switch(type) {
        case 'success': return '#28a745';
        case 'error': return '#dc3545';
        case 'warning': return '#ffc107';
        default: return '#17a2b8';
    }
}

// Make showNotification globally available
window.showNotification = showNotification;

// ==============================
// LOADER FUNCTIONS
// ==============================

function showLoader() {
    let loader = document.querySelector('.loader');
    
    if (!loader) {
        loader = document.createElement('div');
        loader.className = 'loader';
        loader.innerHTML = '<div class="spinner"></div>';
        document.body.appendChild(loader);
    }
    
    loader.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function hideLoader() {
    const loader = document.querySelector('.loader');
    if (loader) {
        loader.classList.remove('active');
        document.body.style.overflow = '';
    }
}

// Make loader functions globally available
window.showLoader = showLoader;
window.hideLoader = hideLoader;

// ==============================
// COUNTER ANIMATIONS
// ==============================

function initCounters() {
    const counters = document.querySelectorAll('.counter');
    
    counters.forEach(counter => {
        const target = parseInt(counter.dataset.target) || 0;
        const duration = parseInt(counter.dataset.duration) || 2000;
        const increment = target / (duration / 16); // 60fps
        
        let current = 0;
        
        const updateCounter = () => {
            current += increment;
            if (current < target) {
                counter.textContent = Math.round(current);
                requestAnimationFrame(updateCounter);
            } else {
                counter.textContent = target;
            }
        };
        
        // Start counter when element is in view
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    updateCounter();
                    observer.unobserve(entry.target);
                }
            });
        });
        
        observer.observe(counter);
    });
}

// ==============================
// TOOLTIP FUNCTIONS
// ==============================

function initTooltips() {
    const tooltips = document.querySelectorAll('[data-tooltip]');
    
    tooltips.forEach(element => {
        let tooltipElement = null;
        
        element.addEventListener('mouseenter', (e) => {
            const text = element.dataset.tooltip;
            
            tooltipElement = document.createElement('div');
            tooltipElement.className = 'custom-tooltip';
            tooltipElement.textContent = text;
            
            // Style tooltip
            tooltipElement.style.cssText = `
                position: absolute;
                background: var(--text-dark);
                color: white;
                padding: 8px 15px;
                border-radius: 6px;
                font-size: 12px;
                z-index: 10000;
                pointer-events: none;
                white-space: nowrap;
                animation: fadeIn 0.2s ease;
                box-shadow: var(--shadow-md);
            `;
            
            document.body.appendChild(tooltipElement);
            
            const rect = element.getBoundingClientRect();
            const tooltipRect = tooltipElement.getBoundingClientRect();
            
            tooltipElement.style.top = rect.bottom + window.scrollY + 10 + 'px';
            tooltipElement.style.left = rect.left + window.scrollX + (rect.width / 2) - (tooltipRect.width / 2) + 'px';
        });
        
        element.addEventListener('mouseleave', () => {
            if (tooltipElement) {
                tooltipElement.remove();
                tooltipElement = null;
            }
        });
    });
}

// ==============================
// SEARCH SUGGESTIONS
// ==============================

function initSearchSuggestions() {
    const searchInput = document.getElementById('herbSearch');
    if (!searchInput) return;
    
    const suggestionsBox = document.createElement('div');
    suggestionsBox.className = 'search-suggestions';
    suggestionsBox.style.cssText = `
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        background: white;
        border-radius: 8px;
        box-shadow: var(--shadow-lg);
        margin-top: 5px;
        max-height: 300px;
        overflow-y: auto;
        display: none;
        z-index: 1000;
    `;
    
    searchInput.parentNode.style.position = 'relative';
    searchInput.parentNode.appendChild(suggestionsBox);
    
    // Common herb names for suggestions
        const herbSuggestions = [
        'Turmeric', 'Ashwagandha', 'Neem', 'Tulsi', 'Guduchi', 'Guggulu',
        'Tripala', 'Brahmi', 'Amla', 'Arjuna', 'Shatavari', 'Fenugreek',
        'Ginger', 'Aloe Vera', 'Bhringraj', 'Gokshura'
    ];
    
    searchInput.addEventListener('input', function() {
        const value = this.value.toLowerCase().trim();
        
        if (value.length < 2) {
            suggestionsBox.style.display = 'none';
            return;
        }
        
        const matches = herbSuggestions.filter(herb => 
            herb.toLowerCase().includes(value)
        );
        
        if (matches.length > 0) {
            suggestionsBox.innerHTML = '';
            matches.forEach(match => {
                const item = document.createElement('div');
                item.textContent = match;
                item.style.cssText = `
                    padding: 12px 15px;
                    cursor: pointer;
                    border-bottom: 1px solid #f0f0f0;
                    transition: background 0.2s ease;
                `;
                
                item.addEventListener('mouseenter', () => {
                    item.style.background = '#f5f5f5';
                });
                
                item.addEventListener('mouseleave', () => {
                    item.style.background = 'white';
                });
                
                item.addEventListener('click', () => {
                    searchInput.value = match;
                    suggestionsBox.style.display = 'none';
                    
                    // Trigger search
                    const event = new Event('input', { bubbles: true });
                    searchInput.dispatchEvent(event);
                });
                
                suggestionsBox.appendChild(item);
            });
            
            suggestionsBox.style.display = 'block';
        } else {
            suggestionsBox.style.display = 'none';
        }
    });
    
    // Hide suggestions when clicking outside
    document.addEventListener('click', (e) => {
        if (!searchInput.parentNode.contains(e.target)) {
            suggestionsBox.style.display = 'none';
        }
    });
}

// ==============================
// PARALLAX EFFECT
// ==============================

function initParallaxEffect() {
    const hero = document.querySelector('.hero');
    const heroImg = hero?.querySelector('img');
    
    if (!hero || !heroImg) return;
    
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const rate = scrolled * 0.5;
        
        heroImg.style.transform = `translateY(${rate}px) scale(1.1)`;
    });
}

// ==============================
// DARK MODE TOGGLE
// ==============================

function initDarkMode() {
    const darkModeToggle = document.querySelector('#dark-mode-toggle');
    
    if (!darkModeToggle) return;
    
    // Check for saved preference
    const savedMode = localStorage.getItem('darkMode');
    if (savedMode === 'true') {
        document.body.classList.add('dark-mode');
        updateDarkModeIcon(true);
    }
    
    darkModeToggle.addEventListener('click', () => {
        const isDark = document.body.classList.toggle('dark-mode');
        localStorage.setItem('darkMode', isDark);
        updateDarkModeIcon(isDark);
        
        showNotification(`${isDark ? 'Dark' : 'Light'} mode activated`, 'info');
    });
}

function updateDarkModeIcon(isDark) {
    const toggle = document.querySelector('#dark-mode-toggle');
    if (toggle) {
        toggle.innerHTML = isDark ? '☀️' : '🌙';
    }
}

// ==============================
// PREDICTION HISTORY CHARTS
// ==============================

function initPredictionHistory() {
    const historyCanvas = document.getElementById('historyChart');
    if (!historyCanvas) return;
    
    // Fetch history data from API
    fetch('/api/prediction-history')
        .then(response => response.json())
        .then(data => {
            if (data.length > 0) {
                createHistoryChart(data);
            }
        })
        .catch(error => console.error('Error loading history:', error));
}

function createHistoryChart(data) {
    const ctx = document.getElementById('historyChart').getContext('2d');
    
    // Simple chart using canvas
    const maxConfidence = Math.max(...data.map(d => d.confidence));
    const chartHeight = 200;
    const barWidth = 30;
    const spacing = 10;
    
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    
    data.slice(0, 10).forEach((item, index) => {
        const x = index * (barWidth + spacing) + 30;
        const height = (item.confidence / maxConfidence) * chartHeight;
        const y = ctx.canvas.height - height - 30;
        
        // Draw bar
        ctx.fillStyle = 'var(--primary)';
        ctx.fillRect(x, y, barWidth, height);
        
        // Draw confidence text
        ctx.fillStyle = 'var(--text-dark)';
        ctx.font = '10px Montserrat';
        ctx.textAlign = 'center';
        ctx.fillText(item.confidence + '%', x + barWidth/2, y - 5);
        
        // Draw date (shortened)
        const date = new Date(item.timestamp).toLocaleDateString();
        ctx.fillText(date, x + barWidth/2, ctx.canvas.height - 10);
    });
}

// ==============================
// ACCESSIBILITY FUNCTIONS
// ==============================

function initAccessibility() {
    // Add skip to content link
    const skipLink = document.createElement('a');
    skipLink.href = '#main-content';
    skipLink.textContent = 'Skip to main content';
    skipLink.style.cssText = `
        position: absolute;
        top: -40px;
        left: 0;
        background: var(--primary);
        color: white;
        padding: 8px 15px;
        text-decoration: none;
        z-index: 10000;
        transition: top 0.3s ease;
    `;
    
    skipLink.addEventListener('focus', () => {
        skipLink.style.top = '0';
    });
    
    skipLink.addEventListener('blur', () => {
        skipLink.style.top = '-40px';
    });
    
    document.body.insertBefore(skipLink, document.body.firstChild);
    
    // Add ARIA labels where missing
    document.querySelectorAll('button').forEach(btn => {
        if (!btn.getAttribute('aria-label') && !btn.textContent.trim()) {
            btn.setAttribute('aria-label', 'Button');
        }
    });
    
    document.querySelectorAll('img').forEach(img => {
        if (!img.alt) {
            img.alt = img.title || 'Image';
        }
    });
}

// ==============================
// ANIMATION FUNCTIONS
// ==============================

function initAnimations() {
    // Add hover animations to cards
    const cards = document.querySelectorAll('.feature-card, .dashboard-card, .dosha-card');
    
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
    
    // Add pulse animation to buttons
    const buttons = document.querySelectorAll('.btn-primary, .btn-secondary');
    
    buttons.forEach(btn => {
        btn.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.05)';
        });
        
        btn.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
        });
    });
    
    // Typing animation for hero text (optional)
    const heroText = document.querySelector('.hero-text h1');
    if (heroText && heroText.dataset.animate) {
        const text = heroText.textContent;
        heroText.textContent = '';
        
        let i = 0;
        const typeInterval = setInterval(() => {
            if (i < text.length) {
                heroText.textContent += text.charAt(i);
                i++;
            } else {
                clearInterval(typeInterval);
            }
        }, 100);
    }
}

// ==============================
// MOBILE MENU FUNCTIONS
// ==============================

function initMobileMenu() {
    const menuBtn = document.querySelector('.mobile-menu-btn');
    const navRight = document.querySelector('.nav-right');
    
    if (!menuBtn || !navRight) return;
    
    // Close menu when window is resized above mobile breakpoint
    window.addEventListener('resize', () => {
        if (window.innerWidth > 768) {
            navRight.classList.remove('active');
            menuBtn.classList.remove('active');
            document.body.classList.remove('menu-open');
        }
        
        // Update mobile flag
        isMobile = window.innerWidth <= 768;
    });
    
    // Prevent body scroll when menu is open
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && navRight.classList.contains('active')) {
            navRight.classList.remove('active');
            menuBtn.classList.remove('active');
            document.body.classList.remove('menu-open');
        }
    });
}

// ==============================
// RESIZE HANDLER
// ==============================

window.addEventListener('resize', () => {
    // Update mobile flag
    isMobile = window.innerWidth <= 768;
    
    // Reset any inline styles that might be problematic
    if (window.innerWidth > 768) {
        document.querySelectorAll('.flip-card').forEach(card => {
            card.style.height = '';
        });
    }
});

// ==============================
// ERROR HANDLING
// ==============================

window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
    
    // Show user-friendly error message
    if (!e.target.closest('.no-error-handling')) {
        showNotification('An error occurred. Please try again.', 'error');
    }
});

// ==============================
// PERFORMANCE OPTIMIZATIONS
// ==============================

// Debounce function for performance
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle function for performance
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Optimize scroll events
window.addEventListener('scroll', throttle(() => {
    // Handle scroll-based updates
}, 100), { passive: true });

// ==============================
// CLEANUP FUNCTIONS
// ==============================

// Remove any event listeners if needed
window.addEventListener('beforeunload', () => {
    // Clean up any resources
});

// ==============================
// INITIALIZATION COMPLETE
// ==============================

console.log('AyurMedha JS fully initialized');
hideLoader(); // Hide loader if it's showing
