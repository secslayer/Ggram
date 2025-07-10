/**
 * Ggram - Main JavaScript File
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Ggram application loaded');
    
    // Initialize any interactive elements
    initializeApp();
});

/**
 * Initialize the application
 */
function initializeApp() {
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Add event listeners for API documentation link
    const apiDocsLink = document.querySelector('a[href="/api/docs"]');
    if (apiDocsLink) {
        apiDocsLink.addEventListener('click', function(e) {
            console.log('Navigating to API documentation');
        });
    }
    
    // Add event listeners for User Guide link
    const userGuideLink = document.querySelector('a[href="/static/docs/user-guide.html"]');
    if (userGuideLink) {
        userGuideLink.addEventListener('click', function(e) {
            // If the user guide doesn't exist yet, prevent navigation and show a message
            if (!doesUserGuideExist()) {
                e.preventDefault();
                alert('User guide is coming soon!');
            }
        });
    }
}

/**
 * Check if the user guide exists
 * This is a placeholder function that would normally check if the file exists
 */
function doesUserGuideExist() {
    // The user guide now exists, so we return true
    return true;
}

/**
 * Example function to make API requests
 * @param {string} endpoint - The API endpoint to call
 * @param {string} method - The HTTP method to use
 * @param {Object} data - The data to send with the request
 * @returns {Promise} - A promise that resolves with the API response
 */
async function makeApiRequest(endpoint, method = 'GET', data = null) {
    try {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        };
        
        if (data && (method === 'POST' || method === 'PUT' || method === 'PATCH')) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(`/api/${endpoint}`, options);
        
        if (!response.ok) {
            throw new Error(`API request failed with status ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API request error:', error);
        throw error;
    }
}