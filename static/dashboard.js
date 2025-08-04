// Dashboard JavaScript
class SurveillanceDashboard {
    constructor() {
        this.websocket = null;
        this.isRecording = false;
        this.charts = {};
        this.eventCount = 0;
        
        this.init();
    }
    
    init() {
        this.setupWebSocket();
        this.setupEventListeners();
        this.updateCurrentTime();
        this.loadInitialData();
        this.setupVideoFeed();
        
        // Update time every second
        setInterval(() => this.updateCurrentTime(), 1000);
        
        // Refresh data every 30 seconds
        setInterval(() => this.refreshData(), 30000);
    }
    
    setupVideoFeed() {
        const videoFeed = document.getElementById('video-feed');
        if (videoFeed) {
            // Set up continuous video feed refresh
            this.refreshVideoFeed();
            
            // Refresh video feed every 100ms (~10 FPS) to create smooth video effect
            setInterval(() => this.refreshVideoFeed(), 100);
            
            // Handle image load errors
            videoFeed.onerror = () => {
                console.warn('Video feed image failed to load');
                // Try again after a short delay
                setTimeout(() => this.refreshVideoFeed(), 1000);
            };
        }
    }
    
    refreshVideoFeed() {
        const videoFeed = document.getElementById('video-feed');
        if (videoFeed) {
            // Add timestamp parameter to prevent caching and ensure fresh frames
            const timestamp = new Date().getTime();
            videoFeed.src = `/video_feed?t=${timestamp}`;
        }
    }
    
    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
            this.showNotification('Connected to surveillance system', 'success');
        };
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
            this.showNotification('Disconnected from surveillance system', 'warning');
            
            // Attempt to reconnect after 5 seconds
            setTimeout(() => this.setupWebSocket(), 5000);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showNotification('Connection error', 'danger');
        };
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'event':
                this.handleNewEvent(data.data);
                break;
            case 'voice_query':
                this.handleVoiceQuery(data.data);
                break;
            case 'system_status':
                this.updateSystemStatus(data.data);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    handleNewEvent(eventData) {
        // Show event alert
        this.showEventAlert(eventData);
        
        // Add to recent events list
        this.addEventToList(eventData);
        
        // Update event overlay on video
        this.updateEventOverlay(eventData);
        
        // Update statistics
        this.eventCount++;
        this.updateEventCounter();
        
        // Play alert sound
        this.playAlertSound();
    }
    
    handleVoiceQuery(queryData) {
        // Display voice query result
        this.displayQueryResult(queryData.result, queryData.query);
        
        this.showNotification(`Voice query processed: "${queryData.query}"`, 'info');
    }
    
    setupEventListeners() {
        // Query button
        document.getElementById('query-btn').addEventListener('click', () => {
            this.processQuery();
        });
        
        // Query input enter key
        document.getElementById('query-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.processQuery();
            }
        });
        
        // Voice button
        document.getElementById('voice-btn').addEventListener('click', () => {
            this.toggleVoiceRecording();
        });
        
        // Fullscreen button
        document.getElementById('fullscreen-btn').addEventListener('click', () => {
            this.toggleFullscreen();
        });
        
        // Snapshot button
        document.getElementById('snapshot-btn').addEventListener('click', () => {
            this.takeSnapshot();
        });
        
        // Refresh events button
        document.getElementById('refresh-events-btn').addEventListener('click', () => {
            this.refreshEvents();
        });
        
        // Test notifications button
        document.getElementById('test-notifications-btn').addEventListener('click', () => {
            this.testNotifications();
        });
        
        // View statistics button
        document.getElementById('view-statistics-btn').addEventListener('click', () => {
            this.showStatistics();
        });
        
        // Export events button
        document.getElementById('export-events-btn').addEventListener('click', () => {
            this.exportEvents();
        });
    }
    
    async processQuery() {
        const queryInput = document.getElementById('query-input');
        const query = queryInput.value.trim();
        
        if (!query) {
            this.showNotification('Please enter a query', 'warning');
            return;
        }
        
        // Show loading state
        const queryBtn = document.getElementById('query-btn');
        const originalText = queryBtn.innerHTML;
        queryBtn.innerHTML = '<span class="loading-spinner"></span> Processing...';
        queryBtn.disabled = true;
        
        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query })
            });
            
            const result = await response.json();
            this.displayQueryResult(result, query);
            
            // Clear input
            queryInput.value = '';
            
        } catch (error) {
            console.error('Query error:', error);
            this.showNotification('Error processing query', 'danger');
        } finally {
            // Restore button state
            queryBtn.innerHTML = originalText;
            queryBtn.disabled = false;
        }
    }
    
    displayQueryResult(result, query) {
        const resultsContainer = document.getElementById('query-results');
        
        if (result.total_count === 0) {
            resultsContainer.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle"></i>
                    No events found for query: "${query}"
                </div>
            `;
            return;
        }
        
        let html = `
            <div class="alert alert-success">
                <i class="fas fa-check-circle"></i>
                Found ${result.total_count} events for: "${query}"
            </div>
        `;
        
        result.results.forEach(event => {
            const eventTypeIcon = this.getEventTypeIcon(event.event_type);
            const timestamp = new Date(event.timestamp).toLocaleString();
            
            html += `
                <div class="query-result-item">
                    <div class="query-result-type">
                        ${eventTypeIcon} ${event.event_type.replace('_', ' ').toUpperCase()}
                    </div>
                    <div class="query-result-description">${event.description}</div>
                    <div class="query-result-meta">
                        <span><i class="fas fa-clock"></i> ${timestamp}</span>
                        <span><i class="fas fa-percentage"></i> ${(event.confidence_score * 100).toFixed(1)}%</span>
                    </div>
                </div>
            `;
        });
        
        resultsContainer.innerHTML = html;
    }
    
    getEventTypeIcon(eventType) {
        const icons = {
            'fall': '<i class="fas fa-user-injured text-danger"></i>',
            'weapon_detected': '<i class="fas fa-exclamation-triangle text-danger"></i>',
            'vehicle_crash': '<i class="fas fa-car-crash text-warning"></i>',
            'suspicious_activity': '<i class="fas fa-eye text-info"></i>'
        };
        return icons[eventType] || '<i class="fas fa-exclamation-circle text-secondary"></i>';
    }
    
    async toggleVoiceRecording() {
        const voiceBtn = document.getElementById('voice-btn');
        
        if (!this.isRecording) {
            // Start voice recording
            try {
                const response = await fetch('/voice/start', { method: 'POST' });
                const result = await response.json();
                
                if (result.success) {
                    this.isRecording = true;
                    voiceBtn.classList.add('voice-recording');
                    voiceBtn.innerHTML = '<i class="fas fa-stop"></i> Stop';
                    this.showNotification('Voice interface started - say "surveillance" to begin', 'info');
                } else {
                    this.showNotification('Failed to start voice interface', 'danger');
                }
            } catch (error) {
                console.error('Voice start error:', error);
                this.showNotification('Error starting voice interface', 'danger');
            }
        } else {
            // Stop voice recording
            try {
                await fetch('/voice/stop', { method: 'POST' });
                this.isRecording = false;
                voiceBtn.classList.remove('voice-recording');
                voiceBtn.innerHTML = '<i class="fas fa-microphone"></i> Voice';
                this.showNotification('Voice interface stopped', 'info');
            } catch (error) {
                console.error('Voice stop error:', error);
                this.showNotification('Error stopping voice interface', 'danger');
            }
        }
    }
    
    toggleFullscreen() {
        const videoContainer = document.querySelector('.video-container');
        
        if (!document.fullscreenElement) {
            videoContainer.requestFullscreen().catch(err => {
                console.error('Error attempting to enable fullscreen:', err);
            });
        } else {
            document.exitFullscreen();
        }
    }
    
    takeSnapshot() {
        const videoFeed = document.getElementById('video-feed');
        
        // Create canvas to capture current frame
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = videoFeed.naturalWidth || videoFeed.width;
        canvas.height = videoFeed.naturalHeight || videoFeed.height;
        
        ctx.drawImage(videoFeed, 0, 0);
        
        // Download the snapshot
        canvas.toBlob(blob => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `snapshot_${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            this.showNotification('Snapshot saved', 'success');
        });
    }
    
    async refreshEvents() {
        const refreshBtn = document.getElementById('refresh-events-btn');
        const originalHTML = refreshBtn.innerHTML;
        
        refreshBtn.innerHTML = '<span class="loading-spinner"></span>';
        refreshBtn.disabled = true;
        
        try {
            const response = await fetch('/events?limit=10');
            const data = await response.json();
            
            this.updateEventsList(data.events);
            this.showNotification('Events refreshed', 'success');
            
        } catch (error) {
            console.error('Refresh error:', error);
            this.showNotification('Error refreshing events', 'danger');
        } finally {
            refreshBtn.innerHTML = originalHTML;
            refreshBtn.disabled = false;
        }
    }
    
    async testNotifications() {
        const testBtn = document.getElementById('test-notifications-btn');
        const originalHTML = testBtn.innerHTML;
        
        testBtn.innerHTML = '<span class="loading-spinner"></span> Testing...';
        testBtn.disabled = true;
        
        try {
            const response = await fetch('/test_notifications', { method: 'POST' });
            const results = await response.json();
            
            let message = 'Notification test results:\n';
            message += `SMS: ${results.sms ? '✓' : '✗'}\n`;
            message += `Email: ${results.email ? '✓' : '✗'}\n`;
            message += `Push: ${results.push ? '✓' : '✗'}`;
            
            this.showNotification(message, 'info');
            
        } catch (error) {
            console.error('Test notifications error:', error);
            this.showNotification('Error testing notifications', 'danger');
        } finally {
            testBtn.innerHTML = originalHTML;
            testBtn.disabled = false;
        }
    }
    
    async showStatistics() {
        const modal = new bootstrap.Modal(document.getElementById('statisticsModal'));
        const modalBody = document.getElementById('statistics-modal-body');
        
        // Show loading
        modalBody.innerHTML = `
            <div class="text-center">
                <div class="spinner-border" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">Loading statistics...</p>
            </div>
        `;
        
        modal.show();
        
        try {
            const response = await fetch('/statistics');
            const stats = await response.json();
            
            this.renderStatistics(stats, modalBody);
            
        } catch (error) {
            console.error('Statistics error:', error);
            modalBody.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle"></i>
                    Error loading statistics
                </div>
            `;
        }
    }
    
    renderStatistics(stats, container) {
        const html = `
            <div class="row">
                <div class="col-md-6">
                    <div class="card analytics-card mb-3">
                        <div class="card-body text-center">
                            <div class="stat-number">${stats.total_events}</div>
                            <div class="stat-label">Total Events</div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card analytics-card mb-3">
                        <div class="card-body text-center">
                            <div class="stat-number">${Object.keys(stats.event_counts_by_type).length}</div>
                            <div class="stat-label">Event Types</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0">Events by Type</h6>
                        </div>
                        <div class="card-body">
                            <canvas id="eventTypeChart"></canvas>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0">Daily Events (Last Week)</h6>
                        </div>
                        <div class="card-body">
                            <canvas id="dailyEventsChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-3">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0">Event Type Breakdown</h6>
                        </div>
                        <div class="card-body">
                            ${this.renderEventTypeBreakdown(stats.event_counts_by_type)}
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        container.innerHTML = html;
        
        // Render charts
        setTimeout(() => {
            this.renderEventTypeChart(stats.event_counts_by_type);
            this.renderDailyEventsChart(stats.daily_events_last_week);
        }, 100);
    }
    
    renderEventTypeBreakdown(eventCounts) {
        let html = '';
        const total = Object.values(eventCounts).reduce((sum, count) => sum + count, 0);
        
        for (const [eventType, count] of Object.entries(eventCounts)) {
            const percentage = ((count / total) * 100).toFixed(1);
            const icon = this.getEventTypeIcon(eventType);
            
            html += `
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span>${icon} ${eventType.replace('_', ' ').toUpperCase()}</span>
                    <span>
                        <span class="badge bg-primary">${count}</span>
                        <small class="text-muted">(${percentage}%)</small>
                    </span>
                </div>
            `;
        }
        
        return html || '<p class="text-muted">No events recorded</p>';
    }
    
    renderEventTypeChart(eventCounts) {
        const ctx = document.getElementById('eventTypeChart');
        if (!ctx) return;
        
        const labels = Object.keys(eventCounts).map(type => 
            type.replace('_', ' ').toUpperCase()
        );
        const data = Object.values(eventCounts);
        const colors = [
            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0',
            '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF'
        ];
        
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: colors.slice(0, labels.length),
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    renderDailyEventsChart(dailyEvents) {
        const ctx = document.getElementById('dailyEventsChart');
        if (!ctx) return;
        
        const labels = Object.keys(dailyEvents).map(date => {
            const d = new Date(date);
            return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        });
        const data = Object.values(dailyEvents);
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Events',
                    data: data,
                    borderColor: '#36A2EB',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    async exportEvents() {
        try {
            const response = await fetch('/events?limit=1000');
            const data = await response.json();
            
            // Convert to CSV
            const csv = this.convertToCSV(data.events);
            
            // Download CSV file
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `surveillance_events_${new Date().toISOString().split('T')[0]}.csv`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            this.showNotification('Events exported successfully', 'success');
            
        } catch (error) {
            console.error('Export error:', error);
            this.showNotification('Error exporting events', 'danger');
        }
    }
    
    convertToCSV(events) {
        if (events.length === 0) return '';
        
        const headers = ['ID', 'Timestamp', 'Event Type', 'Description', 'Confidence Score'];
        const rows = events.map(event => [
            event.id,
            event.timestamp,
            event.event_type,
            `"${event.description.replace(/"/g, '""')}"`,
            event.confidence_score
        ]);
        
        return [headers, ...rows].map(row => row.join(',')).join('\n');
    }
    
    showEventAlert(eventData) {
        const modal = new bootstrap.Modal(document.getElementById('eventModal'));
        const modalBody = document.getElementById('event-modal-body');
        
        const eventTypeIcon = this.getEventTypeIcon(eventData.event_type);
        const timestamp = new Date(eventData.timestamp).toLocaleString();
        
        modalBody.innerHTML = `
            <div class="alert alert-warning alert-live">
                <h5>${eventTypeIcon} ${eventData.event_type.replace('_', ' ').toUpperCase()}</h5>
                <p><strong>Description:</strong> ${eventData.description}</p>
                <p><strong>Time:</strong> ${timestamp}</p>
                <p><strong>Confidence:</strong> ${(eventData.confidence * 100).toFixed(1)}%</p>
            </div>
        `;
        
        modal.show();
        
        // Auto-hide after 10 seconds
        setTimeout(() => modal.hide(), 10000);
    }
    
    addEventToList(eventData) {
        const eventsList = document.getElementById('recent-events');
        const eventTypeIcon = this.getEventTypeIcon(eventData.event_type);
        const timestamp = new Date(eventData.timestamp).toLocaleString();
        
        const eventItem = document.createElement('div');
        eventItem.className = 'event-item';
        eventItem.innerHTML = `
            <div class="event-type">
                ${eventTypeIcon} ${eventData.event_type.replace('_', ' ').toUpperCase()}
            </div>
            <div class="event-description">${eventData.description}</div>
            <div class="event-time">${timestamp}</div>
            <div class="event-confidence">Confidence: ${(eventData.confidence * 100).toFixed(1)}%</div>
        `;
        
        // Add to top of list
        eventsList.insertBefore(eventItem, eventsList.firstChild);
        
        // Remove old events (keep only 10)
        while (eventsList.children.length > 10) {
            eventsList.removeChild(eventsList.lastChild);
        }
        
        // Highlight new event
        eventItem.classList.add('success-flash');
        setTimeout(() => eventItem.classList.remove('success-flash'), 500);
    }
    
    updateEventOverlay(eventData) {
        const overlay = document.getElementById('event-overlay');
        
        if (eventData.bbox) {
            const [x1, y1, x2, y2] = eventData.bbox;
            const videoFeed = document.getElementById('video-feed');
            const rect = videoFeed.getBoundingClientRect();
            
            // Calculate relative positions
            const left = (x1 / videoFeed.naturalWidth) * 100;
            const top = (y1 / videoFeed.naturalHeight) * 100;
            const width = ((x2 - x1) / videoFeed.naturalWidth) * 100;
            const height = ((y2 - y1) / videoFeed.naturalHeight) * 100;
            
            const bbox = document.createElement('div');
            bbox.className = 'event-bbox';
            bbox.style.left = `${left}%`;
            bbox.style.top = `${top}%`;
            bbox.style.width = `${width}%`;
            bbox.style.height = `${height}%`;
            
            const label = document.createElement('div');
            label.className = 'event-label';
            label.textContent = eventData.event_type.replace('_', ' ').toUpperCase();
            bbox.appendChild(label);
            
            overlay.appendChild(bbox);
            
            // Remove after 5 seconds
            setTimeout(() => {
                if (bbox.parentNode) {
                    bbox.parentNode.removeChild(bbox);
                }
            }, 5000);
        }
    }
    
    updateEventCounter() {
        // Update any event counters in the UI
        const counters = document.querySelectorAll('.event-counter');
        counters.forEach(counter => {
            counter.textContent = this.eventCount;
        });
    }
    
    playAlertSound() {
        // Create and play alert sound
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
        oscillator.frequency.setValueAtTime(600, audioContext.currentTime + 0.1);
        
        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.5);
    }
    
    updateCurrentTime() {
        const timeElement = document.getElementById('current-time');
        if (timeElement) {
            timeElement.textContent = new Date().toLocaleString();
        }
    }
    
    updateEventsList(events) {
        const eventsList = document.getElementById('recent-events');
        eventsList.innerHTML = '';
        
        events.forEach(event => {
            const eventTypeIcon = this.getEventTypeIcon(event.event_type);
            const timestamp = new Date(event.timestamp).toLocaleString();
            
            const eventItem = document.createElement('div');
            eventItem.className = 'event-item';
            eventItem.innerHTML = `
                <div class="event-type">
                    ${eventTypeIcon} ${event.event_type.replace('_', ' ').toUpperCase()}
                </div>
                <div class="event-description">${event.description}</div>
                <div class="event-time">${timestamp}</div>
                <div class="event-confidence">Confidence: ${(event.confidence_score * 100).toFixed(1)}%</div>
            `;
            
            eventsList.appendChild(eventItem);
        });
    }
    
    updateSystemStatus(status) {
        // Update system status indicators
        const indicators = {
            'video-status': status.video_capture,
            'detection-status': status.event_detection,
            'voice-status': status.voice_interface,
            'notification-status': status.notifications
        };
        
        for (const [elementId, isActive] of Object.entries(indicators)) {
            const element = document.getElementById(elementId);
            if (element) {
                const icon = element.querySelector('i');
                if (isActive) {
                    icon.className = 'fas fa-circle text-success';
                    element.innerHTML = icon.outerHTML + ' Active';
                } else {
                    icon.className = 'fas fa-circle text-danger';
                    element.innerHTML = icon.outerHTML + ' Inactive';
                }
            }
        }
    }
    
    showNotification(message, type = 'info') {
        // Create toast notification
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        // Add to toast container (create if doesn't exist)
        let toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            document.body.appendChild(toastContainer);
        }
        
        toastContainer.appendChild(toast);
        
        // Show toast
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        // Remove from DOM after hiding
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }
    
    async loadInitialData() {
        try {
            // Load recent events
            const eventsResponse = await fetch('/events?limit=10');
            const eventsData = await eventsResponse.json();
            this.updateEventsList(eventsData.events);
            
            // Load system status
            const statusResponse = await fetch('/system_status');
            const statusData = await statusResponse.json();
            this.updateSystemStatus(statusData);
            
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showNotification('Error loading initial data', 'warning');
        }
    }
    
    refreshData() {
        this.loadInitialData();
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new SurveillanceDashboard();
});
