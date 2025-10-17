function showTab(index) {
    document.querySelectorAll('.tab').forEach((tab, i) => {
        tab.classList.toggle('active', i === index);
    });
    document.querySelectorAll('.content').forEach((content, i) => {
        content.classList.toggle('active', i === index);
    });
}

// Client-side caches for the dashboard views; data is refreshed every 15 seconds.
let shipmentsData = [];
let defaultShipmentDateBounds = { min: null, max: null };
let materialsData = [];
let activeAlerts = [];
let isPollingShipments = false;
let recommendationsData = [];

function formatTons(value) {
    const normalized = Number(value) || 0;
    return `${normalized.toLocaleString()} tons`;
}

function classifyMaterialByBin(material) {
    const fillPercentage = Math.max(0, Math.min(100, Math.round((material.fill_ratio || 0) * 100)));
    if (material.needs_reorder) {
        return { label: 'Reorder Needed', tone: 'red', fillPercentage };
    }
    if (fillPercentage >= 80) {
        return { label: 'Healthy Stock', tone: 'green', fillPercentage };
    }
    return { label: 'Moderate Stock', tone: 'yellow', fillPercentage };
}

function getCurrentDate() {
    const today = new Date();
    return today.toISOString().split('T')[0];
}

function setDateInputs(start, end) {
    const startInput = document.getElementById('startDate');
    const endInput = document.getElementById('endDate');
    const rangeInput = document.getElementById('dateRange');

    startInput.value = start;
    endInput.value = end;

    if (start === end) {
        rangeInput.value = start;
    } else {
        rangeInput.value = `${start} to ${end}`;
    }

    resizeDateInput();
}

async function loadShipments() {
    if (isPollingShipments) {
        return;
    }
    isPollingShipments = true;

    try {
        const [deliveriesResponse, materialsResponse, alertsResponse, recommendationsResponse] = await Promise.all([
            // In production, these calls are fed by live MES/TMS integrations.
            fetch('http://localhost:8000/api/deliveries'),
            fetch('http://localhost:8000/api/materials'),
            fetch('http://localhost:8000/api/alerts'),
            fetch('http://localhost:8000/api/recommendations?days=7')
        ]);

        if (!deliveriesResponse.ok) {
            throw new Error(`Failed to load deliveries: ${deliveriesResponse.status}`);
        }

        const deliveries = await deliveriesResponse.json();
        materialsData = materialsResponse.ok ? await materialsResponse.json() : [];
        activeAlerts = alertsResponse.ok ? await alertsResponse.json() : [];
        recommendationsData = recommendationsResponse.ok ? await recommendationsResponse.json() : [];

        renderMaterials();
        renderAlerts();
        renderRecommendations();

        const materialMap = new Map(materialsData.map(material => [material.id, material.type]));

        shipmentsData = deliveries.map(delivery => {
            const deliveryDateTime = (delivery.delivery_time || '').replace('T', ' ');
            return {
                deliveryNumber: delivery.delivery_num,
                material: materialMap.get(delivery.material_id) || `Material ${delivery.material_id}`,
                incomingWeight: Number(delivery.incoming_weight) || 0,
                materialWeight: Number(delivery.material_weight ?? delivery.incoming_weight) || 0,
                deliveryDateTime,
                status: delivery.status || 'pending'
            };
        });

        if (shipmentsData.length > 0) {
            const shipmentDates = shipmentsData
                .map(shipment => shipment.deliveryDateTime.split(' ')[0])
                .filter(Boolean)
                .sort();

            defaultShipmentDateBounds.min = shipmentDates[0];
            defaultShipmentDateBounds.max = shipmentDates[shipmentDates.length - 1];
            setDateInputs(defaultShipmentDateBounds.min, defaultShipmentDateBounds.max);
        } else {
            const today = getCurrentDate();
            defaultShipmentDateBounds = { min: today, max: today };
            setDateInputs(today, today);
        }

        renderShipments();
    } catch (error) {
        console.error('Failed to load dashboard data:', error);
        shipmentsData = [];
        const today = getCurrentDate();
        defaultShipmentDateBounds = { min: today, max: today };
        setDateInputs(today, today);
        renderShipments();
        materialsData = [];
        renderMaterials();
        activeAlerts = [];
        renderAlerts();
        recommendationsData = [];
        renderRecommendations();
    } finally {
        isPollingShipments = false;
    }
}

function renderShipments() {
    const tbody = document.getElementById('truckTableBody');
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    tbody.innerHTML = '';
    
    // Filter shipments by date range and sort by status (completed first)
    const filteredShipments = shipmentsData
        .map((shipment, index) => ({ ...shipment, originalIndex: index }))
        .filter(shipment => {
            const shipmentDate = shipment.deliveryDateTime.split(' ')[0];
            return shipmentDate >= startDate && shipmentDate <= endDate;
        })
        .sort((a, b) => {
            if (a.status === 'completed' && b.status === 'upcoming') return -1;
            if (a.status === 'upcoming' && b.status === 'completed') return 1;
            return 0;
        });
    
    filteredShipments.forEach(shipment => {
        const row = document.createElement('tr');
        row.className = shipment.status;
        row.innerHTML = `
            <td>${shipment.deliveryNumber}</td>
            <td>${shipment.material}</td>
            <td>${shipment.incomingWeight.toLocaleString()} lbs</td>
            <td>${shipment.materialWeight} stn</td>
            <td>${shipment.deliveryDateTime}</td>
            <td><span class="status-${shipment.status} status-clickable" onclick="toggleStatus(${shipment.originalIndex})">${shipment.status.charAt(0).toUpperCase() + shipment.status.slice(1)}</span></td>
        `;
        tbody.appendChild(row);
    });
}

function showDatePicker() {
    document.getElementById('datePicker').style.display = 'block';
}

function onStartDateChange() {
    const startDate = document.getElementById('startDate').value;
    const endDateInput = document.getElementById('endDate');
    
    // If start date surpasses end date, update end date to match start date
    if (startDate > endDateInput.value) {
        endDateInput.value = startDate;
    }
    
    updateDateRange();
}

function onEndDateChange() {
    const startDateInput = document.getElementById('startDate');
    const endDate = document.getElementById('endDate').value;
    
    // If end date predates start date, update start date to match end date
    if (endDate < startDateInput.value) {
        startDateInput.value = endDate;
    }
    
    updateDateRange();
}

function resizeDateInput() {
    const input = document.getElementById('dateRange');
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    context.font = '16px Roboto, sans-serif';
    const textWidth = context.measureText(input.value || input.placeholder).width;
    input.style.width = Math.max(textWidth + 30, 100) + 'px';
}

function updateDateRange() {
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    const dateRangeInput = document.getElementById('dateRange');
    
    if (startDate === endDate) {
        dateRangeInput.value = startDate;
    } else {
        dateRangeInput.value = `${startDate} to ${endDate}`;
    }
    
    resizeDateInput();
    renderShipments();
}

function renderMaterials() {
    const cardsContainer = document.getElementById('materialCards');
    const totalWeightLabel = document.getElementById('totalWeight');

    if (!cardsContainer || !totalWeightLabel) {
        return;
    }

    cardsContainer.innerHTML = '';

    if (!Array.isArray(materialsData) || materialsData.length === 0) {
        totalWeightLabel.textContent = 'Total Weight: 0 tons';
        return;
    }

    let runningTotal = 0;

    materialsData.forEach((material) => {
        const weight = Number(material.weight) || 0;
        runningTotal += weight;

        const { label, tone, fillPercentage } = classifyMaterialByBin(material);

        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
            <div class="row">
                <div class="card-header ${tone}">${material.type}</div>
            </div>
            <div class="card-number">${formatTons(weight)}</div>
            <div class="card-subtitle">Fill: ${fillPercentage}% (${Number(material.bins_filled ?? 0).toFixed(2)} bins)</div>
            <div class="card-indicator ${tone}">${label}</div>
        `;
        cardsContainer.appendChild(card);
    });

    totalWeightLabel.textContent = `Total Weight: ${runningTotal.toLocaleString()} tons`;
}

function renderAlerts() {
    const warningBanner = document.getElementById('iconWarning');
    const warningText = document.getElementById('warningText');

    if (!warningBanner || !warningText) {
        return;
    }

    if (!Array.isArray(activeAlerts) || activeAlerts.length === 0) {
        warningBanner.hidden = true;
        warningBanner.style.display = 'none';
        warningText.innerHTML = '';
        return;
    }

    const filteredAlerts = activeAlerts.filter(alert => alert.alert_level === 'critical');
    if (filteredAlerts.length === 0) {
        warningBanner.hidden = true;
        warningBanner.style.display = 'none';
        warningText.innerHTML = '';
        return;
    }

    const alertMarkup = filteredAlerts
        .map(alert => `<div class="alert-line alert-${alert.alert_level}">${alert.message}</div>`)
        .join('');

    warningText.innerHTML = alertMarkup;
    warningBanner.hidden = false;
    warningBanner.style.display = 'flex';
}

function renderRecommendations() {
    const list = document.getElementById('recommendationsList');
    if (!list) {
        return;
    }

    list.innerHTML = '';

    if (!Array.isArray(recommendationsData) || recommendationsData.length === 0) {
        list.innerHTML = '<div class="recommendation-item"><span>No order actions needed this week.</span></div>';
        return;
    }

    recommendationsData.forEach((rec) => {
        const item = document.createElement('div');
        item.className = 'recommendation-item';

        const left = document.createElement('div');
        left.innerHTML = `<strong>${rec.material_type}</strong><div class="recommendation-meta">Current: ${formatTons(rec.current_weight)} · Avg daily: ${rec.average_daily_consumption.toLocaleString()} tons</div>`;

        const right = document.createElement('div');
        if (rec.recommended_order_tons) {
            const eta = rec.recommended_order_date ? new Date(rec.recommended_order_date).toLocaleDateString() : 'ASAP';
            right.innerHTML = `<strong>${rec.recommended_order_tons.toLocaleString()} tons</strong><div class="recommendation-meta">Order by ${eta}</div>`;
        } else {
            right.innerHTML = `<div class="recommendation-meta">${rec.rationale}</div>`;
        }

        item.appendChild(left);
        item.appendChild(right);
        list.appendChild(item);
    });
}

// Close date picker when clicking outside
document.addEventListener('click', function(event) {
    const datePicker = document.getElementById('datePicker');
    const dateRange = document.getElementById('dateRange');
    const container = document.querySelector('.date-filter-container');
    
    if (!container.contains(event.target)) {
        datePicker.style.display = 'none';
    }
});

function setToday() {
    const today = getCurrentDate();
    setDateInputs(today, today);
    renderShipments();
}

function setWeek() {
    const today = new Date();
    const startOfWeek = new Date(today);
    const endOfWeek = new Date(today);
    endOfWeek.setDate(today.getDate() + 6);
    
    const start = startOfWeek.toISOString().split('T')[0];
    const end = endOfWeek.toISOString().split('T')[0];
    setDateInputs(start, end);
    renderShipments();
}

function clearRange() {
    if (defaultShipmentDateBounds.min && defaultShipmentDateBounds.max) {
        setDateInputs(defaultShipmentDateBounds.min, defaultShipmentDateBounds.max);
    } else {
        const today = getCurrentDate();
        setDateInputs(today, today);
    }
    document.getElementById('datePicker').style.display = 'none';
    renderShipments();
}

function searchTruck() {
    const input = document.getElementById('truckSearch').value.toLowerCase();
    const table = document.getElementById('truckTable');
    const rows = table.getElementsByTagName('tr');
    
    for (let i = 1; i < rows.length; i++) {
        const truckNumber = rows[i].getElementsByTagName('td')[0].textContent.toLowerCase();
        rows[i].style.display = truckNumber.includes(input) ? '' : 'none';
    }
}

function toggleStatus(index) {
    shipmentsData[index].status = shipmentsData[index].status === 'completed' ? 'upcoming' : 'completed';
    renderShipments();
    updateJsonFile();
}

function updateJsonFile() {
    console.log('Updated shipments data:', shipmentsData);
    // In a real application, this would send data to a backend API
    // For now, we'll just log the updated data
}

function filterStatus(status) {
    const table = document.getElementById('truckTable');
    const rows = table.getElementsByTagName('tr');
    const buttons = document.querySelectorAll('.filter-btn');
    
    buttons.forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    for (let i = 1; i < rows.length; i++) {
        if (status === 'all') {
            rows[i].style.display = '';
        } else {
            rows[i].style.display = rows[i].classList.contains(status) ? '' : 'none';
        }
    }
}

window.addEventListener('load', () => {
    document.getElementById('btnGenerateTotal')?.addEventListener('click', renderMaterials);
    document.getElementById('btnPDFExport')?.addEventListener('click', () => {
        console.log('PDF export requested. Implement export pipeline here.');
    });
    loadShipments();
    setInterval(loadShipments, 15000);
});
