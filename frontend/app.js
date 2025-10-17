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
let uploadedFile = null;

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
    const year = today.getFullYear();
    const month = String(today.getMonth() + 1).padStart(2, '0');
    const day = String(today.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
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
        renderUsageRates();
        populateMaterialDropdown();

        const materialMap = new Map(materialsData.map(material => [material.id, material.type]));

        shipmentsData = deliveries.map((delivery, index) => {
            const deliveryDateTime = (delivery.delivery_time || '').replace('T', ' ');
            return {
                id: index,
                deliveryNumber: delivery.delivery_num,
                material: materialMap.get(delivery.material_id) || `Material ${delivery.material_id}`,
                expectedWeight: Number(delivery.expected_weight || delivery.incoming_weight) || 0,
                actualWeight: Number(delivery.actual_weight || delivery.material_weight) || 0,
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
        }
        
        // Always default to current day
        const today = getCurrentDate();
        setDateInputs(today, today);

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
            <td>${shipment.expectedWeight.toLocaleString()} lbs</td>
            <td>${shipment.actualWeight.toLocaleString()} lbs</td>
            <td>${shipment.deliveryDateTime}</td>
            <td><span class="status-${shipment.status} status-clickable" data-shipment-id="${shipment.id}">${shipment.status.charAt(0).toUpperCase() + shipment.status.slice(1)}</span></td>
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
    
    if (container && !container.contains(event.target)) {
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
    const endDate = new Date(today);
    endDate.setDate(today.getDate() + 6);
    
    const start = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
    const end = `${endDate.getFullYear()}-${String(endDate.getMonth() + 1).padStart(2, '0')}-${String(endDate.getDate()).padStart(2, '0')}`;
    
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

function toggleStatus(shipmentId) {
    const shipment = shipmentsData.find(s => s.id === shipmentId);
    if (shipment) {
        shipment.status = shipment.status === 'completed' ? 'upcoming' : 'completed';
        renderShipments();
        updateJsonFile();
    }
}

function updateJsonFile() {
    console.log('Updated shipments data:', shipmentsData);
    // In a real application, this would send data to a backend API
    // For now, we'll just log the updated data
}

function filterStatus(status) {
    const table = document.getElementById('truckTable');
    const rows = table.getElementsByTagName('tr');
    
    for (let i = 1; i < rows.length; i++) {
        if (status === 'all') {
            rows[i].style.display = '';
        } else {
            rows[i].style.display = rows[i].classList.contains(status) ? '' : 'none';
        }
    }
}

function renderUsageRates() {
    const grid = document.getElementById('usageRatesGrid');
    const warningBanner = document.getElementById('usageWarning');
    const warningText = document.getElementById('usageWarningText');
    const avgLabel = document.getElementById('avgUsage');

    if (!grid) {
        return;
    }

    grid.innerHTML = '';

    if (!Array.isArray(materialsData) || materialsData.length === 0) {
        if (avgLabel) {
            avgLabel.textContent = 'Average Usage: -- lbs/min';
        }
        if (warningBanner) {
            warningBanner.hidden = true;
            warningBanner.style.display = 'none';
        }
        return;
    }

    let totalRate = 0;
    const elevatedMaterials = [];

    materialsData.forEach((material) => {
        const baseRate = (Number(material.weight) || 0) * 0.0012;
        const usageRate = Number((baseRate + Math.random() * 1.5 + 1).toFixed(1));
        totalRate += usageRate;

        const tone = usageRate < 2 ? 'green' : usageRate < 4 ? 'yellow' : 'red';
        if (tone === 'red') {
            elevatedMaterials.push(material.type);
        }

        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
            <div class="row">
                <div class="card-header ${tone}">${material.type}</div>
            </div>
            <div class="card-number">${usageRate.toFixed(1)} lbs/min</div>
        `;
        grid.appendChild(card);
    });

    if (avgLabel) {
        const averageRate = totalRate / materialsData.length;
        avgLabel.textContent = `Average Usage: ${averageRate.toFixed(1)} lbs/min`;
    }

    if (warningBanner && warningText) {
        if (elevatedMaterials.length) {
            warningText.textContent = `Notice: ${elevatedMaterials.join(', ')} showing elevated usage rates.`;
            warningBanner.hidden = false;
            warningBanner.style.display = 'flex';
        } else {
            warningBanner.hidden = true;
            warningBanner.style.display = 'none';
        }
    }
}

function populateMaterialDropdown() {
    const select = document.getElementById('materialSelect');
    const selectionBlock = document.getElementById('materialSelection');
    if (!select || !selectionBlock) {
        return;
    }

    select.innerHTML = '<option value="">Select a material...</option>';

    if (Array.isArray(materialsData) && materialsData.length > 0) {
        materialsData.forEach((material) => {
            const option = document.createElement('option');
            option.value = material.type;
            option.textContent = material.type;
            select.appendChild(option);
        });
        selectionBlock.hidden = false;
    } else {
        selectionBlock.hidden = true;
    }

    checkSubmitButton();
}

function handleImageUpload(event) {
    const files = event.target.files || event.dataTransfer?.files;
    if (!files || files.length === 0) {
        return;
    }

    uploadedFile = files[0];
    const uploadArea = document.getElementById('uploadArea');
    if (uploadedFile && uploadArea) {
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadArea.innerHTML = `<img src="${e.target.result}" alt="Selected image" style="max-width: 100%; max-height: 220px; border-radius: 12px;">`;
        };
        reader.readAsDataURL(uploadedFile);
    }

    const selectionBlock = document.getElementById('materialSelection');
    if (selectionBlock) {
        selectionBlock.hidden = false;
    }

    checkSubmitButton();
}

function checkSubmitButton() {
    const analyzeButton = document.getElementById('btnAnalyzeImage');
    const materialSelect = document.getElementById('materialSelect');
    const densityInput = document.getElementById('densityInput');

    if (!analyzeButton) {
        return;
    }

    const hasImage = uploadedFile !== null;
    const hasMaterial = !!(materialSelect && materialSelect.value);
    const hasDensity = densityInput && densityInput.value && Number(densityInput.value) > 0;

    analyzeButton.disabled = !(hasImage && hasMaterial && hasDensity);
}

async function analyzeImage() {
    const analyzeButton = document.getElementById('btnAnalyzeImage');
    if (!analyzeButton || analyzeButton.disabled || !uploadedFile) {
        return;
    }

    const uploadSection = document.getElementById('uploadSection');
    const loadingSection = document.getElementById('loadingSection');
    const analysisSection = document.getElementById('analysisSection');
    const newImageSection = document.getElementById('newImageSection');
    const resultNumber = document.getElementById('resultNumber');
    const resultDetails = document.getElementById('resultDetails');
    const materialSelect = document.getElementById('materialSelect');
    const densityInput = document.getElementById('densityInput');

    const densityValue = densityInput ? parseFloat(densityInput.value) : NaN;
    if (!densityValue || Number.isNaN(densityValue)) {
        alert('Please enter a valid density value.');
        return;
    }

    analyzeButton.disabled = true;
    if (materialSelect) {
        materialSelect.disabled = true;
    }
    if (densityInput) {
        densityInput.disabled = true;
    }

    if (uploadSection) {
        uploadSection.style.display = 'none';
    }
    if (loadingSection) {
        loadingSection.hidden = false;
    }

    try {
        const formData = new FormData();
        formData.append('file', uploadedFile);
        formData.append('density', densityValue.toString());

        const response = await fetch('http://localhost:8000/api/weight', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Analysis failed (${response.status})`);
        }

        const massHeader = response.headers.get('X-Mass-Short-Ton');
        const blob = await response.blob();
        const overlayUrl = URL.createObjectURL(blob);

        const uploadedImage = document.getElementById('uploadedImage');
        if (uploadedImage) {
            uploadedImage.src = overlayUrl;
        }

        if (resultNumber) {
            resultNumber.textContent = massHeader ? `${Number(massHeader).toFixed(2)} tons` : '-- tons';
        }
        if (resultDetails) {
            resultDetails.textContent = 'Estimated short tons from uploaded image.';
        }
    } catch (error) {
        console.error('Failed to analyze image:', error);
        if (resultNumber) {
            resultNumber.textContent = '--';
        }
        if (resultDetails) {
            resultDetails.textContent = 'Unable to analyze image. Please try again.';
        }
    } finally {
        if (loadingSection) {
            loadingSection.hidden = true;
        }
        if (analysisSection) {
            analysisSection.style.display = 'flex';
        }
        if (newImageSection) {
            newImageSection.style.display = 'block';
        }
        analyzeButton.disabled = false;
    }
}

function resetImageUpload() {
    uploadedFile = null;
    const uploadSection = document.getElementById('uploadSection');
    const analysisSection = document.getElementById('analysisSection');
    const newImageSection = document.getElementById('newImageSection');
    const materialSelect = document.getElementById('materialSelect');
    const densityInput = document.getElementById('densityInput');
    const imageUpload = document.getElementById('imageUpload');
    const uploadArea = document.getElementById('uploadArea');

    if (imageUpload) {
        imageUpload.value = '';
    }
    if (uploadArea) {
        uploadArea.innerHTML = `
            <div class="upload-text">Click to upload image or drag and drop</div>
            <div class="upload-subtext">Supported formats: JPG, PNG, GIF</div>
        `;
    }
    if (uploadSection) {
        uploadSection.style.display = 'block';
    }
    if (analysisSection) {
        analysisSection.style.display = 'none';
    }
    if (newImageSection) {
        newImageSection.style.display = 'none';
    }
    if (materialSelect) {
        materialSelect.disabled = false;
        materialSelect.value = '';
    }
    if (densityInput) {
        densityInput.disabled = false;
        densityInput.value = '';
    }
    checkSubmitButton();
}

async function exportInventoryReport() {
    const button = document.getElementById('btnPDFExport');
    if (!button) {
        return;
    }

    // Request the server-side PDF summary and trigger a download when it returns.
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = 'Generating...';

    try {
        const response = await fetch('http://localhost:8000/api/report/inventory');
        if (!response.ok) {
            throw new Error(`Export failed with status ${response.status}`);
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        const timestamp = new Date().toISOString().split('T')[0];
        link.href = url;
        link.download = `inventory-report-${timestamp}.pdf`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Failed to export PDF:', error);
        alert('Unable to export the PDF report right now. Please try again shortly.');
    } finally {
        button.disabled = false;
        button.textContent = originalText;
    }
}
window.addEventListener('load', () => {
    document.getElementById('btnPDFExport')?.addEventListener('click', exportInventoryReport);
    document.getElementById('btnUsagePDFExport')?.addEventListener('click', exportInventoryReport);
    document.getElementById('btnGenerateUsageReport')?.addEventListener('click', renderUsageRates);
    document.getElementById('btnAnalyzeImage')?.addEventListener('click', analyzeImage);
    document.getElementById('btnNewImage')?.addEventListener('click', resetImageUpload);
    document.getElementById('materialSelect')?.addEventListener('change', checkSubmitButton);
    document.getElementById('densityInput')?.addEventListener('input', checkSubmitButton);
    const imageUpload = document.getElementById('imageUpload');
    imageUpload?.addEventListener('change', handleImageUpload);
    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) {
        uploadArea.addEventListener('dragover', (event) => {
            event.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (event) => {
            event.preventDefault();
            uploadArea.classList.remove('dragover');
            handleImageUpload(event);
        });
    }

    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => showTab(parseInt(tab.dataset.tabIndex)));
    });

    document.getElementById('truckSearch')?.addEventListener('keyup', searchTruck);

    document.getElementById('dateRange')?.addEventListener('click', showDatePicker);
    document.getElementById('startDate')?.addEventListener('change', onStartDateChange);
    document.getElementById('endDate')?.addEventListener('change', onEndDateChange);

    document.getElementById('btnToday')?.addEventListener('click', setToday);
    document.getElementById('btnWeek')?.addEventListener('click', setWeek);
    document.getElementById('btnClear')?.addEventListener('click', clearRange);

    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', (event) => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
            filterStatus(event.target.dataset.status);
        });
    });

    document.getElementById('truckTableBody')?.addEventListener('click', (event) => {
        if (event.target.classList.contains('status-clickable')) {
            const shipmentId = parseInt(event.target.dataset.shipmentId);
            toggleStatus(shipmentId);
        }
    });

    loadShipments();
    setInterval(loadShipments, 15000);
    renderUsageRates();
    populateMaterialDropdown();
    checkSubmitButton();
});
