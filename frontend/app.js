function showTab(index) {
    document.querySelectorAll('.tab').forEach((tab, i) => {
        tab.classList.toggle('active', i === index);
    });
    document.querySelectorAll('.content').forEach((content, i) => {
        content.classList.toggle('active', i === index);
    });
}

let shipmentsData = [];
let defaultShipmentDateBounds = { min: null, max: null };

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
    try {
        const [deliveriesResponse, materialsResponse] = await Promise.all([
            fetch('http://localhost:8000/api/deliveries'),
            fetch('http://localhost:8000/api/materials')
        ]);

        if (!deliveriesResponse.ok) {
            throw new Error(`Failed to load deliveries: ${deliveriesResponse.status}`);
        }

        const deliveries = await deliveriesResponse.json();
        const materials = materialsResponse.ok ? await materialsResponse.json() : [];
        const materialMap = new Map(materials.map(material => [material.id, material.type]));

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
        console.error('Failed to load shipments:', error);
        shipmentsData = [];
        const today = getCurrentDate();
        defaultShipmentDateBounds = { min: today, max: today };
        setDateInputs(today, today);
        renderShipments();
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

async function getAllWeights(material) { 
    const response = await fetch('../backend/data/currMaterialWeight.json'); 
    currentWeight = await response.json(); 
    const arrMaterialType = [ 
        {id:'SS2',cardName:'SS2CurrTonne'}, 
        {id:'TNStone', cardName:'TNStoneCurrTonne'}, 
        {id:'SMSClay', cardName:'intSMSClayCurrTonne'}, 
        {id:'LR28', cardName:'intLR28MWeightCurrTonne'}, 
        {id:'Minispar', cardName:'intMinsparCurrTonne'}, 
        {id:'Sandspar', cardName:'intSandsparCurrTonne'}, 
        { id:'Feldspar', cardName:'intFeldsparCurrTonne'} ] 

        arrMaterialType.forEach(item => { 
            const weightObj = currentWeight.find(w => w.type === item.id); 
            if (weightObj) { 
                document.getElementById(item.cardName).innerHTML = `${weightObj.weight} ${weightObj.metric}`; 
            } 
        }); 
}

window.onload = loadShipments;
window.addEventListener('DOMContentLoaded', getAllWeights);
