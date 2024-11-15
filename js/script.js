// Show popup message
function showPopupMessage(message) {
    const popup = document.getElementById('popup');
    popup.textContent = message;
    popup.classList.add('show');

    // Hide the popup after 3 seconds
    setTimeout(() => {
        popup.classList.remove('show');
    }, 3000);
}

// Preview the uploaded image
function previewImage(event) {
    const imagePreview = document.getElementById('imagePreview');
    const file = event.target.files[0];

    if (file) {
        imagePreview.src = URL.createObjectURL(file);
        imagePreview.classList.remove('hidden');
    }
}

// Handle image upload and model selection
function uploadImage() {
    const fileInput = document.getElementById('imageUpload');
    const modelSelect = document.getElementById('modelSelect').value;
    const resultDiv = document.getElementById('result');

    if (!fileInput.files[0]) {
        alert("Please select an image.");
        return;
    }

    if (modelSelect === "coming_soon") {
        alert("This model is not yet available. Please select another model.");
        return;
    }

    // Show popup with selected model
    showPopupMessage(`Using ${modelSelect.replace('_', ' ')} model`);

    const reader = new FileReader();
    reader.onload = function(event) {
        const base64Image = event.target.result;

        // Send image data and model selection to Flask server via POST
        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: base64Image,
                model: modelSelect
            })
        })
        .then(response => response.json())
        .then(data => {
            resultDiv.textContent = `Result: ${data.result}`;
            resultDiv.style.opacity = 1; // Animate result appearance
        })
        .catch(error => {
            console.error('Fetch error:', error);
            resultDiv.textContent = 'An error occurred.';
            resultDiv.style.opacity = 1;
        });
    };
    reader.readAsDataURL(fileInput.files[0]);
}

// Enable/Disable model selection based on availability
function updateModelDropdown() {
    const modelSelect = document.getElementById('modelSelect');
    const options = modelSelect.options;

    for (let i = 0; i < options.length; i++) {
        if (options[i].value === "coming_soon") {
            options[i].disabled = true; // Disable unavailable options
        }
    }
}

// Initialize dropdown options on page load
document.addEventListener('DOMContentLoaded', updateModelDropdown);
