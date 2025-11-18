document.addEventListener('DOMContentLoaded', function() {
    // --- Theme Switcher Logic ---
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    const body = document.body;

    darkModeToggle.addEventListener('change', function() {
        if (this.checked) {
            body.classList.add('dark');
        } else {
            body.classList.remove('dark');
        }
    });

    // --- Prediction Logic ---
    const predictBtn = document.getElementById('predictBtn');
    const errorMessageDiv = document.getElementById('errorMessage');
    const predictionResultDiv = document.getElementById('predictionResult');
    
    const getValue = id => document.getElementById(id).value.trim();

    predictBtn.addEventListener('click', async () => {
        const payload = {
            'Age': getValue('age'),
            'Gender': getValue('gender'),
            'Diagnosis': getValue('diagnosis')
        };

        if (!payload.Age || !payload.Gender || !payload.Diagnosis) {
            errorMessageDiv.textContent = 'Please fill in all fields.';
            errorMessageDiv.classList.remove('hidden');
            predictionResultDiv.classList.add('hidden');
            return;
        }

        // --- Show loading spinner ---
        errorMessageDiv.classList.add('hidden');
        predictBtn.querySelector('.btn-text').classList.add('hidden');
        predictBtn.querySelector('.spinner').classList.remove('hidden');
        predictBtn.disabled = true;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const result = await response.json();

            if (response.ok) {
                document.getElementById('pred-drug').textContent = result.drug;
                document.getElementById('pred-dosage').textContent = result.dosage;
                document.getElementById('pred-route').textContent = result.route;
                document.getElementById('pred-freq').textContent = result.frequency;
                predictionResultDiv.classList.remove('hidden');
            } else {
                throw new Error(result.error || 'Prediction failed.');
            }
        } catch (error) {
            errorMessageDiv.textContent = "An error occurred. Please try again.";
            errorMessageDiv.classList.remove('hidden');
        } finally {
            // --- Hide loading spinner ---
            predictBtn.querySelector('.btn-text').classList.remove('hidden');
            predictBtn.querySelector('.spinner').classList.add('hidden');
            predictBtn.disabled = false;
        }
    });
});