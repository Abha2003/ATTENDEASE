// static/app.js

document.addEventListener('DOMContentLoaded', () => {
    const webcamLink = document.querySelector('a[href="/webcam"]');
    const imageFileField = document.getElementById('image_file');
    const imageDataField = document.getElementById('image_data');

    // Function to handle messages from the webcam window
    function handleWebcamMessage(event) {
        // Ensure the message comes from the same origin for security
        if (event.origin !== window.location.origin) {
            console.warn('MESSAGE FROM UNKNOWN ORIGIN:', event.origin); // Converted to uppercase
            return;
        }

        if (event.data && event.data.type === 'webcam_image_data') {
            const imageDataUrl = event.data.data;
            imageDataField.value = imageDataUrl; // Populate the hidden input
            
            // Optionally, disable the file input if webcam data is provided
            if (imageFileField) { // Check if element exists
                imageFileField.disabled = true;
                imageFileField.required = false; // Make file input not required if webcam is used
            }

            console.log('WEBCAM IMAGE DATA RECEIVED AND SET.'); // Converted to uppercase
            // Using alert for quick feedback; consider a custom modal in a real app
            alert('WEBCAM PHOTO CAPTURED SUCCESSFULLY! YOU CAN NOW REGISTER THE STUDENT.'); // Converted to uppercase
        }
    }

    // Listen for messages from other windows (like the webcam capture window)
    window.addEventListener('message', handleWebcamMessage);

    // When the webcam link is clicked, open the webcam page in a new window/tab
    if (webcamLink) {
        webcamLink.addEventListener('click', (e) => {
            e.preventDefault(); // Prevent default link behavior
            window.open(webcamLink.href, '_blank', 'width=800,height=600,resizable,scrollbars=yes');
        });
    }

    // Optional: Clear webcam data if a file is selected
    if (imageFileField) {
        imageFileField.addEventListener('change', () => {
            if (imageFileField.files.length > 0) {
                imageDataField.value = ''; // Clear hidden webcam data
                // Optionally re-enable webcam link if it was disabled
                if (webcamLink) webcamLink.style.pointerEvents = 'auto';
            }
        });
    }
});
