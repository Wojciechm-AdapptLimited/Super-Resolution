document.getElementById('uploadButton').addEventListener('click', () => {
    const imageInput = document.getElementById('imageInput');
    
    if (!imageInput.files[0] || !imageInput.files[0].name.match(/\.(jpg|jpeg|png|gif)$/)) {
        alert('Please select an image');
        return;
    }

    imageToTensor(imageInput.files[0]).then((tensor) => {
        postRequest(Array.from(tensor.dataSync()));
    });
});

document.getElementById('imageInput').addEventListener('change', () => {
    const uploadButton = document.getElementById('uploadButton');
    uploadButton.disabled = false;

    const imageInput = document.getElementById('imageInput');
    const image = imageInput.files[0];

    const reader = new FileReader();

    reader.onloadend = (e) => {
        document.getElementById('originalImage').src = e.target.result;
    }

    reader.readAsDataURL(image);
});

const postRequest = async (data) => {
    console.log(data);

    fetch('http://localhost:8501/v1/models/UNet:predict', {
            method: 'POST',
            mode: 'no-cors',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({instances: [data]})
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('enhancedImage').src = 'data:image/jpeg;base64,' + data.predictions;
        })
        .catch(error => console.error('Error:', error));
}

// Function to convert image to tensor
async function imageToTensor(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = 400;
                canvas.height = 400;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, 400, 400);
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                // Assuming TensorFlow.js is available
                const tensor = window.tf.browser.fromPixels(imageData).toFloat().div(window.tf.scalar(255.0)).expandDims(0);
                resolve(tensor);
            };
            img.onerror = reject;
            img.src = reader.result;
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}
