const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const targetLetterElement = document.getElementById('target-letter');
const userPredictionElement = document.getElementById('user-prediction');
const scoreElement = document.getElementById('score');
let score = 0;
let targetLetter = '';
let lastPredictionTime = 0;
const predictionDelay = 3000; // 3 seconds

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        video.play();
    })
    .catch(err => {
        console.error("Error accessing webcam: ", err);
    });

function updateUI() {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frame = context.getImageData(0, 0, canvas.width, canvas.height);
    sendFrameToServer(frame);
}

function sendFrameToServer(frame) {
    const data = new FormData();
    data.append('frame', frame);

    fetch('/predict', {
        method: 'POST',
        body: data
    })
    .then(response => response.json())
    .then(data => {
        if (data.prediction) {
            userPredictionElement.innerText = data.prediction;
            if (data.prediction === targetLetter && (Date.now() - lastPredictionTime) > predictionDelay) {
                score++;
                scoreElement.innerText = score;
                lastPredictionTime = Date.now();
                targetLetter = getRandomTargetLetter();
                targetLetterElement.innerText = targetLetter;
            }
        }
    })
    .catch(err => {
        console.error("Error sending frame to server: ", err);
    });
}

function getRandomTargetLetter() {
    const letters = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    return letters.charAt(Math.floor(Math.random() * letters.length));
}

setInterval(updateUI, 100); // Update UI every 100ms