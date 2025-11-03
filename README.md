# Indian Sign Language Learning Web Application

This project is a web application designed to help users learn Indian Sign Language (ISL) through interactive gestures using a webcam interface. The application utilizes a trained machine learning model to recognize hand gestures and provide real-time feedback to users.

## Project Structure

```
isl-learning-web-app
├── app
│   ├── __init__.py
│   ├── routes.py
│   ├── models
│   │   └── predict.py
│   ├── static
│   │   ├── css
│   │   │   └── styles.css
│   │   └── js
│   │       └── webcam.js
│   └── templates
│       ├── base.html
│       ├── index.html
│       └── about.html
├── config.py
├── requirements.txt
├── run.py
├── models
│   ├── isl_landmark_model.h5
│   └── label_encoder.pkl
└── README.md
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd isl-learning-web-app
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python run.py
   ```

5. **Access the Application**
   Open your web browser and navigate to `http://127.0.0.1:5000`.

## Usage

- The main page will display a webcam feed where users can perform hand gestures.
- The application will show the target letter, the user's prediction, and the current score.
- Users can practice and improve their sign language skills through interactive feedback.

## About

This application aims to provide an engaging way to learn Indian Sign Language, making use of modern web technologies and machine learning. It is suitable for learners of all ages and can be used in educational settings or for personal development.

## Acknowledgments

- Special thanks to the contributors and the community for their support in developing this application.
- The machine learning model was trained using TensorFlow and is based on hand landmark detection techniques.