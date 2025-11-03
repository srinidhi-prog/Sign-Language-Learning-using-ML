import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    BASEDIR = os.path.abspath(os.path.dirname(__file__))
    MODEL_DIR = os.path.join(BASEDIR, 'models')
    
    # Model paths
    MODEL_PATH = os.path.join(MODEL_DIR, 'isl_landmark_model.h5')
    ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    
    # Use original source model files directly to avoid corruption
    # (pointing to the original working files)
    SOURCE_MODEL_PATH = "C:/Users/Thilak/Desktop/Projects/Sign-language/4-OWN/isl_landmark_model.h5"
    SOURCE_ENCODER_PATH = "C:/Users/Thilak/Desktop/Projects/Sign-language/4-OWN/label_encoder.pkl"