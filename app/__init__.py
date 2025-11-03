from flask import Flask
from config import Config
import os

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Ensure model directory exists
    os.makedirs(app.config['MODEL_DIR'], exist_ok=True)
    
    # Register blueprints
    from app.routes import main_bp, init_models
    app.register_blueprint(main_bp)
    
    # Initialize models
    with app.app_context():
        init_models(app)
    
    return app