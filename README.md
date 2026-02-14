# Real vs Fake Job Posting Prediction - ML Project

## 🎯 Project Description
Machine Learning system to detect fraudulent job postings using NLP and classification algorithms. Built with Django REST API, MLflow tracking, and Docker deployment.


## 🛠️ Tech Stack
- **ML**: Scikit-learn, Transformers, BERT, LightGBM
- **Tracking**: MLflow
- **Backend**: Django 4.2, Django REST Framework
- **Frontend**: Django Templates / React
- **Deployment**: Docker, Render.com
- **Version Control**: Git, GitHub

## 📊 Dataset
Kaggle "Real or Fake: Fake Job Posting Prediction"
- 17,880 job postings
- 18 features (text + metadata)
- ~5% fake postings (imbalanced dataset)

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/TouaibiAhmed/Real-Fake-Job-Posting-Prediction-ML-Project.git
cd Real-Fake-Job-Posting-Prediction-ML-Project
```

### 2. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate fake-job-detector
```

### 3. Download Dataset
Download from Kaggle and place in `Data/raw/fake_job_postings.csv`

### 4. Run Notebooks
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### 5. Train Model with MLflow
```bash
cd src/models
python train_model.py
```

### 6. Start Django API
```bash
cd job_detector_api
python manage.py migrate
python manage.py runserver
```

### 7. Docker Deployment
```bash
docker-compose up --build
```

## 📁 Project Structure
```
├── Data/              # Dataset and delivrables
├── notebooks/         # Jupyter notebooks
├── src/              # Source code
├── models/           # Trained models
├── job_detector_api/ # Django application
└── tests/            # Unit tests
```

## 🔗 API Endpoints
- `POST /api/predict/` - Predict if job posting is fake
- `GET /api/health/` - Health check

## 📈 MLflow Tracking
```bash
mlflow ui
# Visit: http://localhost:5000
```

## 🐳 Docker
```bash
docker build -t fake-job-detector .
docker run -p 8000:8000 fake-job-detector
```

## 📝 License
MIT License

