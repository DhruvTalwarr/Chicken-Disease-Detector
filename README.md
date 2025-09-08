🐔💡 AI-Powered Chicken Disease Detection

This repository contains the code for an AI-powered prototype for automated health assessment in chickens.
The project aims to fight Antimicrobial Resistance (AMR) by enabling early disease detection in poultry using computer vision.

📌 Project Overview

Traditional disease monitoring in poultry is often manual, slow, and reactive.
This project introduces a proactive, digital solution:

1️⃣ 🔍 Automated image classification of chicken health
2️⃣ 🚑 Early detection of diseases for timely intervention
3️⃣ 🌍 Support for responsible antimicrobial usage, reducing AMR risk

The model classifies chicken images into 4 categories:

1️⃣ 🟢 Healthy
2️⃣ 🦠 Coccidiosis
3️⃣ 🐦 Newcastle Disease
4️⃣ 🧫 Salmonella

⚙️ Technology Stack

1️⃣ 🐍 Python – Core programming language
2️⃣ 🧠 TensorFlow/Keras – Deep learning framework
3️⃣ 📊 Pandas – Data handling and analysis
4️⃣ 📈 Matplotlib – Training performance visualization
5️⃣ 🔢 Scikit-learn – Data splitting & class weighting

🔬 How it Works

1️⃣ Data Preparation
Trained on the Kaggle Chicken Disease Dataset.
Data pipeline with augmentation for better diversity.

2️⃣ Model Architecture
Transfer Learning with ResNet50 (pre-trained on ImageNet).
Fine-tuned for detecting chicken-specific diseases.

3️⃣ Training
Loss: Categorical Cross-Entropy.
Class Weights to handle imbalance.
EarlyStopping to avoid overfitting.

4️⃣ Evaluation
Accuracy & loss plots for training visualization.
Predictions with confidence scores on test images.

📁 Files & Folders

1️⃣ 📜 main.py → Training & evaluation script
2️⃣ 🧪 evaluate.py → Test the trained model on a single image
3️⃣ 🏆 best_chicken_disease_model.keras → Saved model file
4️⃣ 📑 train_data.csv → Image labels & metadata
5️⃣ 🗂 Train/ → Folder with training images

🚀 Getting Started
✅ Prerequisites

1️⃣ Python 3.x
2️⃣ TensorFlow
3️⃣ Pandas
4️⃣ Scikit-learn
5️⃣ Matplotlib

📦 Install dependencies:

pip install tensorflow pandas scikit-learn matplotlib

▶️ Running the Code

1️⃣ Clone this repository
2️⃣ Place train_data.csv + Train/ folder in the project directory
3️⃣ Train the model:

python main.py


4️⃣ Test the model on an image (update image_path in evaluate.py):

python evaluate.py

🎤 Jury Presentation & 🌟 Future Scope

1️⃣ 📡 Integration with a real-time monitoring system
2️⃣ 🧬 Expansion of dataset with more diseases
3️⃣ 📱 Deployment with lightweight models (e.g., MobileNet) for mobile applications

👉 This project bridges AI + Poultry Health to fight AMR and promote sustainable farming. 🌍✨
