🌿 Plant Disease Classification 📊🩺


A deep learning-based plant disease classification system using ResNet-9, designed to detect plant diseases from leaf images. The project includes a user-friendly web interface built with Gradio 
for easy interaction with the model.

📌 Table of Contents
About the Project
Demo
Features
Tech Stack
Installation
Usage
Model Architecture
Results
Contributing
License





🌱 About the Project


Plant diseases can significantly impact crop yield and quality. Early and accurate detection is essential to prevent outbreaks. This project uses a ResNet-9 Convolutional Neural Network (CNN) trained on a plant disease 
dataset to classify images of leaves as healthy or diseased, and identify the specific type of disease.

✨ Features
📸 Upload leaf images and get instant predictions.

🖥️ Simple, clean, and responsive Gradio interface.


📊 Visual feedback on prediction confidence.


📂 Custom-trained ResNet-9 model.


🔍 Can be extended to new plant species or diseases.





🛠️ Tech Stack

Python
PyTorch
Gradio
NumPy / Pandas / Matplotlib
Torchvision
Google Colab / Jupyter Notebook


📦 Installation
Clone the repository
bash
CopyEdit
git clone https://github.com/yourusername/plant-disease-classification.git
cd plant-disease-classification

Install dependencies
bash
CopyEdit
pip install -r requirements.txt

Download the trained model

Place the trained model file (e.g., plant_disease_resnet9.pth) inside the models/ directory.

🚀 Usage


Run the Gradio app locally:


bash
CopyEdit
python app.py



Upload an image of a plant leaf, and the model will classify it.



🧠 Model Architecture


This project uses a ResNet-9
 architecture — a lightweight residual neural network with skip 
connections, well-suited for image classification tasks while being 
efficient for deployment.


Key Specs:




9 layers (with skip connections)




ReLU activations




CrossEntropy Loss




Adam Optimizer

📊 Results


MetricValueTraining Accuracy95%Validation Accuracy92%
Dataset used: Kaggle
Sample Prediction:
🤝 Contributing
Contributions, issues, and feature requests are welcome!

Fork the project
Create your feature branch (git checkout -b feature/yourFeature)

Commit your changes (git commit -m 'Add your feature')

Push to the branch (git push origin feature/yourFeature)

Open a Pull Request



