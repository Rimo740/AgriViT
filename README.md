# 🌿 AgriViT: Robust Crop Disease Recognition

**AgriViT** is a research-grade Deep Learning application designed to detect crop diseases under challenging real-world conditions (Blur, Noise, Low Light). 

It compares three architectures to demonstrate the robustness of Vision Transformers (ViT) over traditional CNNs.

## 🚀 Features
* **Multi-Model Support:** Switch between MobileViT, EfficientNet-B0, and MobileNetV3.
* **Real-Time Stress Testing:** Simulate bad camera focus or sensor noise inside the app to test model resilience.
* **Instant Diagnosis:** Upload an image and get a prediction in <1 second.

## 🛠️ Models
| Model | Type | Best Use Case |
| :--- | :--- | :--- |
| **MobileViT** | Transformer | **Best Robustness** (Blur/Noise resilient) |
| **EfficientNet-B0** | CNN | High Accuracy on Clean Data |
| **MobileNetV3** | CNN | Fastest Inference Speed |

## 💻 Local Installation

1. Clone the repo:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/AgriViT.git](https://github.com/YOUR_USERNAME/AgriViT.git)

   Install dependencies:

Bash

pip install -r requirements.txt
Run the app:

Bash

streamlit run app.py
📄 License
This project is part of a thesis research project.


---

### 📝 Final Deployment Steps

1.  **Place Files:** Ensure `app.py`, `requirements.txt`, `.gitignore`, `README.md`, `class_names.json`, and the three `.pth` files are all in the root of your folder.
2.  **Git Commands:** Open your terminal in that folder:
    ```bash
    git init
    git add .
    git commit -m "Initial deploy of AgriViT"
    git branch -M main
    git remote add origin https://github.com/YOUR_USERNAME/AgriViT.git
    git push -u origin main
    ```
    *(Replace `YOUR_USERNAME` with your actual GitHub username)*
3.  **Streamlit Cloud:** Go to [share.streamlit.io](https://share.streamlit.io), select the repo, and click **Deploy**.

**Next Step:** Are you ready to deploy, or do you need help setting up the GitHub remote?