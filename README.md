This repository focuses on the **cloud deployment pipeline and big data architecture**, not the mobile frontend or final model training.

![Logo](https://github.com/AnniRanok/OC_Projet_8/blob/main/fruits.jpg)


##  Goal

To develop a **scalable, privacy-compliant cloud infrastructure** that processes large volumes of fruit image data, distributes a TensorFlow model, and applies dimensionality reduction with PySpark — all while ensuring cost-efficiency and GDPR alignment.

---

##  Contents

```
FruitsAI-Cloud-Pipeline/
├── pca_reduction.py              # Dimensionality reduction with PySpark
├── broadcast_model.py           # TensorFlow weights distribution logic
├── emr_setup.md                 # Step-by-step guide to launching an EMR cluster
├── sample_notebook.ipynb        # Inherited from former intern, updated and extended
├── data/
│   └── fruits_dataset/          # Sample image and label data
├── docs/
│   └── GDPR_compliance.md       # Hosting & privacy considerations
├── README.md                    # Project overview and goals (this file)
```

---

## 🧠 Key Concepts Implemented

* **AWS EMR** as scalable processing platform
* **PySpark** for distributed computation
* **Broadcasting model weights** across nodes
* **PCA** for feature dimensionality reduction
* **GDPR compliance**, via EU-region server enforcement
* **Cost-conscious architecture** for testing and demos only

---

##  Use Case Example

1.  Upload image dataset to AWS S3
2.  Launch EMR cluster (EU region only)
3.  Distribute model weights using `broadcast_model.py`
4.  Apply PCA reduction using `pca_reduction.py`
5.  Store transformed vectors for model training or inference


##  Contributing

Contributions welcome! Please open issues for bugs or suggestions.

---

##  Author

Inna Konar – Cloud & Data Science Consultant | AgriTech AI

📧 [konar.inna@gmail.com](mailto:konar.inna@gmail.com)

---

##  License

MIT License – use freely with attribution.

