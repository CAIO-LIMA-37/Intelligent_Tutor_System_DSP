# 📚 LLMs for Teaching: Algorithm Development & Research

![Status](https://img.shields.io/badge/Status-In_Development-yellow?style=for-the-badge&logo=github)
![License](https://img.shields.io/badge/License-Private-red?style=for-the-badge)
![Field](https://img.shields.io/badge/Field-LLM-blue?style=for-the-badge&logo=python)
![Field](https://img.shields.io/badge/Field-Agent_AI-orange?style=for-the-badge&logo=openai)
![Field](https://img.shields.io/badge/Field-Teaching-green?style=for-the-badge&logo=google-classroom)

> **📌 Note on the Article**  
> The quantitative and qualitative analyses presented in the **"Quantitative Analysis"** and **"Qualitative Analysis"** sections of the article *"Reclaiming Pedagogical Control in DSP Education through Hybrid‑RAG‑Enhanced Local LLM"* are available in the [`/quantitative`](./quantitative) and [`/qualitative`](./qualitative) folders of this repository, respectively.

Central repository for the development, versioning, and experimentation of **Large Language Model (LLM) algorithms** applied to **Digital Signal Processing (DSP) education**. This project is carried out within the scope of [**LASSE - UFPA**](https://www.lasse.ufpa.br/pt).

---

## 🎯 Repository Goal: Intelligent Tutoring System (ITS)

The core focus of this repository is the development of an **Agent‑based AI Intelligent Tutoring System (ITS)**. The objective is to create a multi‑agent (or single‑agent) ecosystem capable of mediating student learning in DSP, acting as a **pedagogical mentor** that guides the learner instead of merely providing direct answers.

### 🛠️ Technical Implementation Fronts
- **Agentic Core:** Experimentation with state‑of‑the‑art frameworks such as `Transformers` and `vLLM` for LLM orchestration.
- **Foundation Models:** Testing and benchmarking of **open‑source LLMs** (Llama 3, Hugging Face models) and proprietary models to find the right balance between latency and pedagogical reasoning.
- **Full‑Stack Architecture:** Development of the complete ecosystem, integrating the AI engine (back‑end) with an intuitive and responsive interface (front‑end).
- **Pedagogical Robustness:** Implementation of hallucination mitigation techniques and prompt design focused on teaching methodologies.
- **Versioning & Evolution:** Version control for algorithms, prompts, and interface components to ensure research reproducibility.

---

## 🚀 Technologies & Frameworks

The project is in the technical exploration phase, considering the following stack:

| Area               | Technologies & Tools                                                                 |
|--------------------|--------------------------------------------------------------------------------------|
| **AI & Orchestration** | Hugging Face `Transformers`, `vLLM`                                                  |
| **Models**         | Llama family, foundation models (open‑source & API‑based)                            |
| **Development**    | Backend (Python / Docker), frontend integrations (HTML / CSS)                       |

---

## 📂 Project Structure

```
/
├── benchmark/       # Tests and preprocessing on the quantitative analysis benchmark dataset
├── qualitative/     # Jupyter notebook containing the qualitative analysis from the article
├── quantitative/    # Jupyter notebook + datasets used for the quantitative analysis from the article
└── src/             # Source code of AcademiCK algorithms (baseline for the ITS developed in this research)
```

---

## 👥 Contributors

- **Caio Lima**
- **Thiago**
- **Mariana**
- **Frank**
- **Prof. Aldebaro**

---

<p align="center">
  <strong>Developed with ☕ and 🤖 at <a href="https://www.lasse.ufpa.br/pt">LASSE - UFPA</a></strong>
</p>
```
