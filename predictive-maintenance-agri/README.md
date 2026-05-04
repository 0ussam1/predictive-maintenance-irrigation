# 🚜 Irrigation Intelligence Pro | MLOps Predictive Maintenance

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLOps](https://img.shields.io/badge/MLOps-DVC%20%7C%20MLflow-blueviolet?style=for-the-badge)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Système intelligent de surveillance IoT et de maintenance prédictive pour l'agriculture moderne.** Anticipez les pannes, optimisez vos récoltes et réduisez vos coûts opérationnels.

---

## 🌟 Vision du Projet
Dans une ferme intelligente, l'irrigation est vitale. La défaillance d'une pompe peut entraîner un stress hydrique fatal pour les cultures. **Irrigation Intelligence Pro** analyse en temps réel les données de vos capteurs IoT pour prédire les pannes avant qu'elles ne surviennent.

### 🚀 Fonctionnalités Clés
- **Diagnostic IA en Temps Réel** : Inférence ultra-rapide via FastAPI et un modèle Random Forest optimisé.
- **Sécurité Hybride** : Combinaison de l'IA et de règles métiers strictes (Safety Overrides) pour une protection maximale.
- **Monitoring MLOps** : Surveillance continue des métriques avec Prometheus et visualisation sur Grafana.
- **Pipeline Reproductible** : Gestion du cycle de vie des données et du modèle avec DVC et MLflow.
- **Dashboard Premium** : Interface utilisateur moderne avec guide interactif et historique des diagnostics.

---

## 🛠 Stack Technique

| Couche | Technologies |
| :--- | :--- |
| **Inférence / API** | FastAPI, Uvicorn, Pydantic |
| **Machine Learning** | Scikit-Learn, Pandas, Joblib |
| **DataOps / MLOps** | DVC, MLflow, Evidently AI |
| **Monitoring** | Prometheus, Grafana |
| **Infrastructure** | Docker, Jenkins (CI/CD) |

---

## 📊 Niveaux d'Alerte IA
Le système évalue le risque de panne sur une échelle de 0 à 100% :
- 🟢 **0% - 40% (Normal)** : Fonctionnement idéal, surveillance standard.
- 🟡 **40% - 70% (Attention)** : Surveillance renforcée recommandée.
- 🟠 **70% - 85% (Élevé)** : Maintenance préventive à planifier.
- 🔴 **85% - 100% (Critique)** : Danger imminent ! Intervention urgente requise.

---

## 📁 Structure du Projet
```bash
.
├── api/             # API FastAPI (Inférence)
├── frontend/        # Dashboard HTML/CSS/JS & Guide
├── monitoring/      # Config Prometheus & Grafana
├── src/             # Code source du pipeline
│   ├── dataops/     # Ingestion, Preprocessing, Features
│   └── monitoring/  # Détection de drift & Alertes
├── tests/           # Tests unitaires & Intégration
├── dvc.yaml         # Orchestration du pipeline
└── requirements.txt # Dépendances Python
```

---

## ⚙️ Installation & Lancement

1. **Cloner le projet**
   ```bash
   git clone https://github.com/0ussam1/predictive-maintenance-irrigation.git
   cd predictive-maintenance-irrigation
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

3. **Lancer l'API**
   ```bash
   uvicorn api.app:app --reload
   ```
   *Accédez au dashboard sur : `http://localhost:8000`*

---

## 👨‍💻 Auteur
**Oussama Akarrou** - *MLOps Engineer* - [GitHub](https://github.com/0ussam1)

---
*Propulsé par Irrigation Intelligence Pipeline & Antigravity IA*