# Plateforme intelligente de surveillance des risques sur chantier

## Description

Ce projet vise à développer une plateforme intelligente d’analyse des risques de sécurité sur chantier basée sur la vision par ordinateur et le Machine Learning.

Le système exploite des flux vidéo issus de caméras fixes afin de :

- Détecter automatiquement les personnes, machines et équipements de protection individuelle (EPI)
- Analyser les interactions spatio-temporelles
- Identifier les situations à risque
- Générer des alertes explicables
- Fournir une interface web de supervision

L’objectif est de transformer des flux vidéo passifs en un outil d’aide à la décision pour les responsables HSE.

---

## Objectifs

- Détection automatique des objets critiques (personnes, machines, EPI)
- Analyse des interactions homme–machine
- Détection de présence en zones dangereuses
- Évaluation automatique du niveau de risque
- Génération d’alertes compréhensibles
- Visualisation via interface web

---

## Architecture du système

Flux vidéo  
→ Modèle Deep Learning (détection + tracking)  
→ Extraction de caractéristiques spatio-temporelles  
→ Modèle Machine Learning (classification du risque)  
→ Moteur d’alertes  
→ Interface web de supervision  

---

## Structure du projet


Structure du projet
/data
    /raw
    /processed
    /features

/src
    /detection
    /feature_extraction
    /ml_models
    /api
    /web

/notebooks
/docs




---

## Technologies utilisées

### Deep Learning
- YOLO (détection d’objets)
- Algorithmes de tracking multi-objets

### Machine Learning
- Random Forest
- XGBoost
- Feature engineering spatio-temporel

### Backend
- FastAPI
- Base de données relationnelle (SQL)

### Frontend
- Interface web de supervision

### Gouvernance ML
- MLflow (suivi des expériences)
- GitHub (versionnement du code)

---

## Données

Les données proviennent :

- Des flux vidéo du chantier (réels ou simulés)
- Des sorties du modèle de détection (bounding boxes, classes, scores)
- Des zones dangereuses définies sur plan
- Des règles métier HSE

Les données sont structurées, nettoyées et versionnées afin de garantir la reproductibilité du pipeline IA.

---

## KPIs

### KPIs IA
- mAP ≥ 0.80 pour la détection
- F1-score ≥ 0.75 pour la classification du risque
- Latence < 2 secondes

### KPIs métier
- Réduction des expositions non détectées
- Réduction des faux positifs
- Amélioration de la réactivité des responsables HSE

---

## Installation

Cloner le dépôt :

git clone https://github.com/username/projet-hse-ia.git

cd projet-hse-ia


Installer les dépendances :


pip install -r requirements.txt


Lancer l’API :


uvicorn src.api.main:app --reload


---

## Roadmap

- Détection personnes et machines
- Extraction de caractéristiques spatio-temporelles
- Modélisation du risque
- Système d’alertes
- Interface web complète
- Déploiement et monitoring

---

## Équipe

Martinien – Vision par ordinateur  
Tareq – Machine Learning et architecture  

Encadrement académique : à compléter  

---

## Licence

Projet académique à usage pédagogique.
