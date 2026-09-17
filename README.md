# Modélisation du risque incendie en assurance agricole

--- Analyse statistique et modélisation de la fréquence, du coût moyen et de la prime pure---

Projet de portfolio orienté: Data Analyst, chargé d’études statistiques et modélisation actuarielle. Il présente une démarche fréquence–coût appliquée au risque incendie d’un portefeuille d’assurance agricole.

- Transparence des données : les données originales du challenge ne sont pas redistribuées dans ce dépôt. La démonstration reproductible utilise un portefeuille entièrement synthétique, sans assuré ni contrat réel. 
Les résultats affichés ci-dessous concernent uniquement cette démonstration.

## Contexte métier

Le projet s’appuie sur le challenge : Prédiction du risque incendie en assurance agricole, proposé par Crédit Agricole Assurances sur la plateforme [Challenge Data de l’ENS](https://challengedata.ens.fr/login/?next=/participants/challenges/161/).

Le risque incendie représente un enjeu important pour un contrat multirisque agricole. L’objectif analytique consiste à estimer séparément :

1. la fréquence attendue des sinistres ;
2. leur coût moyen attendu ;
3. la prime pure obtenue en combinant fréquence, coût moyen et exposition.

		""" Prime pure attendue=Frequence predite×Coût moyen predit × Exposition"""

La charge attendue d’un contrat est ensuite obtenue en tenant compte de son exposition :

{Charge attendue} = {Prime pure attendue} \{Exposition}

## Ma contribution

J’ai conduit le volet analytique du projet, depuis la préparation des données jusqu’à l’interprétation des résultats :

- exploration d’un portefeuille volumineux et contrôle de la qualité des données ;
- traitement des valeurs manquantes et des variables catégorielles ;
- analyse descriptive de la fréquence et du coût des sinistres ;
- sélection de variables contractuelles, géographiques et de prévention ;
- comparaison de modèles statistiques et de machine learning ;
- construction d’une approche fréquence–coût pour estimer la prime pure ;
- visualisation des résultats et interprétation des facteurs de risque.

La version publique améliore la reproductibilité de l’étude : séparation entraînement–test, prétraitement intégré aux pipelines, évaluation hors échantillon et génération de données synthétiques documentée.

## Outils

- Langage : Python
- Environnement : Jupyter Notebook / Google Colab
- Manipulation : pandas, NumPy
- Visualisation : Matplotlib, Seaborn
- Modélisation : scikit-learn
- Méthodes : GLM Poisson, GLM Tweedie, Gradient Boosting, Random Forest

## Démarche analytique

### 1. Fréquence des sinistres

La cible correspond au nombre annuel de sinistres rapporté à l’exposition. Deux modèles sont comparés :

- GLM Poisson, modèle actuariel interprétable servant de référence ;
- Gradient Boosting avec perte de Poisson, capable de représenter des relations non linéaires.

### 2. Coût moyen

Le coût moyen est modélisé uniquement sur les contrats ayant enregistré au moins un sinistre :

- GLM Tweedie, adapté aux coûts positifs et asymétriques ;
- Random Forest, utilisé comme modèle non linéaire de comparaison.

### 3. Prime pure

Les meilleures prédictions de fréquence et de coût sont combinées au niveau du contrat. L’évaluation finale est réalisée sur un échantillon de test qui n’a pas servi à entraîner les modèles.

## Résultats de la démonstration

La démonstration a été exécutée sur 8 000 contrats synthétiques, dont 6 000 pour l’entraînement et 2 000 pour le test.

| Composante | Modèle | MAE | Déviance |
|---|---|---:|---:|
| Fréquence | GLM Poisson | 0,115 | 0,390 |
| Fréquence | Gradient Boosting Poisson | 0,108 | 0,414 |
| Coût moyen | GLM Tweedie | 8 409 € | 75,953 |
| Coût moyen | Random Forest | 8 496 € | 72,679 |

La sélection repose sur la déviance adaptée à chaque composante :

- GLM Poisson retenu pour la fréquence ;
- Random Forest retenu pour le coût moyen ;
- ratio de calibration agrégé sur le test : 94,7 % ;
- MAE individuelle de la prime pure : 1 567 €.

[Distribution des cibles](figures/target_distributions.png)

## Lecture métier

- La séparation fréquence–coût permet d’identifier si un profil est risqué parce qu’il génère davantage de sinistres ou parce que ceux-ci sont plus coûteux.
- Le GLM fournit une référence interprétable et compatible avec les pratiques actuarielles.
- Les modèles d’ensemble captent des non-linéarités et interactions, mais doivent rester accompagnés d’outils d’interprétation.
- La prévention, la nature des bâtiments, leur ancienneté, les capitaux assurés et la distance aux services de secours sont des familles de variables pertinentes à étudier.
- Une bonne performance individuelle ne suffit pas : la calibration au niveau du portefeuille doit également être contrôlée.

## Limites

- Les données publiques de démonstration sont synthétiques et simplifient la réalité assurantielle.
- Les résultats dépendent du mécanisme utilisé pour simuler les sinistres.
- Le projet ne traite pas la dérive temporelle, l’inflation des coûts, la dépendance spatiale ou les événements catastrophiques.
- Une utilisation opérationnelle nécessiterait une validation actuarielle, juridique et métier sur les données autorisées.

## Structure du dépôt

```text
agricultural-fire-insurance-risk-modeling/
├── data/
│   ├── README.md
│   └── synthetic_agricultural_portfolio.csv
├── notebooks/
│   └── agricultural_fire_risk_modeling.ipynb
├── src/
│   ├── generate_synthetic_data.py
│   └── run_analysis.py
├── figures/
│   ├── frequency_feature_importance.png
│   ├── model_comparison.png
│   └── target_distributions.png
├── results/
│   ├── frequency_feature_importance.csv
│   ├── model_metrics.csv
│   ├── summary.json
│   └── test_predictions.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation et exécution

```bash
git clone https://github.com/abdel-gouaghdime/agricultural-fire-insurance-risk-modeling.git
cd agricultural-fire-insurance-risk-modeling
python -m venv .venv
```

Sous Windows :

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python src/generate_synthetic_data.py
python src/run_analysis.py
```

Le notebook fournit une lecture guidée de l’analyse. Les scripts `src/` constituent la version reproductible à exécuter.

## Réponse courte pour un entretien

> J’ai utilisé Python pour analyser un portefeuille d’assurance agricole et construire une approche fréquence–coût. J’ai comparé un GLM Poisson à un modèle de boosting pour la fréquence, puis un GLM Tweedie à un Random Forest pour le coût moyen. Les meilleures prédictions sont combinées avec l’exposition afin d’estimer la prime pure et la charge attendue. J’ai évalué les modèles sur un échantillon séparé et contrôlé à la fois leur erreur individuelle et leur calibration au niveau du portefeuille.

## English summary

This portfolio project presents a frequency–severity framework for agricultural fire insurance. It compares interpretable actuarial GLMs with tree-based machine-learning models and combines out-of-sample frequency and severity predictions to estimate pure premium. The original Challenge Data files are not redistributed; the public demonstration uses fully synthetic data and reports demonstration-only results.

## Auteur

**Abdelilah Gouaghdime**  
Data Analyst · Chargé d’études statistiques · Modélisation actuarielle  
[GitHub](https://github.com/abdel-gouaghdime)
