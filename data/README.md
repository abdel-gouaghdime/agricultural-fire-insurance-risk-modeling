# Données

## Données originales

L’étude initiale s’appuie sur le challenge n°161 de la plateforme Challenge Data de l’ENS :

https://challengedata.ens.fr/login/?next=/participants/challenges/161/

Les fichiers originaux ne sont pas redistribués dans ce dépôt. Leur accès et leur utilisation restent soumis aux conditions de la plateforme et du challenge.

## Données synthétiques

`synthetic_agricultural_portfolio.csv` est un jeu de démonstration entièrement synthétique généré par `src/generate_synthetic_data.py`. Il ne contient aucun assuré, contrat ou résultat réel.

Principales variables :

- "exposure_years" : exposition annuelle ;
- "activity" : type d’activité agricole synthétique ;
- "region_risk" : catégorie géographique synthétique ;
- "building_material" : matériau principal synthétique ;
- "surface_m2" : surface assurée synthétique ;
- "insured_capital_eur" : capital assuré synthétique ;
- "fire_protection" : présence d’équipements de prévention ;
- "claim_count" : nombre synthétique de sinistres ;
- "average_claim_cost_eur" : coût moyen synthétique ;
- "total_claim_cost_eur": coût total synthétique.
