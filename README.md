# Wine Quality Prediction
API applied to AI
TP note pour le 18/12

## Pour commencer

Téléchargez les documents :
-main.py
-model.py
-Wines.csv
-model.pkl


## Pré-requis
Liste des packages nécessaires :
-fastapi
-pydantic
-sklearn
-pandas
-pickle
-csv

## Démarrage
Lancer sur votre terminal la commande : 

uvicorn main:app --reload

Puis rendez-vous sur le lien qui s'affiche.
En rajoutant /docs à l'adresse vous verrez apparaître toutes les commandes API disponibles.

GET/api/model : Permez de lancer le téléchargement du fichier model.pkl conenant le modèle préentrainé.
PUT/api/model : Rentrez les caractéristiques d'un nouveau vin dans la base de données.
GET/api/predict : Affiche les caractéristiques du meilleur vin.
POST/api/predict : Rentrez les caractéristiques d'un vin pour prédire sa qualité.
GET/api/model/description : Affiche les paramètres du modèle ainsi que sa précision.
POST/api/model/retrain : Relance l'entraînement du modèle (il est conseillé d'avoir utilisé la commande PUT/api/model avant si vous espérez un changement. 


## Auteurs
Arboin Mathieu
Ruau Nicolas
