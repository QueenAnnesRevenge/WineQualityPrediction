# Wine Quality Prediction

API applied to AI

TP noté pour le 18/12

## Pré-requis

Téléchargez les documents :

- main.py

- model.py

- Wines.csv

- model.pkl


Liste des packages nécessaires :

- fastapi

- pydantic

- sklearn

- pandas

- pickle

- csv

## Utilisation

Lancez sur votre terminal la commande : 

uvicorn main:app --reload

Puis rendez-vous sur le lien qui s'affiche.

En rajoutant /docs à l'adresse vous verrez apparaître toutes les commandes API disponibles.

GET/api/model : Permez de lancer le téléchargement du fichier model.pkl conenant le modèle préentrainé.

PUT/api/model : Rentrez les caractéristiques d'un nouveau vin dans la base de données. (le fichier Wines.csv doit être fermer + le prompt doit etre au début de la ligne suivant la dernière entrée)

GET/api/predict : Affiche les caractéristiques du meilleur vin.

POST/api/predict : Rentrez les caractéristiques d'un vin pour prédire sa qualité.

GET/api/model/description : Affiche les paramètres du modèle ainsi que sa précision.

POST/api/model/retrain : Relance l'entraînement du modèle.
(il est conseillé d'avoir utilisé la commande PUT/api/model avant si vous espérez un changement dans la prédiction) 


## Auteurs

Arboin Mathieu

Ruau Nicolas
