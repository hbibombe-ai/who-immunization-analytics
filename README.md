# WHO Immunization Analytics

Application Streamlit de démonstration pour analyser les données mensuelles du PEV.

## Fonctions

- Couverture BCG, Penta1, Penta3, VPI, Rougeole et HPV
- Taux d'abandon Penta1-Penta3
- Estimation des enfants zéro dose
- Contrôles automatiques de qualité des données
- Classement et carte des districts prioritaires
- Explication analytique par district
- Téléchargement des tableaux CSV et des graphiques PNG

## Données

L'application démarre avec des données entièrement simulées pour 2024-2026. Elles servent uniquement à tester les calculs et l'interface. Un fichier Excel ou CSV réel peut être importé depuis la barre latérale.

## Lancement local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Déploiement

Déposer tous les fichiers de ce dossier dans un dépôt GitHub, puis sélectionner `app.py` dans Streamlit Community Cloud.
