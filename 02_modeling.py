"""
PROJET ML : PRÉDICTION DE SURVIE DU TITANIC
Étape 2 : Évaluation d'une sélection de modèles de Classification
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import warnings
import os

# Ignorer les avertissements pour une sortie plus propre
warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = 'processed_data'
RANDOM_SEED = 42

# ==============================================================================
# 1. CHARGEMENT DES DONNÉES PRÉTRAITÉES
# ==============================================================================

def load_data():
    """
    Charge les jeux de données prétraités depuis le dossier 'processed_data'.
    """
    X_train_path = os.path.join(OUTPUT_DIR, 'X_train_processed.csv')
    y_train_path = os.path.join(OUTPUT_DIR, 'y_train_processed.csv') # Correction du nom de fichier pour la cible
    X_test_path = os.path.join(OUTPUT_DIR, 'X_test_processed.csv')
    
    X_train, y_train, X_test = pd.DataFrame(), pd.Series(dtype='int'), pd.DataFrame()
    
    if not (os.path.exists(X_train_path) and os.path.exists(y_train_path) and os.path.exists(X_test_path)):
        print(f"ATTENTION: Les fichiers de données prétraitées sont introuvables dans le dossier '{OUTPUT_DIR}'.")
        print("Le script va créer des données SIMULÉES pour la démonstration des modèles.")
        
        # Créer des données simulées basées sur les shapes et la structure attendues
        n_train = 891
        n_test = 418
        n_features = 29
        
        X_train = pd.DataFrame(np.random.rand(n_train, n_features), 
                                columns=[f'feature_{i}' for i in range(n_features)])
        y_train = pd.Series(np.random.randint(0, 2, n_train))
        X_test = pd.DataFrame(np.random.rand(n_test, n_features), 
                              columns=[f'feature_{i}' for i in range(n_features)])
        
        # S'assurer que les colonnes sont identiques
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        
    else:
        try:
            # Charger les données réelles
            X_train = pd.read_csv(X_train_path)
            # IMPORTANT: Le script 01_preprocessing a sauvé y_train dans un fichier avec une colonne 'Survived'
            y_train = pd.read_csv(y_train_path)['Survived'] 
            X_test = pd.read_csv(X_test_path)
            
            # Revenir à la simulation si le fichier y_train est vide après lecture
            if X_train.empty or y_train.empty or X_test.empty:
                 raise ValueError("Les DataFrames chargés sont vides.")

        except Exception as e:
            print(f"Erreur lors du chargement des données réelles: {e}")
            print("Retour à la SIMULATION de données pour l'exécution.")
            
            # Créer des données simulées en cas d'échec de lecture
            n_train = 891
            n_test = 418
            n_features = 29
            X_train = pd.DataFrame(np.random.rand(n_train, n_features), columns=[f'feature_{i}' for i in range(n_features)])
            y_train = pd.Series(np.random.randint(0, 2, n_train))
            X_test = pd.DataFrame(np.random.rand(n_test, n_features), columns=[f'feature_{i}' for i in range(n_features)])
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


    print(f"\n✅ Données chargées et prêtes pour l'entraînement.")
    print(f"Shape X_train: {X_train.shape}")
    print(f"Shape y_train: {y_train.shape}")
    print(f"Shape X_test: {X_test.shape}")
    
    return X_train, y_train, X_test


# ==============================================================================
# 2. ÉVALUATION DES MODÈLES
# ==============================================================================

def evaluate_models(X_train, y_train):
    """
    Initialise et évalue une sélection de modèles de classification
    en utilisant la validation croisée stratifiée.
    """
    if X_train.empty or y_train.empty:
        print("\n❌ Impossible d'évaluer les modèles: Les données d'entraînement sont vides.")
        return None

    print("\n==================================================")
    print("🧠 ÉVALUATION DES MODÈLES AVEC VALIDATION CROISÉE")
    print("==================================================")

    # Définition des modèles à tester
    classifiers = {
        "Régression Logistique": LogisticRegression(random_state=RANDOM_SEED, max_iter=200),
        "K-plus Proches Voisins": KNeighborsClassifier(n_neighbors=5),
        # SVC avec probabilité activée pour pouvoir comparer les probabilités si besoin
        "Machine à Vecteurs de Support": SVC(random_state=RANDOM_SEED, probability=True), 
        "Arbre de Décision": DecisionTreeClassifier(random_state=RANDOM_SEED),
        # Random Forest est souvent un bon point de départ
        "Forêt Aléatoire": RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=100),
        # Gradient Boosting est souvent très performant
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
        "Naïf Bayes Gaussien": GaussianNB()
    }

    # Configuration de la validation croisée
    cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    
    results = {}

    for name, model in classifiers.items():
        try:
            # Calculer les scores de validation croisée
            # Utiliser n_jobs=-1 pour paralléliser l'entraînement et accélérer le processus
            scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='accuracy', n_jobs=-1) 
            
            # Enregistrer la moyenne et l'écart-type des scores
            results[name] = {
                'Moyenne Précision (CV)': scores.mean(),
                'Écart-type (CV)': scores.std()
            }
            
            # Affichage des résultats intermédiaires
            print(f"  {name:<30}: {scores.mean():.4f} (+/- {scores.std():.4f})")
            
        except Exception as e:
            print(f"  {name:<30}: ÉCHEC de l'entraînement. Erreur: {e}")
            results[name] = {'Moyenne Précision (CV)': 0.0, 'Écart-type (CV)': 0.0}

    # Trier et afficher le meilleur modèle
    best_model_name = max(results, key=lambda k: results[k]['Moyenne Précision (CV)'])
    best_score = results[best_model_name]['Moyenne Précision (CV)']

    print("\n==================================================")
    print(f"🏆 MEILLEUR MODÈLE (Précision Moyenne): {best_model_name}")
    print(f"   Score de Précision Moyen: {best_score:.4f}")
    print("==================================================")

    return classifiers[best_model_name]

# ==============================================================================
# 3. ENTRAÎNEMENT FINAL ET PRÉDICTION
# ==============================================================================

def final_prediction(best_model, X_train, y_train, X_test):
    """
    Entraîne le meilleur modèle sur l'intégralité du jeu d'entraînement
    et génère les prédictions pour le jeu de test.
    """
    if X_train.empty or y_train.empty or X_test.empty:
        print("\n❌ Impossible de faire les prédictions finales: Jeux de données incomplets.")
        return

    print("\n==================================================")
    print(f"🚀 ENTRAÎNEMENT FINAL et PRÉDICTION")
    print("==================================================")
    
    # 1. Entraînement final
    print(f"Entraînement du modèle {best_model.__class__.__name__} sur toutes les données d'entraînement...")
    best_model.fit(X_train, y_train)
    
    # 2. Prédictions
    print("Génération des prédictions pour le jeu de test...")
    predictions = best_model.predict(X_test)
    
    # 3. Création du fichier de soumission (format Kaggle)
    
    # Tenter de charger les PassengerId du fichier de test original
    try:
        test_raw = pd.read_csv('data/test.csv')
        passenger_ids = test_raw['PassengerId']
        
    except FileNotFoundError:
        print("\n⚠️ Impossible de charger 'data/test.csv' pour récupérer les PassengerId. Utilisation d'IDs simulés.")
        passenger_ids = range(892, 892 + len(predictions))
    
    submission_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions.astype(int)
    })
    
    submission_file = 'submission_titanic.csv'
    submission_df.to_csv(submission_file, index=False)
    
    print(f"\n✅ PRÉDICTIONS TERMINÉES et Fichier de soumission créé: {submission_file}")
    print(f"   Premières 5 prédictions: {predictions[:5]}")
    print(f"   Shape du fichier de soumission: {submission_df.shape}")
    print("==================================================")


if __name__ == '__main__':
    # Étape 1: Charger les données
    X_train, y_train, X_test = load_data()

    # Étape 2: Évaluer les modèles
    if not X_train.empty:
        best_model = evaluate_models(X_train, y_train)

        # Étape 3: Entraînement final et soumission
        final_prediction(best_model, X_train, y_train, X_test)

    print("\nFin du script 02_model_evaluation.py")