"""
PROJET ML : PRÉDICTION DE SURVIE DU TITANIC
Étape 1 : Analyse Exploratoire des Données (EDA) et Preprocessing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import os # Import nécessaire pour la sauvegarde
warnings.filterwarnings('ignore')

# Configuration visualisation
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================
# 1. CHARGEMENT DES DONNÉES
# ============================================

def load_data():
    """Charge les données Titanic depuis Kaggle"""
    # À télécharger depuis : https://www.kaggle.com/c/titanic/data
    # Assurez-vous d'avoir un dossier 'data/' contenant train.csv et test.csv
    try:
        train = pd.read_csv('data/train.csv')
        test = pd.read_csv('data/test.csv')
    except FileNotFoundError as e:
        print(f"❌ Erreur lors du chargement des données: {e}")
        print("Veuillez vous assurer que les fichiers 'train.csv' et 'test.csv' se trouvent dans un dossier 'data/' au même niveau que ce script.")
        raise

    print("📊 Dimensions du dataset d'entraînement:", train.shape)
    print("📊 Dimensions du dataset de test:", test.shape)
    
    return train, test

# ============================================
# 2. ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
# ============================================

def explore_data(df):
    """Analyse exploratoire complète"""
    
    print("\n" + "="*50)
    print("📈 INFORMATIONS GÉNÉRALES")
    print("="*50)
    # print(df.info()) # Commenté pour éviter un affichage trop long dans le terminal
    
    print("\n" + "="*50)
    print("📊 STATISTIQUES DESCRIPTIVES")
    print("="*50)
    print(df.describe())
    
    print("\n" + "="*50)
    print("❓ VALEURS MANQUANTES")
    print("="*50)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Colonnes': missing.index,
        'Valeurs_manquantes': missing.values,
        'Pourcentage': missing_pct.values
    })
    print(missing_df[missing_df['Valeurs_manquantes'] > 0].sort_values('Pourcentage', ascending=False))
    
    return missing_df

def visualize_survival(df):
    """Visualisations du taux de survie"""
    # Cette fonction suppose un environnement graphique actif (plt.show()).
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('📊 Analyse du Taux de Survie', fontsize=16, fontweight='bold')
    
    # 1. Distribution de survie globale
    survival_counts = df['Survived'].value_counts()
    axes[0, 0].pie(survival_counts, labels=['Décédé', 'Survivant'], autopct='%1.1f%%', 
                    colors=['#ff6b6b', '#51cf66'], startangle=90)
    axes[0, 0].set_title('Répartition Globale')
    
    # 2. Survie par Sexe
    sns.countplot(data=df, x='Sex', hue='Survived', ax=axes[0, 1], palette='Set2')
    axes[0, 1].set_title('Survie par Sexe')
    axes[0, 1].legend(['Décédé', 'Survivant'])
    
    # 3. Survie par Classe
    sns.countplot(data=df, x='Pclass', hue='Survived', ax=axes[0, 2], palette='Set1')
    axes[0, 2].set_title('Survie par Classe')
    axes[0, 2].legend(['Décédé', 'Survivant'])
    
    # 4. Distribution de l'âge
    axes[1, 0].hist([df[df['Survived']==0]['Age'].dropna(), 
                     df[df['Survived']==1]['Age'].dropna()],
                     label=['Décédé', 'Survivant'], bins=30, alpha=0.7, color=['#ff6b6b', '#51cf66'])
    axes[1, 0].set_title('Distribution de l\'Âge')
    axes[1, 0].set_xlabel('Âge')
    axes[1, 0].legend()
    
    # 5. Survie par Nombre de Siblings/Spouses
    sns.countplot(data=df, x='SibSp', hue='Survived', ax=axes[1, 1], palette='pastel')
    axes[1, 1].set_title('Survie par SibSp')
    axes[1, 1].legend(['Décédé', 'Survivant'])
    
    # 6. Distribution du prix du ticket
    axes[1, 2].hist([df[df['Survived']==0]['Fare'].dropna(), 
                     df[df['Survived']==1]['Fare'].dropna()],
                     label=['Décédé', 'Survivant'], bins=30, alpha=0.7, color=['#ff6b6b', '#51cf66'])
    axes[1, 2].set_title('Distribution du Prix')
    axes[1, 2].set_xlabel('Fare')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show() # Afficher le graphique

def correlation_analysis(df):
    """Matrice de corrélation"""
    # Sélection des colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0, 
                 square=True, linewidths=1, fmt='.2f')
    plt.title('🔥 Matrice de Corrélation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show() # Afficher le graphique

# ============================================
# 3. FEATURE ENGINEERING
# ============================================

def feature_engineering(df):
    """Création de nouvelles features"""
    
    df_copy = df.copy()
    
    # 1. Extraction du titre du nom
    df_copy['Title'] = df_copy['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Regroupement des titres rares
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    df_copy['Title'] = df_copy['Title'].map(title_mapping)
    
    # 2. Taille de la famille
    df_copy['FamilySize'] = df_copy['SibSp'] + df_copy['Parch'] + 1
    
    # 3. Est seul ?
    df_copy['IsAlone'] = (df_copy['FamilySize'] == 1).astype(int)
    
    # 4. Catégories d'âge
    df_copy['AgeGroup'] = pd.cut(df_copy['Age'], bins=[0, 12, 18, 35, 60, 100],
                                 labels=['Enfant', 'Adolescent', 'Adulte', 'Senior', 'Âgé'],
                                 right=False,  # Important: [0, 12[, [12, 18[, etc.
                                 include_lowest=True)
    
    # 5. Catégories de prix
    df_copy['FareGroup'] = pd.qcut(df_copy['Fare'].fillna(df_copy['Fare'].median()), 
                                    q=4, labels=['Bas', 'Moyen', 'Élevé', 'Très_élevé'], 
                                    duplicates='drop')
    
    # 6. Pont (première lettre de Cabin)
    df_copy['Deck'] = df_copy['Cabin'].str[0]
    
    print("\n✨ Nouvelles features créées:")
    print("- Title (titre extrait du nom)")
    print("- FamilySize (taille de la famille)")
    print("- IsAlone (passager seul)")
    print("- AgeGroup (catégories d'âge)")
    print("- FareGroup (catégories de prix)")
    print("- Deck (pont du bateau)")
    
    return df_copy

# ============================================
# 4. GESTION DES VALEURS MANQUANTES
# ============================================

def handle_missing_values(df):
    """Imputation des valeurs manquantes"""
    
    df_copy = df.copy()
    
    # Age : imputation par la médiane selon le titre et la classe
    df_copy['Age'] = df_copy.groupby(['Title', 'Pclass'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Si encore des NaN, utiliser la médiane globale
    df_copy['Age'].fillna(df_copy['Age'].median(), inplace=True)
    
    # Embarked : imputation par le mode
    df_copy['Embarked'].fillna(df_copy['Embarked'].mode()[0], inplace=True)
    
    # Fare : imputation par la médiane
    df_copy['Fare'].fillna(df_copy['Fare'].median(), inplace=True)
    
    # Cabin : création d'une catégorie "Unknown"
    df_copy['Cabin'].fillna('Unknown', inplace=True)
    df_copy['Deck'].fillna('Unknown', inplace=True)
    
    # Après imputation, il faut reconstruire AgeGroup et FareGroup car les NaN ont été remplacés
    # Reconstruire AgeGroup
    df_copy['AgeGroup'] = pd.cut(df_copy['Age'], bins=[0, 12, 18, 35, 60, 100],
                                 labels=['Enfant', 'Adolescent', 'Adulte', 'Senior', 'Âgé'],
                                 right=False, 
                                 include_lowest=True)
    # Reconstruire FareGroup (ne devrait pas changer si Fare a été rempli avec la médiane avant le qcut dans feature_engineering)
    
    
    print("\n✅ Valeurs manquantes traitées")
    print(f"Valeurs manquantes restantes: {df_copy.isnull().sum().sum()}")
    
    return df_copy

# ============================================
# 5. ENCODAGE DES VARIABLES CATÉGORIELLES
# ============================================

def encode_features(df):
    """Encodage des variables catégorielles"""
    
    df_copy = df.copy()
    
    # Label Encoding pour variables binaires
    le = LabelEncoder()
    df_copy['Sex'] = le.fit_transform(df_copy['Sex'])
    
    # One-Hot Encoding pour variables catégorielles
    # Pclass peut être traité comme une catégorielle ordinale mais pour l'encodage, 
    # on le laisse tel quel (numérique) ou on l'encode. On le laisse numérique ici.
    categorical_cols = ['Embarked', 'Title', 'Deck', 'AgeGroup', 'FareGroup']
    df_copy = pd.get_dummies(df_copy, columns=categorical_cols, drop_first=True)
    
    print("\n🔢 Encodage effectué")
    print(f"Nombre de features après encodage: {df_copy.shape[1]}")
    
    return df_copy

# ============================================
# 6. PRÉPARATION FINALE DES DONNÉES
# ============================================

def prepare_final_data(df, is_train=True):
    """Préparation finale pour l'entraînement"""
    
    # Colonnes à supprimer
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'] # SibSp et Parch sont inclus dans FamilySize/IsAlone
    
    df_clean = df.drop(drop_cols, axis=1, errors='ignore')
    
    if is_train:
        X = df_clean.drop('Survived', axis=1, errors='ignore')
        y = df_clean['Survived']
        return X, y
    else:
        return df_clean

# ============================================
# 7. FONCTION PRINCIPALE
# ============================================

def main():
    """Pipeline complet de preprocessing et sauvegarde."""
    
    print("🚢 PROJET TITANIC - PREPROCESSING")
    print("="*50)
    
    # Chargement
    train, test = load_data()
    
    # EDA
    explore_data(train)
    visualize_survival(train) # ➡️ ACTIVÉ
    correlation_analysis(train) # ➡️ ACTIVÉ
    
    # Feature Engineering
    train = feature_engineering(train)
    test = feature_engineering(test)
    
    # Gestion valeurs manquantes
    train = handle_missing_values(train)
    test = handle_missing_values(test)
    
    # Encodage
    train = encode_features(train)
    test = encode_features(test)
    
    # Préparation finale
    X_train, y_train = prepare_final_data(train, is_train=True)
    X_test = prepare_final_data(test, is_train=False)
    
    # Alignement des colonnes entre train et test
    # Cela garantit que X_train et X_test ont les mêmes colonnes après One-Hot Encoding.
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    print("\n✅ PREPROCESSING TERMINÉ")
    print(f"Shape X_train: {X_train.shape}")
    print(f"Shape y_train: {y_train.shape}")
    print(f"Shape X_test: {X_test.shape}")
    
    # ============================================
    # ÉTAPE AJOUTÉE: SAUVEGARDE DES DONNÉES TRAITÉES
    # ============================================
    output_dir = 'processed_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X_train.to_csv(os.path.join(output_dir, 'X_train_processed.csv'), index=False)
    y_train.to_frame(name='Survived').to_csv(os.path.join(output_dir, 'y_train_processed.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test_processed.csv'), index=False)
    
    print(f"💾 Données prétraitées sauvegardées dans le dossier '{output_dir}/'.")

    return X_train, y_train, X_test

# Exécution
if __name__ == "__main__":
    X_train, y_train, X_test = main()