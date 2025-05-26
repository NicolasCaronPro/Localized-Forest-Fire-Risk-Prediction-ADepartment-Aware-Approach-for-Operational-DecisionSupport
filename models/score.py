from sklearn.metrics import f1_score
import numpy as np

def calculate_area_under_curve(y_values):
    """
    Calcule l'aire sous la courbe pour une série de valeurs données (méthode de trapèze).

    :param y_values: Valeurs sur l'axe des ordonnées pour calculer l'aire sous la courbe.
    :return: Aire sous la courbe.
    """
    return np.trapz(y_values, dx=1)

def iou_score(y_true, y_pred):
    """
    Calcule les scores (aire commune, union, sous-prédiction, sur-prédiction) entre deux signaux.

    Args:
        t (np.array): Tableau de temps ou indices (axe x).
        y_pred (np.array): Signal prédiction (rouge).
        y_true (np.array): Signal vérité terrain (bleu).

    Returns:
        dict: Dictionnaire contenant les scores calculés.
    """

    if isinstance(y_pred, DMatrix):
        y_pred = np.copy(y_pred.get_data().toarray())

    if isinstance(y_true, DMatrix):
        y_true = np.copy(y_true.get_label())

    y_pred = np.reshape(y_pred, y_true.shape)
    # Calcul des différentes aires
    intersection = np.trapz(np.minimum(y_pred, y_true))  # Aire commune
    union = np.trapz(np.maximum(y_pred, y_true))         # Aire d'union

    return intersection / union if union > 0 else 0

def under_prediction_score(y_true, y_pred):
    """
    Calcule le score de sous-prédiction, c'est-à-dire l'aire correspondant
    aux valeurs où la prédiction est inférieure à la vérité terrain,
    normalisée par l'union des deux signaux.

    Args:
        y_true (np.array): Signal vérité terrain.
        y_pred (np.array): Signal prédiction.

    Returns:
        float: Score de sous-prédiction.
    """

    y_pred = np.reshape(y_pred, y_true.shape)
    # Calcul de l'aire de sous-prédiction
    under_prediction_area = np.trapz(np.maximum(y_true - y_pred, 0))  # Valeurs positives où y_true > y_pred
    
    # Calcul de l'union (le maximum des deux signaux à chaque point)
    union_area = np.trapz(np.maximum(y_true, y_pred))  # Union des signaux
    
    return under_prediction_area / union_area if union_area > 0 else 0

def over_prediction_score(y_true, y_pred):
    """
    Calcule le score de sur-prédiction, c'est-à-dire l'aire correspondant
    aux valeurs où la prédiction est supérieure à la vérité terrain,
    normalisée par l'union des deux signaux.

    Args:
        y_true (np.array): Signal vérité terrain.
        y_pred (np.array): Signal prédiction.

    Returns:
        float: Score de sur-prédiction.
    """
    y_pred = np.reshape(y_pred, y_true.shape)
    # Calcul de l'aire de sur-prédiction
    over_prediction_area = np.trapz(np.maximum(y_pred - y_true, 0))  # Valeurs positives où y_pred > y_true
    
    # Calcul de l'union (le maximum des deux signaux à chaque point)
    union_area = np.trapz(np.maximum(y_true, y_pred))  # Union des signaux
    
    return over_prediction_area / union_area if union_area > 0 else 0

def evaluate_metrics(df, y_true_col='target', y_pred=None):
    """
    Calcule l'IoU et le F1-score sur chaque département, puis calcule l'aire sous la courbe normalisée (aire / aire maximale).
    
    :param dff: DataFrame contenant les colonnes ['Department', 'Scale', 'nbsinister', 'target']
    :param dataset: Nom du dataset à filtrer
    :param y_true_col: Colonne représentant les cibles réelles
    :param y_pred: Liste ou tableau des prédictions
    :param metric: Choix de la métrique ('IoU' ou 'F1')
    :param top: Nombre de départements à afficher (ou 'all' pour tout afficher)
    :return: Dictionnaire contenant l'aire normalisée pour chaque modèle.
    """
    
    # Trier les valeurs par 'nbsinister' décroissant
    #df_sorted = df.sort_values(by='nbsinister', ascending=False)
    df_sorted = df

    y_true = df[y_true_col]
    
    iou = iou_score(y_true, y_pred)
    f1 = f1_score((y_true > 0).astype(int), (y_pred > 0).astype(int))

    under = under_prediction_score(y_true, y_pred)
    over = over_prediction_score(y_true, y_pred)

    # Initialiser un dictionnaire pour les résultats
    results = {'iou' : iou, 'f1' : f1, 'under' : under, 'over' : over}

    # Calculer l'IoU et F1 pour chaque département
    IoU_scores = []
    F1_scores = []
    
    for i, department in enumerate(df_sorted['departement'].unique()):
        # Extraire les valeurs pour chaque département
        y_true = df_sorted[df_sorted['departement'] == department][y_true_col].values
        if np.all(y_true == 0):
            continue
        y_pred_department = y_pred[df_sorted['departement'] == department]  # Récupérer les prédictions associées au département
        
        # Calcul des scores IoU et F1
        IoU = iou_score(y_true, y_pred_department)
        F1 = f1_score(y_true > 0, y_pred_department > 0)
        
        IoU_scores.append(IoU)
        F1_scores.append(F1)
    
    # Calcul de l'aire maximale possible (cas parfait où toutes les prédictions sont correctes)
    max_area = np.trapz(np.ones(len(df_sorted)), dx=1)
    
    # Calcul de l'aire sous la courbe pour l'IoU et le F1
    IoU_area = calculate_area_under_curve(IoU_scores)
    F1_area = calculate_area_under_curve(F1_scores)
    
    # Normalisation par l'aire maximale
    normalized_IoU = IoU_area / max_area if max_area > 0 else 0
    normalized_F1 = F1_area / max_area if max_area > 0 else 0
    
    # Stocker les résultats dans le dictionnaire
    results['normalized_iou'] = normalized_IoU
    results['normalized_f1'] = normalized_F1
    
    return results