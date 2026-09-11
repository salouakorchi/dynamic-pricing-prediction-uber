import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# ===========================================================
# ÉTAPE 1 : CHARGEMENT ET NETTOYAGE
# ===========================================================
df = pd.read_csv("rideshare_kaggle.csv")
df = df.dropna(subset=['price'])
df = df[df['cab_type'] == 'Uber']
df = df.sort_values('timestamp')

col_garder = ['timestamp', 'hour', 'name', 'distance', 'surge_multiplier',
              'temperature', 'precipProbability', 'price']
df = df[col_garder]

# ===========================================================
# ÉTAPE 2 : FEATURE ENGINEERING
# ===========================================================
df['hour_sin']         = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos']         = np.cos(2 * np.pi * df['hour'] / 24)
df['distance_x_surge'] = df['distance'] * df['surge_multiplier']
df['surge_flag']       = (df['surge_multiplier'] > 1.0).astype(int)
df['distance_sq']      = df['distance'] ** 2
df['surge_sq']         = df['surge_multiplier'] ** 2
df['rain_x_surge']     = df['precipProbability'] * df['surge_multiplier']

df = df.drop(columns=['hour', 'timestamp'])
df = pd.get_dummies(df, columns=['name'], dtype=int)

colonnes = [c for c in df.columns if c != 'price'] + ['price']
df = df[colonnes]

print(f"\nDimensions dataset : {df.shape}")
print(f"Features utilisées : {df.shape[1] - 1}")

# ✅ SAUVEGARDE DES NOMS DE COLONNES (indispensable pour l'appli web)
feature_columns = [c for c in df.columns if c != 'price']
joblib.dump(feature_columns, 'feature_columns.pkl')
print(f"Colonnes sauvegardées : {feature_columns}")

# ===========================================================
# ÉTAPE 3 : SCALERS SÉPARÉS X et Y
# ===========================================================
features = df.drop(columns=['price']).values
cibles   = df['price'].values.reshape(-1, 1)

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))

X = scaler_X.fit_transform(features)
Y = scaler_Y.fit_transform(cibles)

# ✅ SAUVEGARDE DES SCALERS (indispensable pour l'appli web)
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_Y, 'scaler_Y.pkl')
print("✅ Scalers sauvegardés : scaler_X.pkl, scaler_Y.pkl")

print(f"X={X.shape}, Y={Y.shape}")

# ===========================================================
# ÉTAPE 4 : SPLIT CHRONOLOGIQUE 80% TRAIN / 20% TEST
# ===========================================================
limite   = int(len(X) * 0.8)
X_train  = X[:limite];  Y_train = Y[:limite]
X_test   = X[limite:];  Y_test  = Y[limite:]

print(f"\nTrain : {X_train.shape[0]} trajets")
print(f"Test  : {X_test.shape[0]} trajets")

# ===========================================================
# ÉTAPE 5 : ARCHITECTURE DENSE + BatchNorm
# ===========================================================
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1)
])

model.summary()

def quantile_loss(q):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    return loss

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=quantile_loss(0.5)
)

# ===========================================================
# ÉTAPE 6 : ENTRAÎNEMENT
# ===========================================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
]

print("\n--- Entraînement en cours ---")
historique = model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=128,
    validation_data=(X_test, Y_test),
    callbacks=callbacks,
    verbose=1
)

model.save("uber_prix_dynamique_v5.keras")
print("\n✅ Modèle sauvegardé : uber_prix_dynamique_v5.keras")

# ===========================================================
# ÉTAPE 7 : PRÉDICTIONS → reconversion en dollars
# ===========================================================
predictions_norm = model.predict(X_test)
prix_predits     = scaler_Y.inverse_transform(predictions_norm)[:, 0]
prix_reels       = scaler_Y.inverse_transform(Y_test.reshape(-1, 1))[:, 0]

# ===========================================================
# ÉTAPE 8 : MÉTRIQUES COMPLÈTES
# ===========================================================
mae        = mean_absolute_error(prix_reels, prix_predits)
rmse       = np.sqrt(mean_squared_error(prix_reels, prix_predits))
r2         = r2_score(prix_reels, prix_predits)
erreurs    = np.abs(prix_reels - prix_predits)
accuracy_2 = np.mean(erreurs <= 2.0) * 100
accuracy_5 = np.mean(erreurs <= 5.0) * 100
mape       = np.mean(erreurs / np.where(prix_reels == 0, 1, prix_reels)) * 100

print("\n" + "="*60)
print("   BILAN DE PERFORMANCE — MODÈLE V5")
print("="*60)
print(f"  MAE      : {mae:.2f} $")
print(f"  RMSE     : {rmse:.2f} $")
print(f"  R²       : {r2:.4f}")
print(f"  MAPE     : {mape:.2f} %")
print(f"  Accuracy ±2$ : {accuracy_2:.2f} %")
print(f"  Accuracy ±5$ : {accuracy_5:.2f} %")
print("="*60)

if accuracy_2 >= 70 and r2 >= 0.90:
    print("  🟢 EXCELLENT — Modèle prêt pour la production")
elif accuracy_2 >= 50 and r2 >= 0.80:
    print("  🟡 BON — Résultats solides")
else:
    print("  🔴 FAIBLE — Revoir les features")

print("\n💰 PRÉDICTION vs RÉALITÉ (20 premiers trajets)")
print(f"{'#':<6} {'IA ($)':<12} {'Réel ($)':<12} {'Erreur ($)':<12} {'Statut'}")
print("-" * 55)
for i in range(20):
    ia    = prix_predits[i]
    reel  = prix_reels[i]
    diff  = abs(ia - reel)
    statut = "✓  Précis" if diff < 2.0 else ("~  Proche" if diff < 5.0 else "✗  Loin")
    print(f"  {i+1:<4} {ia:<12.2f} {reel:<12.2f} {diff:<12.2f} {statut}")