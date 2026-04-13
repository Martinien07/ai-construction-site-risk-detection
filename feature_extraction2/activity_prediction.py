import pandas as pd


class ActivityPredictor:

    def __init__(self, model):
        """
        model : modèle ML déjà chargé (RandomForest, XGBoost, etc.)
        """
        self.model = model
        self.feature_names = model.feature_names_in_


    def predict(self, df_ml: pd.DataFrame):

        print("\n========== ACTIVITY PREDICTION ==========")

        # --------------------------------------------------
        # 1 vérifier que les features existent
        # --------------------------------------------------

        missing = [f for f in self.feature_names if f not in df_ml.columns]

        if missing:
            raise ValueError(f"Features manquantes pour le modèle : {missing}")

        X = df_ml[self.feature_names]

        print("Features utilisées par le modèle :")
        print(self.feature_names)


        # --------------------------------------------------
        # 2 prédiction activité
        # --------------------------------------------------

        predictions = self.model.predict(X)


        # --------------------------------------------------
        # 3 probabilités
        # --------------------------------------------------

        if hasattr(self.model, "predict_proba"):

            probs = self.model.predict_proba(X)

            max_prob = probs.max(axis=1)

        else:

            max_prob = [None] * len(predictions)


        # --------------------------------------------------
        # 4 ajouter au dataframe
        # --------------------------------------------------

        df_ml["activity_pred"] = predictions
        df_ml["activity_confidence"] = max_prob


        print("\nActivités détectées :")
        print(df_ml["activity_pred"].value_counts())

        return df_ml