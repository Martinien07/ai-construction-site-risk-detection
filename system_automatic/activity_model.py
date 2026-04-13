import joblib


class ActivityPredictor:

    def __init__(self, model_path):

        self.model = joblib.load(model_path)

    def predict(self, df_features):

        predictions = self.model.predict(df_features)

        return predictions