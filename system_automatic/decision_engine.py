import pandas as pd

class DecisionEngine:
    def __init__(self, frequency_threshold=0.20):
        """
        Args:
            frequency_threshold (float): Seuil de présence minimum (ex: 0.20 pour 20%).
        """
        self.frequency_threshold = frequency_threshold

    def aggregate_decisions(self, df):
        """
        Filtre les faux positifs et agrège les risques par activité.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        # 1. Calculer la fréquence de chaque activité par rapport au total des fenêtres
        total_windows = len(df)
        activity_counts = df['activity_pred'].value_counts()
        
        # Filtrer les activités qui respectent le seuil (ex: >= 20%)
        valid_activities = activity_counts[activity_counts / total_windows >= self.frequency_threshold].index.tolist()
        
        if not valid_activities:
            print(f" Aucune activité n'a atteint le seuil de {self.frequency_threshold*100}%")
            return pd.DataFrame()

        # 2. Filtrer le DF original pour ne garder que les activités validées
        df_filtered = df[df['activity_pred'].isin(valid_activities)].copy()

        # 3. Agrégation : Pour chaque (Caméra, Activité), on prend le risque MAX
        # Cela permet de conserver l'alerte la plus grave si l'activité est répétée.
        df_final = df_filtered.sort_values('risk_score', ascending=False).drop_duplicates(
            subset=['camera_id', 'activity_pred'], 
            keep='first'
        )

        # Nettoyage et tri final
        return df_final.reset_index(drop=True)