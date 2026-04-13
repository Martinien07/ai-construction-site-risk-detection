import pandas as pd
import json

class RiskEngine:
    def __init__(self, rules_path):
        with open(rules_path, 'r') as f:
            self.config = json.load(f)
        self.general_rules = self.config['general_requirements']
        self.activity_rules = self.config['activity_rules']
        self.risk_levels = self.config['risk_levels']

    def _get_level_name(self, score):
        """Détermine le libellé du risque selon le score."""
        for level, bounds in self.risk_levels.items():
            if bounds['min'] <= score <= bounds['max']:
                return level
        return "CRITIQUE" if score > 70 else "FAIBLE"

    def process_dataframe(self, df):
        """Analyse le dataframe et ajoute les colonnes de risque."""
        
        scores = []
        levels = []
        messages = []
        weights_detail = []

        for _, row in df.iterrows():
            total_score = 0
            row_msgs = []
            row_weights = []
            veto_active = False
            
            # 1. Identification de l'activité (basé sur activity_pred)
            act_id = str(int(row['activity_pred']))
            act_config = self.activity_rules.get(act_id, {"name": "Inconnue", "base_risk": 0})
            
            total_score += act_config.get("base_risk", 0)
            if act_config.get("base_risk", 0) > 0:
                row_weights.append(f"Base_{act_config['name']}:{act_config['base_risk']}")

            # 2. Vérification General Requirements (EPI)
            for rule_id, rule in self.general_rules.items():
                val = row.get(rule['feature'], 1.0)
                if val < rule['threshold']:
                    total_score += rule['weight']
                    row_msgs.append(rule['msg'])
                    row_weights.append(f"{rule_id}:{rule['weight']}")
                    if rule.get('critical_veto'):
                        veto_active = True

            # 3. Vérification des règles spécifiques à l'activité
            if act_id in self.activity_rules:
                for check in self.activity_rules[act_id].get('checks', []):
                    val = row.get(check['feature'], 0.0)
                    breach = False
                    
                    if 'max' in check and val > check['max']:
                        breach = True
                    elif 'min' in check and val < check['min']:
                        breach = True
                        
                    if breach:
                        total_score += check['weight']
                        row_msgs.append(check['msg'])
                        row_weights.append(f"{check['feature']}:{check['weight']}")
                        if check.get('critical_veto'):
                            veto_active = True

            # 4. Calcul final avec plafonnement
            final_score = 100 if veto_active else min(100, total_score)
            
            scores.append(final_score)
            levels.append(self._get_level_name(final_score))
            messages.append(" | ".join(row_msgs) if row_msgs else "Conforme")
            weights_detail.append(", ".join(row_weights) if row_weights else "Aucun")

        # Ajout des colonnes au DataFrame
        df['risk_score'] = scores
        df['risk_level'] = levels
        df['risk_messages'] = messages
        df['risk_weights'] = weights_detail
        
        return df
