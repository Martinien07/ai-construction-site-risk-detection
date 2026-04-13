

import json
rules_config = json.load(open("../rules_identification/config/hse_rules.json"))

class HSERuleEngine:
    """
    Moteur d'évaluation des règles HSE.
    """

    def __init__(self, rules_config: dict):
        """
        Initialise le moteur avec la configuration des règles.

        :param rules_config: dictionnaire chargé depuis hse_rules.json
        """
        self.rules = rules_config.get("rules", [])

    @staticmethod
    def _evaluate_condition(feature_value, operator, threshold):
        if feature_value is None:
            return False
        if operator == "<":
            return feature_value < threshold
        if operator == "<=":
            return feature_value <= threshold
        if operator == ">":
            return feature_value > threshold
        if operator == ">=":
            return feature_value >= threshold
        if operator == "==":
            return feature_value == threshold
        return False

    def evaluate(self, features: dict, activity_id: int):
        """
        Évalue les règles HSE pour une situation donnée.

        :param features: dictionnaire des caractéristiques extraites
        :param activity_id: activité prédite (int)
        :return: dict contenant violations, EPI manquants et score de risque
        """

        violated_rules = []
        missing_epi_global = set()
        total_risk_score = 0

        for rule in self.rules:

            # Filtrage par activité
            if rule["activity_id"] != activity_id:
                continue

            # Évaluation des conditions
            conditions_met = True
            for cond in rule["conditions"]:
                feature_name = cond["feature"]
                operator = cond["operator"]
                threshold = cond["value"]

                feature_value = features.get(feature_name)
                if not self._evaluate_condition(feature_value, operator, threshold):
                    conditions_met = False
                    break

            if not conditions_met:
                continue

            # ➜ RÈGLE VIOLÉE
            violated_rule = {
                "rule_id": rule["rule_id"],
                "name": rule["name"],
                "severity": rule["severity"],
                "risk_weight": rule["risk_weight"],
                "description": rule["description"],
                "recommendation": rule["recommendation"]["message"],
                "missing_epi": []
            }

            # Analyse explicable des EPI
            if "epi_analysis" in rule:
                for epi_name, epi_cfg in rule["epi_analysis"].items():
                    epi_value = features.get(epi_cfg["feature"])
                    if epi_value is not None and epi_value < epi_cfg["threshold"]:
                        msg = f"{epi_name} probablement manquant"
                        violated_rule["missing_epi"].append(msg)
                        missing_epi_global.add(msg)

            violated_rules.append(violated_rule)
            total_risk_score += rule["risk_weight"]

        # Détermination du niveau de risque
        if total_risk_score >= 60:
            risk_level = "HIGH"
        elif total_risk_score >= 30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "violated_rules": violated_rules,
            "missing_epi": list(missing_epi_global),
            "risk_score": total_risk_score,
            "risk_level": risk_level
        }
