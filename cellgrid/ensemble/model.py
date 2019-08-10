from xgboost import XGBClassifier
from ..core import Model


class XgbModel(Model):
    def init_model_instance(self, **kwargs):
        params = {
            'n_jobs': 10,
            'max_depth': 10,
            'n_estimators': 40
        }
        params.update(kwargs)
        return XGBClassifier(**params)
