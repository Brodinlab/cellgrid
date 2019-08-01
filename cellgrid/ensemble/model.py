import abc
from sklearn.metrics.scorer import check_scoring
from xgboost import XGBClassifier


class Model(abc.ABC):
    def __init__(self, name, markers, level=None, **kwargs):
        self._name = name
        self._markers = markers
        self._level = level
        self._model_ins = self.init_model_instance(**kwargs)

    @abc.abstractmethod
    def init_model_instance(self, **kwargs):
        pass

    def fit(self, x_train, y_train):
        self._model_ins.fit(x_train[self._markers], y_train)

    def score(self, x_train, y_train):
        return check_scoring(self._model_ins)(self._model_ins,
                                              x_train[self._markers],
                                              y_train)

    def predict(self, x):
        return self._model_ins.predict(x[self._markers])


class XgbModel(Model):
    def init_model_instance(self, **kwargs):
        params = {
            'n_jobs': 10,
            'max_depth': 10,
            'n_estimators': 40
        }
        params.update(kwargs)
        return XGBClassifier(**params)
