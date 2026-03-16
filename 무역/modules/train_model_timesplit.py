from modules.create_features import create_X_y
from modules.time_split import create_time_series_datasets
from sklearn.ensemble import RandomForestRegressor


class TrainModelWithTimeSplit:
    def __init__(self, features, target, pairs, scaler):
        self.features = features
        self.target = target
        self.pairs = pairs
        self.scaler = scaler
        self.time_datasets = create_time_series_datasets(features)

    def train_model_timesplit(self, model):
        """
        TimeSeriesSplit 데이터를 활용한 교차검증 예시
        마지막 fold의 스케일러를 저장하여 예측에 사용
        """
        models = []

        for i, dataset in enumerate(self.time_datasets):
            train_set = dataset["train_set"]

            X_train_features, y_train = create_X_y(train_set, self.pairs)
            X_train_features = self.scaler.transform(X_train_features)

            model = RandomForestRegressor(
                n_estimators=400, max_depth=9, random_state=10, criterion="friedman_mse"
            )
            model.fit(X_train_features, y_train)
            models.append(model)

        return models
