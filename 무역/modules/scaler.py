from sklearn.preprocessing import StandardScaler


def get_scaler(fit_data):
    scaler = StandardScaler()
    scaler.fit(fit_data)
    return scaler
