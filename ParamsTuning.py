import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=ConvergenceWarning)

if __name__ == '__main__':
    df = pd.read_csv("../Resources/BostonHousing.csv")
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Huấn luyện model với dữ liệu đã chuẩn hóa
    model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=20000, random_state=42)
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    score = model.score(x_test_scaled, y_test)
    print(f"R2=:{score}")

    #Định nghĩa lưới tham số
    param_grid = {
        'hidden_layer_sizes': [(10,), (50,), (100,), (50, 30)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'lbfgs'],
        'learning_rate_init': [0.001, 0.01],
        'alpha': [0.0001, 0.001]
    }
    #Tìm tham số tốt nhất
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(x_train_scaled, y_train)
    print("Best parameters found:", grid_search.best_params_)
    print("Best R2 on training set:", grid_search.best_score_)
    #Đánh giá trên tập test
    best_model = grid_search.best_estimator_
    tuning_score = best_model.score(x_test_scaled, y_test)
    print("Param tuning R2=", tuning_score)



