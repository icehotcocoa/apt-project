# 특성 메모리 사용량 감소 int 64 -> 적당한 형태로
def reduce_memory_usage(df):
    """
    데이터프레임의 메모리 사용량을 줄이기 위해 데이터 타입을 축소합니다.
    오버플로우를 방지하기 위해 각 열의 값 범위를 확인하여 적절한 데이터 타입으로 변환합니다.

    Parameters:
        df (pd.DataFrame): 데이터 타입 축소를 적용할 데이터프레임.

    Returns:
        pd.DataFrame: 데이터 타입이 축소된 데이터프레임.
    """
    import pandas as pd
    import numpy as np
    
    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_integer_dtype(col_type):
            # 정수형 처리
            col_min = df[col].min()
            col_max = df[col].max()

            if col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype("int16")
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype("int32")
            # 값이 더 클 경우, int64 유지

        elif pd.api.types.is_float_dtype(col_type):
            # 부동소수점 처리
            col_min = df[col].min()
            col_max = df[col].max()

            if col_max - col_min < 65504:
                df[col] = df[col].astype("float16")
            else:
                df[col] = df[col].astype("float32")
            # float64는 유지되지 않도록 float32로 축소

        elif pd.api.types.is_object_dtype(col_type):
            # 문자열 형식은 그대로 유지
            continue

    print(df.info())

    return df


# 모델 평가 함수 정의
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    모델 평가 함수
    
    전달인자 설명:
    model: 학습된 머신러닝 모델
    X_train: 학습 데이터 피처
    X_test: 테스트 데이터 피처
    y_train: 학습 데이터 타겟
    y_test: 테스트 데이터 타겟
    """
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import numpy as np

    # 테스트 데이터에 대한 예측 수행
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # 평가
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # 평균값 계산
    train_mean = y_train.mean()
    test_mean = y_test.mean()

    # MAE 대비 비율 계산
    train_mae_ratio = (train_mae / train_mean) * 100
    test_mae_ratio = (test_mae / test_mean) * 100

    # 평가 결과 반환
    return {
        'train_mse': round(train_mse, 2),
        'test_mse': round(test_mse, 2),
        'train_rmse': round(train_rmse, 2),
        'test_rmse': round(test_rmse, 2),
        'train_r2': round(train_r2, 2),
        'test_r2': round(test_r2, 2),
        'train_mae': round(train_mae, 2),
        'test_mae': round(test_mae, 2),
        'train_mae_ratio': round(train_mae_ratio, 2),
        'test_mae_ratio': round(test_mae_ratio, 2)
    }

