import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Khởi tạo các thư viện core của PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean
from pyspark.sql.functions import min as spark_min, max as spark_max
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml import Pipeline

# Các thuật toán Hồi quy
from pyspark.ml.regression import LinearRegression
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso

# Thư viện giải thích mô hình
import shap


# Cấu hình giao diện đồ thị
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
warnings.filterwarnings('ignore')

if __name__ == '__main__':

   # 1. KHỞI TẠO SPARKSESSION
   print("--- Bước 1: Đang khởi tạo Apache Spark Session kết nối vào cụm Cluster... ---")
   spark = SparkSession.builder \
       .appName("Nhom04_SparkML_StressPrediction") \
       .master("spark://26.142.182.248:7077") \
       .config("spark.executor.memory", "1g") \
       .config("spark.driver.memory", "2g") \
       .config("spark.executor.cores", "4") \
       .config("spark.sql.shuffle.partitions", "100") \
       .config("spark.driver.host", "26.211.116.140") \
       .config("spark.driver.bindAddress", "26.211.116.140") \
       .getOrCreate()


   spark.sparkContext.setLogLevel("WARN")
   print(" Đã kết nối Master SparkSession phân tán thành công!\n")

   # 2. NẠP DỮ LIỆU TỪ HDFS & ÉP KIỂU
   INPUT_HDFS_PATH = "hdfs://26.142.182.248:9000/BigData_Nhom04/instagram_cleaned"


   print(f"--- Bước 2: Đang nạp dữ liệu sạch từ HDFS: {INPUT_HDFS_PATH} ---")
   df = spark.read.csv(INPUT_HDFS_PATH, header=True, inferSchema=False)

   input_numeric_cols = [
       'daily_active_minutes_instagram', 'sessions_per_day', 'time_on_reels_per_day',
       'notification_response_rate', 'sleep_hours_per_night', 'exercise_hours_per_week',
       'daily_steps_count', 'age', 'body_mass_index', 'blood_pressure_systolic',
       'blood_pressure_diastolic', 'time_on_feed_per_day'
   ]
   input_categorical_cols = ['employment_status', 'urban_rural']
   target_col = 'perceived_stress_score'

   for c in (input_numeric_cols + [target_col]):
       df = df.withColumn(c, col(c).cast("double"))

   df.cache()
   total_rows = df.count()
   print(f" Đã nạp thành công! Tổng số bản ghi: {total_rows:,}  | Số cột:  {len(df.columns)}")

   # =========================================================================
   # 3. EDA & HEATMAP TƯƠNG QUAN (Trực quan hóa trên mẫu ban đầu)
   print("\n--- Bước 3: Đang thực hiện phân tích EDA và vẽ Heatmap tương quan... ---")
   sample_ratio = min(10000.0 / total_rows, 1.0)
   eda_sample_pd = df.select(input_numeric_cols + [target_col]) \
       .sample(False, sample_ratio, seed=42) \
       .toPandas()

   plt.figure(figsize=(8, 4))
   sns.histplot(eda_sample_pd[target_col], kde=True, color='#E1306C')
   plt.title('Biểu đồ phân phối của biến mục tiêu: Perceived Stress Score')
   plt.xlabel('Điểm số Stress (0 - 40)')
   plt.ylabel('Tần suất')
   plt.tight_layout()
   plt.savefig("Images/eda_stress_distribution.png", dpi=150, bbox_inches="tight")
   plt.show()

   plt.figure(figsize=(12, 10))
   corr_matrix = eda_sample_pd.corr()
   sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
   plt.title('Ma trận tương quan Heatmap hệ số Pearson')
   plt.tight_layout()
   plt.savefig("Images/eda_heatmap.png", dpi=150, bbox_inches="tight")
   plt.show()
   print(" Đã lưu biểu đồ EDA thành công.")

   # =========================================================================
   # 4. CHIA DỮ LIỆU TRAIN/TEST
   # =========================================================================
   # 4. CHIA DỮ LIỆU TRAIN/TEST (Tránh Data Leakage)
   print("\n--- Bước 4: Đang phân tách dữ liệu Train/Test... ---")
   train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)
   print(f" Đã phân tách: Train (70%) = {train_data.count():,} | Test (30%) = {test_data.count():,} dòng.")

   # 5. XỬ LÝ KHUYẾT THIẾU & FEATURE ENGINEERING (Dựa trên Train Data)
   print("\n--- Bước 5: Xử lý khuyết thiếu và Kỹ nghệ đặc trưng... ---")
   # 5.1. Imputation (Chỉ lấy Mean từ TẬP TRAIN)
   mean_dict = train_data.select([mean(c).alias(c) for c in input_numeric_cols + [target_col]]).first().asDict()
   train_data = train_data.na.fill(mean_dict)
   test_data = test_data.na.fill(mean_dict)

   # 5.2. Tính min/max cho chuẩn hóa nội bộ (Chỉ lấy từ TẬP TRAIN)
   bmi_stats = train_data.select(spark_min('body_mass_index').alias('min'),
                                 spark_max('body_mass_index').alias('max')).first()
   bp_stats = train_data.select(spark_min('blood_pressure_systolic').alias('min'),
                                spark_max('blood_pressure_systolic').alias('max')).first()


   def apply_feature_engineering(dataframe):
       df_fe = dataframe.withColumn('screen_to_sleep_ratio',
                                    col('daily_active_minutes_instagram') / col('sleep_hours_per_night'))
       df_fe = df_fe.withColumn('passive_consumption', col('time_on_feed_per_day') + col('time_on_reels_per_day'))


       bmi_scaled = ((col('body_mass_index') - bmi_stats['min']) / (bmi_stats['max'] - bmi_stats['min'] + 1e-5)) * 100
       bp_scaled = ((col('blood_pressure_systolic') - bp_stats['min']) / (
                   bp_stats['max'] - bp_stats['min'] + 1e-5)) * 100
       df_fe = df_fe.withColumn('health_composite', (bmi_scaled + bp_scaled) / 2)
       return df_fe

   train_data = apply_feature_engineering(train_data)
   test_data = apply_feature_engineering(test_data)

   engineered_numeric_cols = [
       'daily_active_minutes_instagram', 'sessions_per_day', 'time_on_reels_per_day',
       'notification_response_rate', 'sleep_hours_per_night', 'exercise_hours_per_week',
       'daily_steps_count', 'age', 'screen_to_sleep_ratio', 'passive_consumption', 'health_composite'
   ]
   print(f" Đã tạo thành công các biến FE và Imputation an toàn.")

   # =========================================================================
   # 6. PIPELINE MÃ HÓA BIẾN CHỮ & CHUẨN HÓA DỮ LIỆU TOÀN DIỆN
   print("\n--- Bước 6: Thiết lập Pipeline mã hóa và Standard Scaler... ---")
   stages = []
   encoded_categorical_cols = []


   for col_name in input_categorical_cols:
       indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_index", handleInvalid="skip")
       encoder = OneHotEncoder(inputCol=col_name + "_index", outputCol=col_name + "_vec")
       stages += [indexer, encoder]
       encoded_categorical_cols.append(col_name + "_vec")

   assembler_inputs = engineered_numeric_cols + encoded_categorical_cols
   assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="raw_features")
   stages.append(assembler)

   scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)
   stages.append(scaler)

   pipeline = Pipeline(stages=stages)
   pipeline_model = pipeline.fit(train_data)

   train_transformed = pipeline_model.transform(train_data)
   test_transformed = pipeline_model.transform(test_data)
   print(" Hoàn thành trích xuất và chuẩn hóa đặc trưng thông minh.")

   # =========================================================================
   # 7. ĐÀO TẠO 7 THUẬT TOÁN HỒI QUY
   print("\n --- Bước 7: Tiến hành huấn luyện 7 thuật toán Hồi quy riêng lẻ... ---")

   # Thuật toán 1: Linear Regression (Spark phân tán)
   print("    [1/7] Đang huấn luyện: Linear Regression (Spark)...")
   lr = LinearRegression(featuresCol="features", labelCol=target_col)
   lr_model = lr.fit(train_transformed)
   lr_preds = lr_model.transform(test_transformed)

   # Lấy mẫu về Pandas cho 6 thuật toán còn lại CÙNG LÚC với kết quả dự đoán của Spark
   print("    Đang chiết xuất mẫu dữ liệu an toàn cho các mô hình đơn lẻ...")
   sample_fraction = min(50000.0 / total_rows, 0.2)

   train_pd = train_transformed.select(engineered_numeric_cols + [target_col]) \
       .sample(False, sample_fraction, seed=42).toPandas()

   test_pd = lr_preds.select(engineered_numeric_cols + [target_col, "prediction"]) \
       .sample(False, sample_fraction, seed=42).toPandas()

   X_train = train_pd[engineered_numeric_cols]
   y_train = train_pd[target_col]
   X_test = test_pd[engineered_numeric_cols]
   y_test = test_pd[target_col]
   lr_preds_pd = test_pd["prediction"]


   # Thuật toán 2: Ridge
   print("    [2/7] Đang huấn luyện: Ridge Regression...")
   ridge_reg = Ridge(alpha=1.0)
   ridge_reg.fit(X_train, y_train)
   ridge_preds_np = ridge_reg.predict(X_test)


   # Thuật toán 3: Lasso
   print("    [3/7] Đang huấn luyện: Lasso Regression...")
   lasso_reg = Lasso(alpha=0.1, max_iter=2000)
   lasso_reg.fit(X_train, y_train)
   lasso_preds_np = lasso_reg.predict(X_test)


   # Thuật toán 4: Random Forest
   print("    [4/7] Đang huấn luyện: Random Forest Regressor...")
   rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
   rf_reg.fit(X_train, y_train)
   rf_preds_np = rf_reg.predict(X_test)


   # Thuật toán 5: Gradient Boosting
   print("    [5/7] Đang huấn luyện: Gradient Boosting Regressor...")
   gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
   gb_reg.fit(X_train, y_train)
   gb_preds_np = gb_reg.predict(X_test)


   # Thuật toán 6: XGBoost
   print("    [6/7] Đang huấn luyện: XGBoost Regressor...")
   xgb_reg = xgb.XGBRegressor(
       n_estimators=100, max_depth=6, learning_rate=0.1,
       objective='reg:squarederror', tree_method='hist',
       device='cpu', n_jobs=-1, random_state=42, verbosity=0
   )
   xgb_reg.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
   xgb_preds_np = xgb_reg.predict(X_test)


   # Thuật toán 7: LightGBM
   print("    [7/7] Đang huấn luyện: LightGBM Regressor...")
   lgb_reg = LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
   lgb_reg.fit(X_train, y_train)
   lgb_preds_np = lgb_reg.predict(X_test)

   # 8. ĐÁNH GIÁ 7 MÔ HÌNH (RMSE, MAE, MAPE, R²)
   print("\n--- Bước 8: Đánh giá hiệu suất 7 mô hình (RMSE, MAE, MAPE, R²)... ---")
   from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


   def mape(y_true, y_pred):
       y_true, y_pred = np.array(y_true), np.array(y_pred)
       mask = y_true != 0
       return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


   def calc_metrics(y_true, y_pred):
       rmse = np.sqrt(mean_squared_error(y_true, y_pred))
       mae = mean_absolute_error(y_true, y_pred)
       mape_val = mape(y_true, y_pred)
       r2 = r2_score(y_true, y_pred)
       return rmse, mae, mape_val, r2


   lr_rmse, lr_mae, lr_mape, lr_r2 = calc_metrics(y_test, lr_preds_pd)
   ridge_rmse, ridge_mae, ridge_mape, ridge_r2 = calc_metrics(y_test, ridge_preds_np)
   lasso_rmse, lasso_mae, lasso_mape, lasso_r2 = calc_metrics(y_test, lasso_preds_np)
   rf_rmse, rf_mae, rf_mape, rf_r2 = calc_metrics(y_test, rf_preds_np)
   gb_rmse, gb_mae, gb_mape, gb_r2 = calc_metrics(y_test, gb_preds_np)
   xgb_rmse, xgb_mae, xgb_mape, xgb_r2 = calc_metrics(y_test, xgb_preds_np)
   lgb_rmse, lgb_mae, lgb_mape, lgb_r2 = calc_metrics(y_test, lgb_preds_np)


   results_df = pd.DataFrame({
       'Thuật toán': ['Linear Regression', 'Ridge Regression', 'Lasso Regression',
                      'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM'],
       'RMSE': [lr_rmse, ridge_rmse, lasso_rmse, rf_rmse, gb_rmse, xgb_rmse, lgb_rmse],
       'MAE': [lr_mae, ridge_mae, lasso_mae, rf_mae, gb_mae, xgb_mae, lgb_mae],
       'MAPE (%)': [lr_mape, ridge_mape, lasso_mape, rf_mape, gb_mape, xgb_mape, lgb_mape],
       'R²': [lr_r2, ridge_r2, lasso_r2, rf_r2, gb_r2, xgb_r2, lgb_r2],
   }).sort_values('RMSE').reset_index(drop=True)
   results_df.index += 1


   print("\n Bảng xếp hạng hiệu suất 7 mô hình (sắp xếp theo RMSE tăng dần):")
   print(results_df.to_string())


   # Biểu đồ so sánh RMSE và R²
   fig, axes = plt.subplots(1, 2, figsize=(14, 5))
   colors = ['#2ecc71' if i == 0 else '#3498db' if i < 3 else '#e74c3c' for i in range(len(results_df))]


   axes[0].barh(results_df['Thuật toán'], results_df['RMSE'], color=colors)
   axes[0].set_title('So sánh RMSE (nhỏ hơn = tốt hơn)', fontsize=13)
   axes[0].set_xlabel('RMSE')
   axes[0].invert_yaxis()
   for i, v in enumerate(results_df['RMSE']):
       axes[0].text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10)


   axes[1].barh(results_df['Thuật toán'], results_df['R²'], color=colors)
   axes[1].set_title('So sánh R² (lớn hơn = tốt hơn)', fontsize=13)
   axes[1].set_xlabel('R²')
   axes[1].invert_yaxis()
   for i, v in enumerate(results_df['R²']):
       axes[1].text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=10)


   plt.suptitle('Bảng so sánh hiệu suất 7 mô hình Hồi quy dự đoán Stress Score (Tỷ lệ 7:3)',
                fontsize=14, fontweight='bold')
   plt.tight_layout()
   plt.savefig("Images/model_comparison.png", dpi=150, bbox_inches="tight")
   plt.show()
   print(" Đã lưu biểu đồ so sánh mô hình.")


   # 9. GIẢI THÍCH MÔ HÌNH BẰNG SHAP (LightGBM)
   print("\n--- Bước 9: Đang tính toán giá trị SHAP (dùng LightGBM)... ---")
   n_shap = 500 if len(X_test) >= 500 else len(X_test)
   shap_sample = X_test.sample(n=n_shap, random_state=42)


   explainer = shap.TreeExplainer(lgb_reg)
   shap_values = explainer.shap_values(shap_sample)


   plt.figure(figsize=(10, 6))
   plt.title("SHAP: Mức độ đóng góp của các Feature đến điểm Stress")
   shap.summary_plot(shap_values, shap_sample, feature_names=engineered_numeric_cols,
                     plot_type="bar", show=False)
   plt.tight_layout()
   plt.savefig("Images/shap_bar.png", dpi=150, bbox_inches="tight")
   plt.show()


   plt.figure(figsize=(11, 7))
   plt.title("SHAP Beeswarm: Chiều hướng tác động của các tính năng lên điểm Stress")
   shap.summary_plot(shap_values, shap_sample, feature_names=engineered_numeric_cols, show=False)
   plt.tight_layout()
   plt.savefig("Images/shap_beeswarm.png", dpi=150, bbox_inches="tight")
   plt.show()
   print(" Đã lưu biểu đồ SHAP.")


   print("\n TOÀN BỘ PIPELINE HỒI QUY DỰ ĐOÁN STRESS ĐÃ CHẠY HOÀN TẤT THÀNH CÔNG!")
   spark.stop()
   print("[INFO] SparkSession đã đóng.")


