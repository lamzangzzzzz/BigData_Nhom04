import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, abs, mean
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkStandardScaler
from pyspark.ml.clustering import KMeans
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.decomposition import PCA


plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
warnings.filterwarnings('ignore')


if __name__ == '__main__':
   # =========================================================================
   # 1. KHỞI TẠO SPARKSESSION (CẤU HÌNH CLUSTER PHÂN TÁN)
   # =========================================================================R
   print("--- Bước 1: Đang khởi tạo Apache Spark Session ---")
   spark = SparkSession.builder \
       .appName("Nhom04_KMeans_Clustering") \
       .master("spark://26.142.182.248:7077") \
       .config("spark.executor.memory", "2g") \
       .config("spark.driver.memory", "2g") \
       .config("spark.executor.cores", "4") \
       .config("spark.cores.max", "16") \
       .config("spark.sql.shuffle.partitions", "50") \
       .config("spark.memory.fraction", "0.6") \
       .config("spark.executor.heartbeatInterval", "30s") \
       .config("spark.network.timeout", "300s") \
       .config("spark.driver.host", "26.39.47.113") \
       .config("spark.driver.bindAddress", "26.39.47.113") \
       .getOrCreate()


   spark.sparkContext.setLogLevel("WARN")
   print("Đã khởi tạo SparkSession phân tán thành công\n")


   # =========================================================================
   # 2. NẠP DỮ LIỆU TỪ HDFS
   # =========================================================================
   INPUT_HDFS_PATH = "hdfs://26.142.182.248:9000/BigData_Nhom04/instagram_cleaned"
   print(f"--- Bước 2: Đang đọc dữ liệu sạch từ HDFS: {INPUT_HDFS_PATH} ---")


   df = spark.read.csv(INPUT_HDFS_PATH, header=True, inferSchema=False)


   raw_numeric_cols = [
       'daily_active_minutes_instagram', 'sessions_per_day', 'average_session_length_minutes',
       'posts_created_per_week', 'reels_watched_per_day', 'stories_viewed_per_day',
       'likes_given_per_day', 'comments_written_per_day', 'dms_sent_per_week', 'dms_received_per_week',
       'self_reported_happiness', 'followers_count', 'following_count',
       'notification_response_rate', 'body_mass_index', 'blood_pressure_systolic',
       'blood_pressure_diastolic', 'daily_steps_count', 'exercise_hours_per_week',
       'sleep_hours_per_night', 'follower_following_ratio',
       'stress_score_normalized'
   ]


   cast_exprs = [col(c).cast(DoubleType()).alias(c) for c in raw_numeric_cols]
   other_cols = [col(c) for c in df.columns if c not in raw_numeric_cols]
   df = df.select(other_cols + cast_exprs)


   total_rows = df.count()
   print(f"Đã nạp dữ liệu thành công. Tổng số bản ghi: {total_rows:,} | Số cột: {len(df.columns)}")


   # =========================================================================
   # 3. XỬ LÝ MISSING VALUE
   # =========================================================================
   print("--- Bước 3: Điền giá trị khuyết ---")
   mean_dict = df.select([mean(c).alias(c) for c in raw_numeric_cols]).first().asDict()
   df = df.na.fill(mean_dict)
   print("Xử lý khuyết thiếu hoàn tất.")


   # =========================================================================
   # 4. FEATURE ENGINEERING
   # =========================================================================
   print("\n--- Bước 4: Thực hiện Feature Engineering ---")


   df = df.withColumn('creator_score', col('posts_created_per_week') * 1.5 + col('comments_written_per_day') * 0.5)
   df = df.withColumn('consumer_score', col('reels_watched_per_day') * 1.0 + col('stories_viewed_per_day') * 0.8)
   df = df.withColumn('passive_active_ratio', (col('consumer_score') + 1) / (col('creator_score') + 1))
   df = df.withColumn('social_butterfly_index', col('dms_sent_per_week') + col('comments_written_per_day') * 7)
   df = df.withColumn('dm_reciprocity', (col('dms_received_per_week') + 1) / (col('dms_sent_per_week') + 1))


   bmi_score = 100 - (abs(col('body_mass_index') - 22) * 3)
   bp_score = 100 - (abs(col('blood_pressure_systolic') - 120) * 0.5) - (
               abs(col('blood_pressure_diastolic') - 80) * 0.5)
   steps_score = (col('daily_steps_count') / 150) + (col('exercise_hours_per_week') * 5)
   df = df.withColumn('wellness_index',
                      (bmi_score + bp_score + steps_score + (col('sleep_hours_per_night') * 12.5)) / 4)


   df = df.withColumn('addiction_risk_score',
                      (col('daily_active_minutes_instagram') * 0.4) + (col('sessions_per_day') * 1.5) + (
                              col('notification_response_rate') * 10))
   df = df.withColumn('usage_intensity_score', col('daily_active_minutes_instagram') * col('sessions_per_day'))


   feature_list = [
       'daily_active_minutes_instagram', 'sessions_per_day', 'average_session_length_minutes',
       'creator_score', 'consumer_score', 'passive_active_ratio',
       'social_butterfly_index', 'dm_reciprocity',
       'wellness_index', 'stress_score_normalized', 'self_reported_happiness',
       'followers_count', 'follower_following_ratio',
       'addiction_risk_score', 'usage_intensity_score'
   ]


   df.cache()
   print(f"Hoàn thành trích xuất {len(feature_list)} đặc trưng thông minh.")


   # =========================================================================
   # 5. XÁC ĐỊNH K TỐI ƯU BẰNG ELBOW METHOD
   # =========================================================================
   print("\n--- Bước 5: Tìm K tối ưu bằng Elbow Method ---")
   inertia_scores = []
   k_range = range(2, 7)


   df_elbow_sample = df.sample(withReplacement=False, fraction=0.05, seed=42).cache()


   assembler = VectorAssembler(inputCols=feature_list, outputCol="raw_features", handleInvalid="skip")
   scaler = SparkStandardScaler(inputCol="raw_features", outputCol="scaled_features", withStd=True, withMean=True)


   for k in k_range:
       print(f"  Đang tính toán chi phí huấn luyện với K = {k}...")
       km_test = KMeans(featuresCol="scaled_features", k=k, seed=42, maxIter=10)
       pl_test = Pipeline(stages=[assembler, scaler, km_test])
       pl_model = pl_test.fit(df_elbow_sample)
       cost = pl_model.stages[-1].summary.trainingCost
       inertia_scores.append(cost)
       print(f"   => K={k} | Training Cost (Inertia): {cost:.2f}")


   df_elbow_sample.unpersist()
   print("Đã hoàn thành tính toán Elbow.")


   print("Đang kết xuất đồ thị Elbow")
   plt.figure(figsize=(8, 4))
   plt.plot(list(k_range), inertia_scores, 'o-', color='#E1306C', linewidth=2, markersize=7)
   plt.xlabel('Số lượng cụm K')
   plt.ylabel('Training Cost (Inertia)')
   plt.title('Biểu đồ Elbow Method: Xác định K tối ưu trên PySpark Cluster')
   plt.xticks(list(k_range))
   plt.grid(True, linestyle='--', alpha=0.5)
   plt.tight_layout()
   plt.savefig("elbow_method.png", dpi=150, bbox_inches="tight")
   plt.show()
   plt.close()
   print("Đã lưu đồ thị Elbow: 'elbow_method.png'")


   # =========================================================================
   # 6. HUẤN LUYỆN PIPELINE CHÍNH THỨC VỚI K CỐ ĐỊNH
   # =========================================================================
   NUM_CLUSTERS = 4
   print(f"\n--- Bước 6: Huấn luyện Pipeline chính thức với K = {NUM_CLUSTERS} ---")


   kmeans_main = KMeans(featuresCol="scaled_features", predictionCol="Cluster", k=NUM_CLUSTERS, seed=42, maxIter=20)
   pipeline_main = Pipeline(stages=[assembler, scaler, kmeans_main])
   pipeline_model = pipeline_main.fit(df)
   df_clustered = pipeline_model.transform(df)


   df_clustered.cache()
   df_clustered.count()
   print("Pipeline chính đã thực thi và cache dữ liệu phân cụm thành công.")


   print("\nPhân bổ người dùng theo từng cụm:")
   df_clustered.groupBy("Cluster").count().orderBy("Cluster").show()


   # =========================================================================
   # 7. TRỰC QUAN HÓA KẾT QUẢ PHÂN CỤM (PCA & Radar Chart)
   # =========================================================================
   print("\n--- Bước 7: Trực quan hóa kết quả phân cụm ---")


   fraction_sample = min(10000.0 / total_rows, 1.0)
   sample_pd = df_clustered.select(feature_list + ['Cluster']).sample(
       withReplacement=False, fraction=fraction_sample, seed=42
   ).toPandas()


   # Đồ thị 1: PCA 2D
   if not sample_pd.empty:
       print("  Vẽ biểu đồ PCA 2D")
       scaler_np = SklearnStandardScaler().fit_transform(sample_pd[feature_list])
       pca = PCA(n_components=2, random_state=42)
       pca_coords = pca.fit_transform(scaler_np)


       pca_df = pd.DataFrame(pca_coords, columns=['PC1', 'PC2'])
       pca_df['Cluster'] = sample_pd['Cluster'].astype(str)


       plt.figure(figsize=(10, 6))
       sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='Set1', alpha=0.5, edgecolors='none')
       plt.title('Cấu trúc phân cụm người dùng Instagram (PCA 2D)')
       plt.xlabel(f'PC1 — {pca.explained_variance_ratio_[0] * 100:.1f}% variance')
       plt.ylabel(f'PC2 — {pca.explained_variance_ratio_[1] * 100:.1f}% variance')
       plt.legend(title='Cụm')
       plt.tight_layout()
       plt.savefig("pca_clusters.png", dpi=150, bbox_inches="tight")
       plt.show()
       plt.close()
       print("  Đã lưu: 'pca_clusters.png'")


   # Đồ thị 2: Radar Chart
   if not sample_pd.empty:
       print("  Vẽ Radar Chart...")
       cluster_summary = sample_pd.groupby('Cluster')[feature_list].mean()
       normalized_summary = (cluster_summary - cluster_summary.min()) / (
               cluster_summary.max() - cluster_summary.min() + 1e-5)


       N = len(feature_list)
       angles = [n / float(N) * 2 * np.pi for n in range(N)]
       angles += angles[:1]


       fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
       colors_radar = ['#E1306C', '#405DE6', '#FCAF45', '#58C322']


       for idx, cluster_idx in enumerate(normalized_summary.index):
           values = normalized_summary.loc[cluster_idx].values.flatten().tolist()
           values += values[:1]
           ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cụm {cluster_idx}',
                   color=colors_radar[idx % len(colors_radar)])
           ax.fill(angles, values, alpha=0.05, color=colors_radar[idx % len(colors_radar)])


       plt.xticks(angles[:-1], feature_list, color='grey', size=8)
       ax.set_rlabel_position(0)
       plt.yticks([0.25, 0.50, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=7)
       plt.ylim(0, 1)
       plt.title('Radar Chart: Đặc trưng hành vi & Tâm lý giữa các phân cụm', size=13, y=1.1)
       plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
       plt.tight_layout()
       plt.savefig("radar_clusters.png", dpi=150, bbox_inches="tight")
       plt.show()
       plt.close()
       print("  Đã lưu: 'radar_clusters.png'")


   df.unpersist()
   df_clustered.unpersist()


   print("\nPIPELINE PHÂN CỤM K-MEANS HOÀN TẤT THÀNH CÔNG!")
   spark.stop()
   print("[INFO] SparkSession đã đóng an toàn.")
