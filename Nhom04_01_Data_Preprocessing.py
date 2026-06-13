from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, DateType
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

# KHỞI TẠO MÔI TRƯỜNG
spark = SparkSession.builder \
   .appName("Nhom04_Data_Preprocessing") \
   .config("spark.executor.memory", "4g") \
   .config("spark.driver.memory", "2g") \
   .config("spark.sql.shuffle.partitions", "100") \
   .getOrCreate()

INPUT_HDFS_PATH  = "hdfs://26.142.182.248:9000/BigData_Nhom04/instagram_usage_lifestyle.csv"
OUTPUT_HDFS_PATH = "hdfs://26.142.182.248:9000/BigData_Nhom04/instagram_cleaned"

print("\n--- BẮT ĐẦU QUY TRÌNH LÀM SẠCH DỮ LIỆU ---")
df_raw = spark.read.csv(INPUT_HDFS_PATH, header=True, inferSchema=True)
total_raw = df_raw.count()
print(f"Số lượng dòng thô ban đầu: {total_raw}")
print(f"Shape (raw): {total_raw:,} rows x {len(df_raw.columns)} cols")

# BƯỚC 1: KIỂM TRA TRÙNG LẶP & MISSING VALUES
print("\n[Bước 1] Loại bỏ trùng lặp theo user_id...")
df_cleaned = df_raw.dropDuplicates(["user_id"])

print("[Bước 1] Xử lý missing values ở các cột cốt lõi...")
df_cleaned = df_cleaned.dropna(subset=["user_id", "age", "gender"])
print(f"Sau bước 1: {df_cleaned.count()} dòng")

# BƯỚC 2: ÉP KIỂU DỮ LIỆU (TYPE CASTING)
print("\n[Bước 2] Ép kiểu dữ liệu...")
cols_to_double = [
   "exercise_hours_per_week",
   "sleep_hours_per_night",
   "weekly_work_hours",
   "volunteer_hours_per_month",
   "user_engagement_score",
   "body_mass_index",
   "notification_response_rate",]
cols_to_int = [
   "age", "perceived_stress_score", "self_reported_happiness",
   "blood_pressure_systolic", "blood_pressure_diastolic",
   "daily_steps_count", "hobbies_count", "social_events_per_month",
   "books_read_per_year", "travel_frequency_per_year",
   "daily_active_minutes_instagram", "sessions_per_day",
   "posts_created_per_week", "reels_watched_per_day", "stories_viewed_per_day",
   "likes_given_per_day", "comments_written_per_day", "dms_sent_per_week",
   "dms_received_per_week", "ads_viewed_per_day", "ads_clicked_per_day",
   "time_on_feed_per_day", "time_on_explore_per_day",
   "time_on_messages_per_day", "time_on_reels_per_day",
   "followers_count", "following_count",
   "account_creation_year", "linked_accounts_count",
   "average_session_length_minutes",]

for col_name in cols_to_double:
   df_cleaned = df_cleaned.withColumn(col_name, F.col(col_name).cast(DoubleType()))

for col_name in cols_to_int:
   df_cleaned = df_cleaned.withColumn(col_name, F.col(col_name).cast(IntegerType()))

df_cleaned = df_cleaned.withColumn(
   "last_login_date",
   F.to_date(F.col("last_login_date"), "M/d/yyyy")
)
print(" Đã ép kiểu toàn bộ cột số và parse last_login_date → DateType")

df_cleaned = df_cleaned.withColumn(
   "education_level",
   F.regexp_replace(F.col("education_level"), "\u2019", "'")
)
df_cleaned = df_cleaned.withColumn(
   "education_level",
   F.regexp_replace(F.col("education_level"), "â€™", "'")
)
print("  Đã fix encoding smart quote trong education_level")

df_cleaned = df_cleaned.withColumn(
   "account_creation_year",
   F.floor(F.col("account_creation_year")).cast(IntegerType())
)
print(" Đã fix lỗi ở cột account_creation_year")

# BƯỚC 3: LỌC NHIỄU & XỬ LÝ OUTLIER HỢP LÝ
print("\n[Bước 3] Lọc outlier theo ngưỡng thực tế...")

df_cleaned = df_cleaned.filter(
   (F.col("age") >= 13) & (F.col("age") <= 65) &
   (F.col("sleep_hours_per_night") >= 3.0) & (F.col("sleep_hours_per_night") <= 10.0) &
   (F.col("exercise_hours_per_week") >= 0.0) & (F.col("exercise_hours_per_week") <= 24.0) &
   (F.col("weekly_work_hours") >= 0.0) & (F.col("weekly_work_hours") <= 80.0) &
   (F.col("body_mass_index") >= 15.0) & (F.col("body_mass_index") <= 45.0) &
   (F.col("average_session_length_minutes") >= 5) & (F.col("average_session_length_minutes") <= 52) &
   (F.col("blood_pressure_systolic") >= 70) & (F.col("blood_pressure_systolic") <= 200) &
   (F.col("account_creation_year") >= 2010) & (F.col("account_creation_year") <= 2025)
)
df_cleaned = df_cleaned.filter(
   F.col("gender") != "Prefer not to say"
)

total_after_outlier = df_cleaned.count()
print(f"Sau lọc outlier: {total_after_outlier} dòng")

# BƯỚC 4: ENCODE CATEGORICAL COLUMNS
print("\n[Bước 4] Encode categorical columns...")

binary_map = {
   "gender":                 {"Male": 0.0, "Female": 1.0, "Non-binary": 2.0, "Prefer not to say": 3.0},
   "has_children":           {"No": 0.0, "Yes": 1.0},
   "uses_premium_features":  {"No": 0.0, "Yes": 1.0},
   "two_factor_auth_enabled":{"No": 0.0, "Yes": 1.0},
   "biometric_login_used":   {"No": 0.0, "Yes": 1.0},
   "smoking":                {"No": 0.0, "Former": 1.0, "Yes": 2.0},
}
for col_name, mapping in binary_map.items():
   df_cleaned = df_cleaned.withColumn(
       col_name + "_encoded",
       F.when(F.col(col_name) == list(mapping.keys())[0], list(mapping.values())[0])
   )
   for k, v in list(mapping.items())[1:]:
       df_cleaned = df_cleaned.withColumn(
           col_name + "_encoded",
           F.when(F.col(col_name) == k, v).otherwise(F.col(col_name + "_encoded"))
       )

ordinal_maps = {
   "income_level":    {"Low": 1.0, "Lower-middle": 2.0, "Middle": 3.0, "Upper-middle": 4.0, "High": 5.0},
   "diet_quality":    {"Very poor": 1.0, "Poor": 2.0, "Average": 3.0, "Good": 4.0, "Excellent": 5.0},
   "alcohol_frequency": {"Never": 0.0, "Rarely": 1.0, "Weekly": 2.0, "Several times a week": 3.0, "Daily": 4.0},
   "privacy_setting_level": {"Public": 0.0, "Friends only": 1.0, "Private": 2.0},
   "education_level": {"High school": 1.0, "Some college": 2.0, "Bachelor's": 3.0, "Master's": 4.0, "PhD": 5.0, "Other": 0.0},
}
for col_name, mapping in ordinal_maps.items():
   mapping_expr = F.lit(None).cast(DoubleType())
   for k, v in mapping.items():
       mapping_expr = F.when(F.col(col_name) == k, v).otherwise(mapping_expr)
   df_cleaned = df_cleaned.withColumn(col_name + "_encoded", mapping_expr)

nominal_cols = [
   "urban_rural", "employment_status", "relationship_status",
   "content_type_preference", "preferred_content_theme",
   "subscription_status", "country"
]
indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in nominal_cols]
pipeline = Pipeline(stages=indexers)
df_cleaned = pipeline.fit(df_cleaned).transform(df_cleaned)
print(" Encode xong tất cả categorical columns")

# BƯỚC 5: FEATURE ENGINEERING
print("\n[Bước 5] Feature engineering...")

df_cleaned = df_cleaned.withColumn(
   "total_time_on_instagram_per_day",
   F.col("time_on_feed_per_day") + F.col("time_on_explore_per_day") +
   F.col("time_on_messages_per_day") + F.col("time_on_reels_per_day")
)
df_cleaned = df_cleaned.withColumn(
   "ads_ctr",
   F.when(F.col("ads_viewed_per_day") > 0,
          F.col("ads_clicked_per_day") / F.col("ads_viewed_per_day")
   ).otherwise(0.0)
)
df_cleaned = df_cleaned.withColumn(
   "follower_following_ratio",
   F.when(F.col("following_count") > 0,
          F.col("followers_count") / F.col("following_count")
   ).otherwise(F.col("followers_count").cast(DoubleType()))
)
df_cleaned = df_cleaned.withColumn(
   "account_age_years",
   F.lit(2025) - F.col("account_creation_year")
)
df_cleaned = df_cleaned.withColumn(
   "stress_score_normalized",
   F.col("perceived_stress_score") / F.lit(40.0)
)
print(" Đã tạo 5 features mới: total_time_on_instagram_per_day, ads_ctr,")
print("    follower_following_ratio, account_age_years, stress_score_normalized")

# TẠO BẢNG TẠM VÀ KIỂM TRA
print("\n--- TẠO BẢNG TẠM (TEMP VIEW) CHO SPARK SQL ---")
df_cleaned.createOrReplaceTempView("instagram_cleaned_view")

df_cleaned.cache()
print(" DataFrame đã được cache vào bộ nhớ cluster")

total_clean = df_cleaned.count()
print(f"\n Tổng kết:")
print(f"   Dòng thô ban đầu : {total_raw:,}")
print(f"   Dòng sạch cuối   : {total_clean:,}")
print(f"   Đã loại          : {total_raw - total_clean:,} dòng")
print(f"   Số features      : {len(df_cleaned.columns)}")
print(f"   Shape (clean)    : {total_clean:,} rows x {len(df_cleaned.columns)} cols")

spark.sql("""
   SELECT gender, COUNT(*) as total_users,
          ROUND(AVG(user_engagement_score), 4) as avg_engagement
   FROM instagram_cleaned_view
   GROUP BY gender
   ORDER BY total_users DESC
""").show()

# GHI KẾT QUẢ LÊN HDFS
print("\n--- ĐANG GHI DỮ LIỆU SẠCH LÊN HDFS ---")
df_cleaned.write.csv(OUTPUT_HDFS_PATH, header=True, mode="overwrite")
print(f" HOÀN THÀNH! Dữ liệu sạch lưu tại: {OUTPUT_HDFS_PATH}")