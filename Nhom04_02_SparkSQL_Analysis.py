from pyspark.sql import SparkSession
# 1. KHOI TAO SPARKSESSION
spark = SparkSession.builder \
    .appName("Nhom04_SQL_Cau20_AgeGroup_IGTime") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "100") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 2. DOC DU LIEU SACH TU HDFS
INPUT_HDFS_PATH = "hdfs://26.142.182.248:9000/BigData_Nhom04/instagram_cleaned"

print("\n[INIT] Dang doc du lieu sach tu HDFS...")
df = spark.read.csv(INPUT_HDFS_PATH, header=True, inferSchema=True)

df.cache()

total_rows = df.count()
print(f"[INIT] Tong ban ghi: {total_rows:,} | So cot: {len(df.columns)}")

# 3. TAO TEMPVIEW
df.createOrReplaceTempView("instagram_cleaned_view")
print("[INIT] Da tao TempView: 'instagram_cleaned_view' | Cache: ON\n")

print("  CAU 7")

spark.sql("""
    WITH health_groups AS (
        SELECT
            CASE
                WHEN exercise_hours_per_week = 0  THEN '1. Khong tap (0h)'
                WHEN exercise_hours_per_week < 3  THEN '2. It tap    (<3h)'
                WHEN exercise_hours_per_week < 6  THEN '3. Vua phai  (3-6h)'
                WHEN exercise_hours_per_week < 10 THEN '4. Tich cuc  (6-10h)'
                ELSE                                   '5. Cuong tap (>10h)'
            END AS exercise_level,
            CASE
                WHEN daily_active_minutes_instagram < 60  THEN '1. Rat it   (<1h)'
                WHEN daily_active_minutes_instagram < 180 THEN '2. Vua phai (1-3h)'
                WHEN daily_active_minutes_instagram < 300 THEN '3. Nhieu    (3-5h)'
                ELSE                                          '4. Nghien   (>5h)'
            END AS ig_usage_level,
            sleep_hours_per_night,
            perceived_stress_score,
            self_reported_happiness,
            body_mass_index,
            daily_steps_count
        FROM instagram_cleaned_view
    )
    SELECT
        exercise_level                                        AS muc_tap_the_duc,
        ig_usage_level                                        AS muc_dung_instagram,
        COUNT(*)                                              AS so_nguoi_dung,
        ROUND(AVG(sleep_hours_per_night), 2)                  AS gio_ngu_tb,
        ROUND(AVG(perceived_stress_score), 2)                 AS diem_stress_tb,
        ROUND(AVG(self_reported_happiness), 2)                AS diem_hanh_phuc_tb,
        ROUND(AVG(body_mass_index), 2)                        AS bmi_tb,
        ROUND(AVG(daily_steps_count), 0)                      AS so_buoc_chan_tb,
        RANK() OVER (
            PARTITION BY exercise_level
            ORDER BY AVG(perceived_stress_score) DESC
        )                                                     AS xep_hang_stress_trong_nhom
    FROM health_groups
    GROUP BY exercise_level, ig_usage_level
    HAVING COUNT(*) >= 100
    ORDER BY exercise_level, ig_usage_level
""").show(20, truncate=False)

print("[DONE] Cau 7 hoan thanh!\n")

print("  CAU 8")

spark.sql("""
    WITH stats AS (
        SELECT
            AVG(daily_active_minutes_instagram)        AS mean_mins,
            AVG(user_engagement_score)                 AS mean_eng,
            AVG(followers_count)                       AS mean_flw,
            AVG(ads_clicked_per_day)                   AS mean_clicks,
            STDDEV_POP(daily_active_minutes_instagram) AS sd_mins,
            STDDEV_POP(user_engagement_score)          AS sd_eng,
            STDDEV_POP(followers_count)                AS sd_flw,
            STDDEV_POP(ads_clicked_per_day)            AS sd_clicks
        FROM instagram_cleaned_view
    ),
    z_scores AS (
        SELECT
            u.user_id,
            u.age,
            u.gender,
            u.daily_active_minutes_instagram,
            u.user_engagement_score,
            u.followers_count,
            u.ads_clicked_per_day,
            ABS((u.daily_active_minutes_instagram - s.mean_mins)  / NULLIF(s.sd_mins,    0)) AS z_mins,
            ABS((u.user_engagement_score          - s.mean_eng)   / NULLIF(s.sd_eng,     0)) AS z_eng,
            ABS((u.followers_count                - s.mean_flw)   / NULLIF(s.sd_flw,     0)) AS z_flw,
            ABS((u.ads_clicked_per_day            - s.mean_clicks)/ NULLIF(s.sd_clicks,  0)) AS z_clicks
        FROM instagram_cleaned_view u
        CROSS JOIN stats s
    ),
    flagged AS (
        SELECT *,
            ROUND(z_mins + z_eng + z_flw + z_clicks, 4) AS total_anomaly_score,
            CASE
                WHEN z_mins > 2 AND z_eng > 2     THEN 'Nghi Bot - Hoat dong qua muc'
                WHEN z_flw  > 3                   THEN 'Nghi mua Followers ao'
                WHEN z_clicks > 2                 THEN 'Nghi gian lan quang cao'
                WHEN (z_mins + z_eng + z_flw) > 4 THEN 'Bat thuong nhieu chieu - Can theo doi'
                ELSE                                   'Nguoi dung binh thuong'
            END AS anomaly_flag
        FROM z_scores
    )
    SELECT
        anomaly_flag                                          AS nhom_phan_loai,
        COUNT(*)                                             AS so_nguoi_dung,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)  AS ty_le_phan_tram,
        ROUND(AVG(total_anomaly_score), 4)                   AS diem_bat_thuong_tb,
        ROUND(AVG(followers_count), 0)                       AS followers_tb,
        ROUND(AVG(ads_clicked_per_day), 2)                   AS so_click_ads_tb,
        ROUND(AVG(daily_active_minutes_instagram), 2)        AS phut_dung_ig_tb,
        ROUND(AVG(user_engagement_score), 4)                 AS diem_tuong_tac_tb
    FROM flagged
    GROUP BY anomaly_flag
    ORDER BY diem_bat_thuong_tb DESC
""").show(10, truncate=False)

print("[DONE] Cau 8 hoan thanh!\n")

print("  CAU 9")

spark.sql("""
   SELECT
       age_group,
       country,
       user_count,
       avg_engagement_score,
       avg_daily_mins,
       RANK() OVER (
           PARTITION BY age_group
           ORDER BY avg_engagement_score DESC
       ) AS ranking
   FROM (
       SELECT
           CASE
               WHEN age < 20                   THEN '1. <20'
               WHEN age BETWEEN 20 AND 30      THEN '2. 20-30'
               WHEN age BETWEEN 31 AND 40      THEN '3. 31-40'
               WHEN age BETWEEN 41 AND 50      THEN '4. 41-50'
               ELSE                                 '5. >50'
           END AS age_group,
           country,
           COUNT(*)                                        AS user_count,
           ROUND(AVG(user_engagement_score), 4)            AS avg_engagement_score,
           ROUND(AVG(daily_active_minutes_instagram), 1)   AS avg_daily_mins
       FROM instagram_cleaned_view
       GROUP BY
           CASE
               WHEN age < 20                   THEN '1. <20'
               WHEN age BETWEEN 20 AND 30      THEN '2. 20-30'
               WHEN age BETWEEN 31 AND 40      THEN '3. 31-40'
               WHEN age BETWEEN 41 AND 50      THEN '4. 41-50'
               ELSE                                 '5. >50'
           END,
           country
   ) t
   ORDER BY age_group, ranking
""").show(60, truncate=False)

print("[DONE] Cau 9 hoan thanh!\n")

print("  CAU 10")


print("[DONE] Cau 10 hoan thanh!\n")

spark.sql("""
    SELECT
        CASE
            WHEN age BETWEEN 14 AND 29 THEN '1. 14-29 (Gen Z)'
            WHEN age BETWEEN 30 AND 45 THEN '2. 30-45 (Millennial)'
            WHEN age BETWEEN 46 AND 61 THEN '3. 46-61 (Gen X)'
            ELSE                            '4. 62+   (Baby Boomer)'
        END AS age_group,


        COUNT(*)                                        AS user_count,
        ROUND(AVG(daily_active_minutes_instagram), 2)   AS avg_instagram_minutes,
        ROUND(AVG(sessions_per_day), 2)                 AS avg_sessions,
        ROUND(AVG(user_engagement_score), 4)            AS avg_engagement,
        ROUND(AVG(perceived_stress_score), 2)           AS avg_stress,
        ROUND(AVG(self_reported_happiness), 2)          AS avg_happiness,
        ROUND(AVG(posts_created_per_week), 2)           AS avg_posts,
        ROUND(AVG(followers_count), 0)                  AS avg_followers


    FROM instagram_cleaned_view


    GROUP BY
        CASE
            WHEN age BETWEEN 14 AND 29 THEN '1. 14-29 (Gen Z)'
            WHEN age BETWEEN 30 AND 45 THEN '2. 30-45 (Millennial)'
            WHEN age BETWEEN 46 AND 61 THEN '3. 46-61 (Gen X)'
            ELSE                            '4. 62+   (Baby Boomer)'
        END


    ORDER BY avg_instagram_minutes DESC
""").show(truncate=False)

spark.stop()