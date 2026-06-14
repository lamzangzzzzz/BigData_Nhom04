from pyspark.sql import SparkSession
# KHOI TAO SPARKSESSION
spark = SparkSession.builder \
    .appName("Nhom04_SQL_Cau20_AgeGroup_IGTime") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "100") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# DOC DU LIEU SACH TU HDFS
INPUT_HDFS_PATH = "hdfs://26.142.182.248:9000/BigData_Nhom04/instagram_cleaned"

print("\n[INIT] Dang doc du lieu sach tu HDFS...")
df = spark.read.csv(INPUT_HDFS_PATH, header=True, inferSchema=True)

df.cache()

total_rows = df.count()
print(f"[INIT] Tong ban ghi: {total_rows:,} | So cot: {len(df.columns)}")

# TAO TEMPVIEW
df.createOrReplaceTempView("instagram_cleaned_view")
print("[INIT] Da tao TempView: 'instagram_cleaned_view' | Cache: ON\n")

print("  CAU 1")


spark.sql("""
  SELECT
      CASE
          WHEN posts_created_per_week = 0 AND reels_watched_per_day > 10
              THEN '1. Pure Lurker   (Chi xem)'
          WHEN posts_created_per_week > 3 AND reels_watched_per_day < 5
              THEN '2. Active Creator (Cham post)'
          ELSE    '3. Casual User    (Binh thuong)'
      END                                       AS user_persona,
      gender,
      COUNT(*)                                  AS total_users,
      ROUND(AVG(followers_count), 0)            AS avg_followers,
      ROUND(AVG(following_count), 0)            AS avg_following,
      ROUND(AVG(likes_given_per_day), 1)        AS avg_likes_given,
      ROUND(AVG(reels_watched_per_day), 1)      AS avg_reels_watched,
      ROUND(AVG(posts_created_per_week), 2)     AS avg_posts_week,
      ROUND(AVG(self_reported_happiness), 2)    AS avg_happiness,
      ROUND(AVG(user_engagement_score), 4)      AS avg_engagement
  FROM instagram_cleaned_view
  GROUP BY
      CASE
          WHEN posts_created_per_week = 0 AND reels_watched_per_day > 10
              THEN '1. Pure Lurker   (Chi xem)'
          WHEN posts_created_per_week > 3 AND reels_watched_per_day < 5
              THEN '2. Active Creator (Cham post)'
          ELSE    '3. Casual User    (Binh thuong)'
      END,
      gender
  ORDER BY user_persona, total_users DESC
""").show(20, truncate=False)


print("  CAU 2")


spark.sql("""
  WITH session_groups AS (
      SELECT
          CASE
              WHEN average_session_length_minutes < 15
                  THEN '1. Short Sessions  (<15 min)'
              WHEN average_session_length_minutes BETWEEN 15 AND 45
                  THEN '2. Medium Sessions (15-45 min)'
              ELSE    '3. Doomscrolling   (>45 min)'
          END                                           AS session_habit,
          daily_steps_count,
          exercise_hours_per_week,
          blood_pressure_systolic,
          blood_pressure_diastolic,
          sleep_hours_per_night,
          perceived_stress_score,
          daily_active_minutes_instagram
      FROM instagram_cleaned_view
  ),
  aggregated AS (
      SELECT
          session_habit,
          COUNT(*)                                          AS user_count,
          ROUND(AVG(daily_steps_count), 0)                 AS avg_daily_steps,
          ROUND(AVG(exercise_hours_per_week), 2)           AS avg_exercise_hours,
          ROUND(AVG(blood_pressure_systolic), 1)           AS avg_bp_systolic,
          ROUND(AVG(blood_pressure_diastolic), 1)          AS avg_bp_diastolic,
          ROUND(AVG(sleep_hours_per_night), 2)             AS avg_sleep_hours,
          ROUND(AVG(perceived_stress_score), 2)            AS avg_stress_score,
          ROUND(AVG(daily_active_minutes_instagram), 1)    AS avg_ig_minutes_day
      FROM session_groups
      GROUP BY session_habit
  )
  SELECT
      session_habit,
      user_count,
      avg_daily_steps,
      RANK() OVER (ORDER BY avg_daily_steps DESC)          AS steps_rank,
      avg_exercise_hours,
      avg_bp_systolic,
      avg_bp_diastolic,
      avg_sleep_hours,
      avg_stress_score,
      avg_ig_minutes_day
  FROM aggregated
  ORDER BY session_habit
""").show(truncate=False)




print("  CAU 3")


spark.sql("""
  WITH privacy_stats AS (
      SELECT
          privacy_setting_level,
          gender,
          COUNT(*)                                                          AS total_users,
          ROUND(AVG(dms_sent_per_week), 2)                                 AS avg_dms_sent,
          ROUND(AVG(dms_received_per_week), 2)                             AS avg_dms_received,
          ROUND(AVG(comments_written_per_day), 2)                          AS avg_comments,
          ROUND(AVG(likes_given_per_day), 2)                               AS avg_likes,
          ROUND(AVG(stories_viewed_per_day), 1)                            AS avg_stories_viewed,
          ROUND(
              AVG(dms_sent_per_week) / NULLIF(AVG(comments_written_per_day), 0)
          , 2)                                                             AS dm_vs_comment_ratio,
          ROUND(AVG(followers_count), 0)                                   AS avg_followers,
          ROUND(AVG(user_engagement_score), 4)                             AS avg_engagement
      FROM instagram_cleaned_view
      GROUP BY privacy_setting_level, gender
  )
  SELECT
      privacy_setting_level,
      gender,
      total_users,
      avg_dms_sent,
      avg_dms_received,
      avg_comments,
      avg_likes,
      avg_stories_viewed,
      dm_vs_comment_ratio,
      RANK() OVER (
          PARTITION BY gender
          ORDER BY dm_vs_comment_ratio DESC
      )                                                                     AS privacy_rank_by_gender,
      avg_followers,
      avg_engagement
  FROM privacy_stats
  ORDER BY privacy_setting_level, gender
""").show(20, truncate=False)

print("  CAU 4")

spark.sql("""
   WITH rfm_base AS (
       SELECT
           CASE
               WHEN sessions_per_day >= 5 AND user_engagement_score >= 1.8
                   THEN '1. Champions       (Trung thanh + Nang no)'
               WHEN sessions_per_day >= 5 AND user_engagement_score < 1.8
                   THEN '2. Browsers        (Vao nhieu - It tuong tac)'
               WHEN sessions_per_day < 5  AND user_engagement_score >= 1.8
                   THEN '3. Quality Engagers(It vao - Chat luong cao)'
               ELSE    '4. At Risk          (Nguy co roi bo App)'
           END                                           AS user_segment,
           gender,
           daily_active_minutes_instagram,
           sessions_per_day,
           user_engagement_score,
           ads_clicked_per_day,
           ads_viewed_per_day,
           posts_created_per_week,
           followers_count,
           self_reported_happiness
       FROM instagram_cleaned_view
   ),
   rfm_agg AS (
       SELECT
           user_segment,
           COUNT(*)                                          AS total_users,
           ROUND(AVG(daily_active_minutes_instagram), 1)    AS avg_minutes_spent,
           ROUND(AVG(sessions_per_day), 2)                  AS avg_sessions,
           ROUND(AVG(user_engagement_score), 4)             AS avg_engagement,
           ROUND(AVG(ads_clicked_per_day), 2)               AS avg_ads_clicked,
           ROUND(
               AVG(ads_clicked_per_day) * 100.0
               / NULLIF(AVG(ads_viewed_per_day), 0)
           , 2)                                             AS ads_ctr_pct,
           ROUND(AVG(posts_created_per_week), 2)            AS avg_posts_week,
           ROUND(AVG(followers_count), 0)                   AS avg_followers,
           ROUND(AVG(self_reported_happiness), 2)           AS avg_happiness
       FROM rfm_base
       GROUP BY user_segment
   )
   SELECT
       user_segment,
       total_users,
       ROUND(total_users * 100.0 / SUM(total_users) OVER (), 2) AS pct_of_total,
       avg_minutes_spent,
       avg_sessions,
       avg_engagement,
       avg_ads_clicked,
       ads_ctr_pct,
       avg_posts_week,
       avg_followers,
       avg_happiness,
       RANK() OVER (ORDER BY avg_ads_clicked DESC)              AS ads_value_rank
   FROM rfm_agg
   ORDER BY user_segment
""").show(truncate=False)

print("  CAU 5")

spark.sql("""
   SELECT
       income_level,
       employment_status,
       preferred_content_theme,
       total_users,
       avg_ads_clicked,
       avg_time_online_mins,
       ROUND(avg_ads_clicked / NULLIF(avg_time_online_mins / 60.0, 0), 3) AS ads_click_per_hour,
       avg_engagement,
       avg_followers,
       RANK() OVER (ORDER BY
           avg_ads_clicked / NULLIF(avg_time_online_mins / 60.0, 0) DESC
       )                                                                   AS targeting_rank
   FROM (
       SELECT
           income_level,
           employment_status,
           preferred_content_theme,
           COUNT(*)                                          AS total_users,
           ROUND(AVG(ads_clicked_per_day), 3)               AS avg_ads_clicked,
           ROUND(AVG(daily_active_minutes_instagram), 2)    AS avg_time_online_mins,
           ROUND(AVG(user_engagement_score), 4)             AS avg_engagement,
           ROUND(AVG(followers_count), 0)                   AS avg_followers
       FROM instagram_cleaned_view
       GROUP BY income_level, employment_status, preferred_content_theme
       HAVING COUNT(*) > 500
   ) ads_subquery
   ORDER BY ads_click_per_hour DESC
   LIMIT 10
""").show(truncate=False)

print("  CAU 6")

spark.sql("""
   WITH freemium_base AS (
       SELECT
           subscription_status,
           CASE
               WHEN notification_response_rate > 0.7  THEN '1. High Response   (>70%)'
               WHEN notification_response_rate >= 0.3 THEN '2. Medium Response (30-70%)'
               ELSE                                        '3. Low Response    (<30%)'
           END                                           AS notif_responsiveness,
           linked_accounts_count,
           user_engagement_score,
           daily_active_minutes_instagram,
           sessions_per_day,
           uses_premium_features,
           notification_response_rate
       FROM instagram_cleaned_view
   ),
   freemium_agg AS (
       SELECT
           subscription_status,
           notif_responsiveness,
           COUNT(*)                                          AS user_count,
           ROUND(AVG(linked_accounts_count), 2)             AS avg_linked_accounts,
           ROUND(AVG(user_engagement_score), 4)             AS avg_engagement,
           ROUND(AVG(daily_active_minutes_instagram), 1)    AS avg_ig_minutes,
           ROUND(AVG(sessions_per_day), 2)                  AS avg_sessions,
           ROUND(AVG(notification_response_rate) * 100, 1)  AS avg_notif_rate_pct
       FROM freemium_base
       GROUP BY subscription_status, notif_responsiveness
   )
   SELECT
       subscription_status,
       notif_responsiveness,
       user_count,
       ROUND(user_count * 100.0
           / SUM(user_count) OVER (PARTITION BY subscription_status), 2) AS pct_in_plan,
       avg_linked_accounts,
       avg_engagement,
       avg_ig_minutes,
       avg_sessions,
       avg_notif_rate_pct,
       ROW_NUMBER() OVER (
           PARTITION BY subscription_status
           ORDER BY avg_engagement DESC
       )                                                                  AS engage_rank_in_plan
   FROM freemium_agg
   ORDER BY subscription_status, notif_responsiveness
""").show(20, truncate=False)

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