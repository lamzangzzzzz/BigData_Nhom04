# BigData_Nhom04 — Ứng dụng Apache Spark Phân tích Hành vi Người dùng Instagram

> Đồ án môn **Dữ liệu Lớn và Ứng dụng** — Nhóm 04  
> Khoa CNTT Kinh doanh, Trường Công nghệ và Thiết kế, **ĐH UEH**  
> GVHD: **TS. Võ Văn Hải**

---

## Thành viên nhóm

| Họ tên | Vai trò | Node |
|---|---|---|
| Lâm Quỳnh Giang | Trưởng nhóm | Master Node |
| Nguyễn Thanh Xinh | Thành viên | Worker 1 |
| Lê Đoan Thy | Thành viên | Worker 2 |

---

## Cấu trúc thư mục

```
├── src/
│   ├── Images/
│   ├── Nhom04_01_Data_Preprocessing.py
│   ├── Nhom04_02_SparkSQL_Analysis.py
│   ├── Nhom04_03_SparkML_Model_1.py
│   └── Nhom04_04_SparkML_Model_2.py
```

---

## Bộ dữ liệu

- **Tên:** `instagram_usage_lifestyle.csv`
- **Nguồn:** [Kaggle — Social Media User Analysis](https://www.kaggle.com/datasets/rockyt07/social-media-user-analysis)
- **Dung lượng:** ~419 MB
- **Số bản ghi:** 1,048,575 dòng
- **Số thuộc tính:** 57 cột

---

## Yêu cầu môi trường

| Thành phần | Phiên bản |
|---|---|
| Python | 3.x |
| Apache Spark | 4.1.1 |
| Apache Hadoop | 3.4.3 |
| PySpark | tương ứng Spark |

---

## Hướng dẫn chạy

### Bước 1: Chuẩn bị dữ liệu trên HDFS

Tải dataset từ Kaggle và upload lên HDFS:

```bash
hdfs dfs -mkdir -p /BigData_Nhom04
hdfs dfs -put instagram_usage_lifestyle.csv /BigData_Nhom04/
```

### Bước 2: Khởi động Hadoop + Spark Cluster

```bash
# ===== MASTER NODE =====

# 1. Khởi động HDFS trong Cmd
start-dfs.cmd

# 2. Khởi động YARN 
start-yarn.cmd

# 3. Khởi động Spark Master
%SPARK_HOME%\bin\spark-class.cmd org.apache.spark.deploy.master.Master --host <Master_IP>


# ===== WORKER 1 & WORKER 2 =====

# 1. Khởi động HDFS DataNode
hdfs.cmd datanode

# 2. Khởi động Spark Worker
spark-class.cmd org.apache.spark.deploy.worker.Worker spark://<Master_IP>:7077
```

### Bước 3: Chạy lần lượt theo thứ tự

```bash
# 1. Tiền xử lý dữ liệu (chạy trước tiên)
python Nhom04_01_Data_Preprocessing.py

# 2. Phân tích Spark SQL
python Nhom04_02_SparkSQL_Analysis.py

# 3. Bài toán KMeans Clustering
python Nhom04_03_SparkML_Model_1.py

# 4. Bài toán Hồi quy Stress
python Nhom04_04_SparkML_Model_2.py
```

> **Lưu ý:** Phải chạy file `01` trước vì các file sau đọc dữ liệu đã được làm sạch từ HDFS tại đường dẫn `hdfs://<Master_IP>:9000/BigData_Nhom04/instagram_cleaned`

---

## Nội dung chính

### Spark SQL — 10+ truy vấn nâng cao
- Phân nhóm người dùng theo mô hình RFM (CTE + Window Function)
- Xác định chân dung người dùng click quảng cáo cao (Subquery + RANK)
- Phân tích hành vi theo thế hệ (Gen Z / Millennial / Gen X / Baby Boomer)
- Và nhiều truy vấn nâng cao khác...

### Spark MLlib

**Bài toán 1 — KMeans Clustering (Unsupervised)**
- Phân cụm 4 nhóm người dùng dựa trên 15 đặc trưng hành vi
- Pipeline: VectorAssembler → StandardScaler → KMeans
- Đánh giá: Elbow Method + Silhouette Score = 0.63

**Bài toán 2 — Hồi quy dự đoán điểm Stress (Supervised)**
- Dự đoán `perceived_stress_score` (thang 0–40)
- 7 mô hình: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, LightGBM
- Kết quả tốt nhất: **LightGBM** với RMSE = 6.09, R² = 0.7359
