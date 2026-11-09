# als_model_100.py
# ALS — min_user=100 Kombinasyonu
# Kullanim: spark-submit --master spark://spark-master:7077 /opt/spark-apps/als_model_100.py

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import numpy as np
from collections import defaultdict
import time

# ── SPARK SESSION ────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("ALS_minuser100") \
    .config("spark.executor.memory", "3g") \
    .config("spark.driver.memory", "3g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("✅ Spark başlatıldı")

# ── VERİ YÜKLE (HDFS'ten) ───────────────────────────────────
print("Veri yükleniyor...")
ratings = spark.read.csv(
    "hdfs://namenode:9000/user/data/rating.csv",
    header=True, inferSchema=True
)
print(f"Toplam rating: {ratings.count():,}")

# ── FİLTRELE: min_movie=50, min_user=100 ────────────────────
print("Filtreler uygulanıyor...")

movie_counts = ratings.groupBy("movieId").count().filter(F.col("count") >= 50)
ratings = ratings.join(movie_counts.select("movieId"), on="movieId")

user_counts = ratings.groupBy("userId").count().filter(F.col("count") >= 100)
ratings = ratings.join(user_counts.select("userId"), on="userId")

total    = ratings.count()
n_users  = ratings.select("userId").distinct().count()
n_movies = ratings.select("movieId").distinct().count()
print(f"Filtrelenmiş rating   : {total:,}")
print(f"Filtrelenmiş kullanıcı: {n_users:,}")
print(f"Filtrelenmiş film     : {n_movies:,}")

# ── STRATİFİED SPLIT ────────────────────────────────────────
print("Stratified split yapılıyor...")

window  = Window.partitionBy("userId").orderBy(F.rand(seed=42))
ratings = ratings.withColumn("row_num", F.row_number().over(window))
ratings = ratings.withColumn(
    "total_per_user",
    F.count("*").over(Window.partitionBy("userId"))
)
ratings = ratings.withColumn(
    "n_test",
    F.greatest(F.lit(1), (F.col("total_per_user") * 0.2).cast("int"))
)

train_df = ratings.filter(F.col("row_num") > F.col("n_test"))
test_df  = ratings.filter(F.col("row_num") <= F.col("n_test"))

print(f"Train: {train_df.count():,}  Test: {test_df.count():,}")

# ── ALS MODELİ EĞİT ─────────────────────────────────────────
print("\n" + "="*55)
print("ALS — rank=50, maxIter=10, regParam=0.1, min_user=100")
print("="*55)

start = time.time()

als = ALS(
    rank              = 50,
    maxIter           = 10,
    regParam          = 0.1,
    userCol           = "userId",
    itemCol           = "movieId",
    ratingCol         = "rating",
    coldStartStrategy = "drop",
    seed              = 42
)

model = als.fit(train_df)
sure  = time.time() - start
print(f"Eğitim süresi: {sure:.1f} saniye")

# ── TAHMİN VE METRİKLER ─────────────────────────────────────
print("Tahminler hesaplanıyor...")

predictions = model.transform(test_df).dropna(subset=["prediction"])
preds_pd    = predictions.select(
    "userId", "movieId", "rating", "prediction"
).toPandas()

print(f"Toplam prediction: {len(preds_pd):,}")

def compute_metrics(df, threshold=3.5, n=10):
    top_n = (
        df.sort_values("prediction", ascending=False)
          .groupby("userId")
          .head(n)
    )

    user_relevant = defaultdict(set)
    for _, row in df.iterrows():
        if row["rating"] >= threshold:
            user_relevant[row["userId"]].add(row["movieId"])

    precisions, ndcgs, aps = [], [], []

    for uid, group in top_n.groupby("userId"):
        relevant = user_relevant[uid]
        if not relevant:
            continue
        recommended = group.sort_values(
            "prediction", ascending=False
        )["movieId"].tolist()

        hits = [1 if mid in relevant else 0 for mid in recommended]

        precisions.append(sum(hits) / n)

        dcg  = sum(h / np.log2(i+2) for i, h in enumerate(hits))
        idcg = sum(1 / np.log2(i+2) for i in range(min(len(relevant), n)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)

        ap, hit_count = 0, 0
        for i, mid in enumerate(recommended):
            if mid in relevant:
                hit_count += 1
                ap += hit_count / (i+1)
        aps.append(ap / min(len(relevant), n))

    return {
        "Precision@10": np.mean(precisions),
        "NDCG@10":      np.mean(ndcgs),
        "MAP":          np.mean(aps),
    }

metrics = compute_metrics(preds_pd, threshold=3.5, n=10)

print("\n" + "="*55)
print("SONUÇLAR")
print("="*55)
print(f"Precision@10 : {metrics['Precision@10']:.4f}")
print(f"NDCG@10      : {metrics['NDCG@10']:.4f}")
print(f"MAP          : {metrics['MAP']:.4f}")
print(f"Eğitim süresi: {sure:.1f} sn")

# ── MODELİ KAYDET ───────────────────────────────────────────
print("\nModel kaydediliyor...")
save_path = "hdfs://namenode:9000/user/models/als_best_model_100"
model.save(save_path)
print(f"✅ Model kaydedildi: {save_path}")

spark.stop()
print("✅ Tamamlandı!")
