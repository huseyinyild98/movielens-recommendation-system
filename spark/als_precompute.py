# als_precompute.py
# ALS modelini yükle, tüm kullanıcılar için top-20 öneri hesapla ve kaydet

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import functions as F
import time

# ── SPARK SESSION ────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("ALS_Precompute") \
    .config("spark.executor.memory", "3g") \
    .config("spark.driver.memory", "3g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("✅ Spark başlatıldı")

# ── MODELİ YÜKLE ────────────────────────────────────────────
print("Model yükleniyor...")
model = ALSModel.load("hdfs://namenode:9000/user/models/als_best_model_100")
print("✅ Model yüklendi")

# ── TÜM KULLANICILAR İÇİN TOP-20 ÖNERİ ─────────────────────
print("Tüm kullanıcılar için top-20 öneri hesaplanıyor...")
print("(Bu işlem birkaç dakika sürebilir...)")

start = time.time()
recommendations = model.recommendForAllUsers(20)
sure = time.time() - start
print(f"Öneri süresi: {sure:.1f} saniye")

# ── PANDAS'A ÇEVİR ──────────────────────────────────────────
print("Pandas'a çevriliyor...")

# recommendations DataFrame yapısı:
# userId | recommendations (array of (movieId, rating) structs)
# Bunu düz bir tabloya çevirelim

recs_exploded = recommendations.select(
    "userId",
    F.explode("recommendations").alias("rec")
).select(
    "userId",
    F.col("rec.movieId").alias("movieId"),
    F.col("rec.rating").alias("predicted_rating")
)

recs_pd = recs_exploded.toPandas()
print(f"Toplam öneri satırı: {len(recs_pd):,}")
print(f"Toplam kullanıcı   : {recs_pd['userId'].nunique():,}")

# ── HDFS'E KAYDET ───────────────────────────────────────────
print("HDFS'e kaydediliyor...")
recs_spark = spark.createDataFrame(recs_pd)
recs_spark.write.mode("overwrite").parquet(
    "hdfs://namenode:9000/user/models/als_recommendations"
)
print("✅ HDFS'e kaydedildi!")

spark.stop()
print("✅ Tamamlandı!")
print(f"\nÖrnek çıktı:")
print(recs_pd.head(10).to_string())
