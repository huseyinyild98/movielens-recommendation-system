from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, collect_list, count, row_number, rand, greatest, lit
from pyspark.sql.window import Window
import time
import math

# ─────────────────────────────────────────
# 1. Spark Session
# ─────────────────────────────────────────
spark = SparkSession.builder \
    .appName("MovieLens-ALS-Recommender") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "1g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ─────────────────────────────────────────
# 2. Kombinasyonlar
# ─────────────────────────────────────────
combinations = [
    {"label": "ALS-K1", "rank": 10,  "maxIter": 5,  "threshold": 4.0, "min_user": 20},
    {"label": "ALS-K2", "rank": 10,  "maxIter": 5,  "threshold": 3.5, "min_user": 50},
    {"label": "ALS-K3", "rank": 50,  "maxIter": 10, "threshold": 4.0, "min_user": 50},
    {"label": "ALS-K4", "rank": 50,  "maxIter": 10, "threshold": 3.5, "min_user": 50},
]

# ─────────────────────────────────────────
# 3. Metrik Fonksiyonları
# ─────────────────────────────────────────
def precision_at_k(recommended, relevant, k):
    rec  = [int(x) for x in recommended[:k]]
    rel  = set(int(x) for x in relevant)
    hits = len(set(rec) & rel)
    return hits / k if k > 0 else 0.0

def recall_at_k(recommended, relevant, k):
    rec  = [int(x) for x in recommended[:k]]
    rel  = set(int(x) for x in relevant)
    hits = len(set(rec) & rel)
    return hits / len(rel) if len(rel) > 0 else 0.0

def ndcg_at_k(recommended, relevant, k):
    rec  = [int(x) for x in recommended[:k]]
    rel  = set(int(x) for x in relevant)
    dcg  = sum(1.0/math.log2(i+2) for i, item in enumerate(rec) if item in rel)
    idcg = sum(1.0/math.log2(i+2) for i in range(min(len(rel), k)))
    return dcg/idcg if idcg > 0 else 0.0

def average_precision(recommended, relevant, k):
    rec  = [int(x) for x in recommended[:k]]
    rel  = set(int(x) for x in relevant)
    hits, ap = 0, 0.0
    for i, item in enumerate(rec):
        if item in rel:
            hits += 1
            ap   += hits / (i+1)
    return ap / min(len(rel), k) if rel else 0.0

# ─────────────────────────────────────────
# 4. Veriyi HDFS'ten Oku
# ─────────────────────────────────────────
print("\n[1] HDFS'ten veri okunuyor...")
start = time.time()

ratings = spark.read.csv(
    "hdfs://namenode:9000/movielens/rating.csv",
    header=True,
    inferSchema=True
).select(
    col("userId").cast("int"),
    col("movieId").cast("int"),
    col("rating").cast("float")
)

print(f"  Toplam rating : {ratings.count():,}")
print(f"  Sure          : {time.time()-start:.1f}s")

# ─────────────────────────────────────────
# 5. Her Kombinasyon İçin Çalıştır
# ─────────────────────────────────────────
all_results = []
K = 10

for combo in combinations:
    label     = combo["label"]
    rank      = combo["rank"]
    maxIter   = combo["maxIter"]
    threshold = combo["threshold"]
    min_user  = combo["min_user"]

    print(f"\n{'='*60}")
    print(f"KOMBINASYON: {label}")
    print(f"  rank={rank}, maxIter={maxIter}, threshold={threshold}, min_user={min_user}")
    print(f"{'='*60}")

    # ── Filtreler ──
    print("\n  [a] Filtreler uygulanıyor...")
    popular_movies = ratings.groupBy("movieId") \
        .agg(count("rating").alias("cnt")) \
        .filter(col("cnt") >= 50) \
        .select("movieId")

    active_users = ratings.groupBy("userId") \
        .agg(count("rating").alias("cnt")) \
        .filter(col("cnt") >= min_user) \
        .select("userId")

    rf = ratings.join(popular_movies, on="movieId").join(active_users, on="userId")
    rf.cache()

    # ── Stratified Split (her kullanıcıdan en az 1 test) ──
    print("  [b] Stratified split yapiliyor...")
    window    = Window.partitionBy("userId").orderBy(rand(seed=42))
    total_win = Window.partitionBy("userId")

    rf_ranked = rf \
        .withColumn("rn",    row_number().over(window)) \
        .withColumn("total", count("*").over(total_win)) \
        .withColumn("n_test", greatest(lit(1), (col("total") * 0.2).cast("int")))

    test_df  = rf_ranked.filter(col("rn") <= col("n_test")) \
                        .select("userId","movieId","rating")
    train_df = rf_ranked.filter(col("rn") > col("n_test")) \
                        .select("userId","movieId","rating")

    train_df.cache()
    test_df.cache()

    print(f"    Train: {train_df.count():,} | Test: {test_df.count():,}")

    # ── Ground Truth ──
    ground_truth = test_df.filter(col("rating") >= threshold) \
        .groupBy("userId") \
        .agg(collect_list(col("movieId").cast("int")).alias("relevant_items"))

    test_users = ground_truth.select("userId")
    print(f"    Ground truth kullanici: {test_users.count():,}")

    # ── ALS Eğitimi ──
    print(f"  [c] ALS egitiliyor (rank={rank}, maxIter={maxIter})...")
    start = time.time()

    als = ALS(
        rank=rank,
        maxIter=maxIter,
        regParam=0.1,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop"
    )
    model = als.fit(train_df)
    print(f"    Egitim suresi: {time.time()-start:.1f}s")

    # ── Sadece Test Setindeki Filmlere Tahmin (Surprise ile aynı yaklaşım) ──
    print(f"  [d] Test setindeki filmlere tahmin yapiliyor...")
    start = time.time()

    # Test setindeki user-movie çiftleri
    test_pairs  = test_df.select("userId", "movieId")
    predictions = model.transform(test_pairs).filter(col("prediction").isNotNull())

    # Her kullanıcı için en yüksek tahminli K filmi seç
    rank_window = Window.partitionBy("userId").orderBy(col("prediction").desc())
    top_k = predictions \
        .withColumn("rank", row_number().over(rank_window)) \
        .filter(col("rank") <= K)

    recommended = top_k.groupBy("userId") \
        .agg(collect_list(col("movieId").cast("int")).alias("recommended_items"))

    print(f"    Sure: {time.time()-start:.1f}s")

    # ── Metrik Hesaplama ──
    print("  [e] Metrikler hesaplaniyor...")
    eval_df = recommended.join(ground_truth, on="userId")
    results = eval_df.collect()

    precision_scores, recall_scores, ndcg_scores, ap_scores = [], [], [], []
    for row in results:
        rec = row["recommended_items"]
        rel = row["relevant_items"]
        precision_scores.append(precision_at_k(rec, rel, K))
        recall_scores.append(recall_at_k(rec, rel, K))
        ndcg_scores.append(ndcg_at_k(rec, rel, K))
        ap_scores.append(average_precision(rec, rel, K))

    n      = len(results)
    result = {
        "label"            : label,
        "rank"             : rank,
        "maxIter"          : maxIter,
        "threshold"        : threshold,
        "min_user"         : min_user,
        f"Precision@{K}"   : round(sum(precision_scores)/n, 4),
        f"Recall@{K}"      : round(sum(recall_scores)/n, 4),
        f"NDCG@{K}"        : round(sum(ndcg_scores)/n, 4),
        "MAP"              : round(sum(ap_scores)/n, 4),
        "N_users"          : n
    }
    all_results.append(result)

    print(f"\n  SONUC: {label}")
    print(f"    Precision@{K} : {result[f'Precision@{K}']}")
    print(f"    Recall@{K}    : {result[f'Recall@{K}']}")
    print(f"    NDCG@{K}      : {result[f'NDCG@{K}']}")
    print(f"    MAP           : {result['MAP']}")
    print(f"    N_users       : {result['N_users']:,}")

    rf.unpersist()
    train_df.unpersist()
    test_df.unpersist()

# ─────────────────────────────────────────
# 6. Tüm Sonuçlar
# ─────────────────────────────────────────
print("\n" + "="*60)
print("TUM KOMBINASYON SONUCLARI")
print("="*60)
print(f"{'Label':<12} {'Rank':<6} {'Iter':<6} {'Thr':<5} {'MinU':<6} "
      f"{'Prec@10':<10} {'Rec@10':<10} {'NDCG@10':<10} {'MAP':<10}")
print("-"*75)
for r in all_results:
    print(f"{r['label']:<12} {r['rank']:<6} {r['maxIter']:<6} "
          f"{r['threshold']:<5} {r['min_user']:<6} "
          f"{r[f'Precision@{K}']:<10} {r[f'Recall@{K}']:<10} "
          f"{r[f'NDCG@{K}']:<10} {r['MAP']:<10}")

best = max(all_results, key=lambda x: x[f"NDCG@{K}"])
print(f"\n🏆 EN IYI KOMBINASYON (NDCG@{K} bazli): {best['label']}")
print(f"   Precision@{K} : {best[f'Precision@{K}']}")
print(f"   Recall@{K}    : {best[f'Recall@{K}']}")
print(f"   NDCG@{K}      : {best[f'NDCG@{K}']}")
print(f"   MAP           : {best['MAP']}")

spark.stop()
print("\nTamamlandi!")
