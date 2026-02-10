from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import time

INPUT_PATH = "hdfs:///user/hadoop/churn_input/Churn_Modelling.csv"

def run(data, use_cats: bool):
    numeric = ["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]
    cats = ["Geography","Gender"]

    stages = []
    inputs = list(numeric)

    if use_cats:
        idx_cols = [c+"Index" for c in cats]
        vec_cols = [c+"Vec" for c in cats]

        for c, out in zip(cats, idx_cols):
            stages.append(StringIndexer(inputCol=c, outputCol=out, handleInvalid="keep"))

        stages.append(OneHotEncoder(inputCols=idx_cols, outputCols=vec_cols))
        inputs += vec_cols

    assembler = VectorAssembler(inputCols=inputs, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
    lr = LogisticRegression(labelCol="label", featuresCol="scaledFeatures", maxIter=50)

    pipeline = Pipeline(stages=stages + [assembler, scaler, lr])

    train, test = data.randomSplit([0.8, 0.2], seed=42)

    t0 = time.time()
    model = pipeline.fit(train)
    preds = model.transform(test)
    runtime = time.time() - t0

    acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    ).evaluate(preds)

    auc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(preds)

    return acc, auc, runtime

def main():
    spark = SparkSession.builder.appName("ChurnPipeline").getOrCreate()

    df = (spark.read.csv(INPUT_PATH, header=True, inferSchema=True)
          .drop("RowNumber","CustomerId","Surname")
          .withColumn("label", col("Exited").cast("double")))
    df.cache()
    print("Rows:", df.count())
    df.printSchema()

    acc1, auc1, t1 = run(df, use_cats=True)
    print("\n=== EXP A: numeric + categoricals (Geography/Gender) ===")
    print(f"Accuracy: {acc1:.4f} | AUC: {auc1:.4f} | Runtime: {t1:.2f}s")

    acc2, auc2, t2 = run(df, use_cats=False)
    print("\n=== EXP B: numeric only (ablation, no categoricals) ===")
    print(f"Accuracy: {acc2:.4f} | AUC: {auc2:.4f} | Runtime: {t2:.2f}s")

    spark.stop()

if __name__ == "__main__":
    main()
