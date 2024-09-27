from pyspark.sql import SparkSession
from pyspark import SparkFiles
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler

# Initialize Spark session
spark = SparkSession.builder.appName("BreastCancerAnalysis").getOrCreate()

# Load the data
URL = "https://raw.githubusercontent.com/pkmklong/Breast-Cancer-Wisconsin-Diagnostic-DataSet/master/data.csv"
df = spark.read.csv(URL, header=True, inferSchema=True)

# Check the initial schema
df.printSchema()

# Rename the diagnosis column for clarity
df = df.withColumnRenamed("diagnosis", "label")

# Map 'M' (malignant) to 1 and 'B' (benign) to 0
df = df.withColumn("label", (col("label") == "M").cast("integer"))

# Drop the 'id' column
df = df.drop("id")

# Convert all feature columns to double
for column in df.columns:
    if column != "label":
        df = df.withColumn(column, col(column).cast("double"))

# Check if all feature columns are successfully cast to double
for column in df.columns:
    if column != "label":
        dtype = df.schema[column].dataType
        if dtype != 'DoubleType':
            raise ValueError(f"Column {column} is not of type DoubleType, it is {dtype}.")

# Create feature vector
feature_columns = [column for column in df.columns if column != "label"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Transform the DataFrame
data = assembler.transform(df)

# Split the data into training and testing sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Show the final schemas and a few rows
data.printSchema()
train_data.show(3)
test_data.show(3)
