from pyspark.sql import SparkSession
from pyspark.sql.functions import row_number, floor, split, col, when, sum
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lag, lead, when, avg, last
import re
import os
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor


class dataset:

    def __init__(self, spark: SparkSession, df):
        self.spark = spark
        self.df = df

    # To have a sneak peek on the telemetry dataset
    def show(self, n=5):
        self.df.show(n=n)

    # Interpolate missing values by replacing each null with the average 
    # of the preceding and following non-null values
    # --> used for numerical values
    def interpolate(df, *columns):
        window_spec = Window.partitionBy("pilot_index").orderBy("frameIdentifier")
        
        for column in columns:
            df = df.withColumn("prev_value", lag(column).over(window_spec))
            df = df.withColumn("next_value", lead(column).over(window_spec))

            df = df.withColumn(column, 
                            when(col(column).isNull(), (col("prev_value") + col("next_value")) / 2)
                            .otherwise(col(column))
                            )
            df = df.drop("prev_value", "next_value")
        return df
    
    # forwardpass funtion, to replace each null value with the preceding
    # non-null values
    # --> used for categorical vaules
    def forwardpass(df, *columns):
        windowSpec = Window.partitionBy("pilot_index").orderBy("frameIdentifier").rowsBetween(-sys.maxsize, 0)

        for column in columns:
            df = df.withColumn(
                column,  # Name the output column based on the input
                last(col(column), ignorenulls=True).over(windowSpec)
            )

        return df
    
    # a mathod to transform the dataset
    def transform(self):
        self.df = self.df.orderBy(["pilot_index", "frameIdentifier"])

        # Making the pit status and resultstatus columns binary instead of String + imputating the Null variables

        self.df = self.df.withColumn("inPitArea", when(col("pitStatus").isNull(), False).otherwise(True))
        self.df = self.df.withColumn("pitting", when(col("pitStatus")=="pitting", True).otherwise(False))
        self.df = self.df.withColumn("active", when(col("resultStatus").isNull(), False).otherwise(True))

        # mapping tyre compounds to integers instead of string

        self.df = self.df.withColumn("actualTyreCompound", when(col("actualTyreCompound")=="soft", 0) \
            .when(col("actualTyreCompound")=="medium", 1) \
            .when(col("actualTyreCompound")=="hard", 2))

        #Splitting tyre and break columns with this format: FL/FR/RL/RR to 4 columns for each tyre
        split_bt = split(col("brakesTemperature"), "/")
        split_tst = split(col("tyresSurfaceTemperature"), "/")
        split_tit = split(col("tyresInnerTemperature"), "/")
        split_tp = split(col("tyresPressure"), "/")
        split_st = split(col("surfaceType"), "/")
        split_tw = split(col("tyresWear"), "/")
        split_td = split(col("tyresDamage"), "/")

        # Create new columns for each tyre
        self.df = self.df.withColumn("FL_tyresSurfaceTemperature", split_tst.getItem(0).cast("double")) \
            .withColumn("FR_tyresSurfaceTemperature", split_tst.getItem(1).cast("double")) \
            .withColumn("RL_tyresSurfaceTemperature", split_tst.getItem(2).cast("double")) \
            .withColumn("RR_tyresSurfaceTemperature", split_tst.getItem(3).cast("double")) \
            .withColumn("FL_tyresInnerTemperature", split_tit.getItem(0).cast("double")) \
            .withColumn("FR_tyresInnerTemperature", split_tit.getItem(1).cast("double")) \
            .withColumn("RL_tyresInnerTemperature", split_tit.getItem(2).cast("double")) \
            .withColumn("RR_tyresInnerTemperature", split_tit.getItem(3).cast("double")) \
            .withColumn("FL_tyresPressure", split_tp.getItem(0).cast("double")) \
            .withColumn("FR_tyresPressure", split_tp.getItem(1).cast("double")) \
            .withColumn("RL_tyresPressure", split_tp.getItem(2).cast("double")) \
            .withColumn("RR_tyresPressure", split_tp.getItem(3).cast("double")) \
            .withColumn("FL_tyresWear", split_tw.getItem(0).cast("double")) \
            .withColumn("FR_tyresWear", split_tw.getItem(1).cast("double")) \
            .withColumn("RL_tyresWear", split_tw.getItem(2).cast("double")) \
            .withColumn("RR_tyresWear", split_tw.getItem(3).cast("double")) \
            .withColumn("FL_tyresDamage", split_td.getItem(0).cast("double")) \
            .withColumn("FR_tyresDamage", split_td.getItem(1).cast("double")) \
            .withColumn("RL_tyresDamage", split_td.getItem(2).cast("double")) \
            .withColumn("RR_tyresDamage", split_td.getItem(3).cast("double")) \
            .withColumn("FL_brakesTemperature", split_bt.getItem(0).cast("double")) \
            .withColumn("FR_brakesTemperature", split_bt.getItem(1).cast("double")) \
            .withColumn("RL_brakesTemperature", split_bt.getItem(2).cast("double")) \
            .withColumn("RR_brakesTemperature", split_bt.getItem(3).cast("double")) \
            .withColumn("FL_surfaceType", split_st.getItem(0).cast("int")) \
            .withColumn("FR_surfaceType", split_st.getItem(1).cast("int")) \
            .withColumn("RL_surfaceType", split_st.getItem(2).cast("int")) \
            .withColumn("RR_surfaceType", split_st.getItem(3).cast("int")) \

        self.df = self.df.drop("resultStatus", "pitStatus", "brakesTemperature", "tyresSurfaceTemperature", "tyresInnerTemperature", "tyresPressure", "surfaceType", "tyresWear", "tyresDamage")
        
    # applying the interpolation and forward passing functions after transformation
    def imputation(self):
        self.df = self.forwardpass(self.df, "FL_surfaceType", "FR_surfaceType", "RL_surfaceType", "RR_surfaceType", "pitLimiterStatus", "actualTyreCompound", "drs", "gear", "ersDeployMode", "fuelMix")
        self.df = self.interpolate(self.df, "speed", "throttle", "steer", "brake", "clutch", "engineRPM", "engineTemperature", "fuelInTank", "fuelRemainingLaps", "ersStoreEnergy", "ersHarvestedThisLapMGUK", "ersHarvestedThisLapMGUH", "ersDeployedThisLap", "FL_tyresSurfaceTemperature","FR_tyresSurfaceTemperature","RL_tyresSurfaceTemperature","RR_tyresSurfaceTemperature","FL_tyresInnerTemperature","FR_tyresInnerTemperature","RL_tyresInnerTemperature","RR_tyresInnerTemperature","FL_tyresPressure","FR_tyresPressure","RL_tyresPressure","RR_tyresPressure","FL_tyresWear","FR_tyresWear","RL_tyresWear","RR_tyresWear","FL_tyresDamage","FR_tyresDamage","RL_tyresDamage","RR_tyresDamage","FL_brakesTemperature","FR_brakesTemperature","RL_brakesTemperature","RR_brakesTemperature")

    def datasetDev(self, dataFilePath, folder):
        #df = self.spark.read.csv(dataFilePath, header=True, inferSchema=True)

        gp_id = re.search(r'_(\d+)', dataFilePath).group(1)

        #df = df.orderBy(["pilot_index", "frameIdentifier"])

        window_spec = Window.partitionBy("pilot_index").orderBy("frameIdentifier")

        df = df.withColumn("row_num", row_number().over(window_spec))

        # Create chunk IDs (200 frames per chunk)
        df = df.withColumn("chunk_id", floor((col("row_num") - 1) / 200))

        # Save as partitioned Parquet files
        df.write.mode("overwrite").partitionBy("pilot_index", "chunk_id").option("header", "true").csv(folder + "/" + gp_id)

    def datasetPBI(self, dataFilePath, folder):
        gp_id = re.search(r'_(\d+)', dataFilePath).group(1)

        window_spec = Window.partitionBy("pilot_index").orderBy("frameIdentifier")

        df = df.withColumn("row_num", row_number().over(window_spec))

        # Save as partitioned Parquet files
        df.write.mode("overwrite").partitionBy("pilot_index").option("header", "true").csv(folder + "/" + gp_id)