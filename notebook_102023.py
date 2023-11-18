#!/usr/bin/env python
# coding: utf-8

# In[80]:


import os
import io
import time
from socket import gethostname

import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import Row, StringType

from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import PCAModel, PCA, MinMaxScaler, StringIndexer, VectorIndexer, IndexToString
from pyspark.ml.linalg import Vectors
from pyspark.ml.functions import vector_to_array, array_to_vector
from pyspark.sql.functions import col, udf, pandas_udf, PandasUDFType, element_at, split

from pyspark import keyword_only
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.types import ArrayType, FloatType, StringType, Row
from pyspark.ml.linalg import Vectors, VectorUDT

from pyspark.sql.types import ArrayType, DoubleType



# In[81]:


# Déterminer si le code s'exécute sur AWS
is_aws = gethostname() != 'Innas-MacBook-Pro.local'

# Configuration des chemins de données
if is_aws:
    # AWS
    PATH = 's3://shved-bucket'
else:
    # Des pistes pour l'environnement local
    PATH = '/Users/innakonar/Desktop/Projet8/'
    os.environ["JAVA_HOME"] = "/usr/local/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"
    os.environ["SPARK_HOME"] = "/usr/local/Cellar/apache-spark/3.5.0/libexec"

# Voies courantes
PATH_Data = PATH + 'Test/'
PATH_RESULTS = PATH + 'Results/'
PATH_pipe_model = PATH + '/pipeline_model'
PATH_Result = PATH + '/Results_PCA'

# Création des répertoires si nécessaire
os.makedirs(PATH_RESULTS, exist_ok=True)
os.makedirs(PATH_pipe_model, exist_ok=True)
os.makedirs(PATH_Result, exist_ok=True)

# SparkSession
spark = SparkSession.builder.appName("FruitScout")

# Options supplémentaires pour le démarrage local
if not is_aws:
    spark = spark.master("local[4]").config("spark.sql.parquet.writeLegacyFormat", 'true').getOrCreate()
else:
    spark = spark.getOrCreate()


# In[82]:


print('PATH:                  ' + PATH + 
      '\nPATH_Data:           ' + PATH_Data + 
      '\nPATH_RESULTS:        ' + PATH_RESULTS + 
      '\nPATH_pipeline_model: ' + PATH_pipe_model +
      '\nPATH_Result:         ' + PATH_Result)


# In[83]:


t0 = time.time()


# In[84]:


#Récupération du label à partir du chemin du fichier
class PathToLabelTransformer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(PathToLabelTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        return dataset.withColumn("label", element_at(split(dataset["path"], "/"), -2))

  


# In[85]:


#Transformation de l’image en features 
class ImageFeatureTransformer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(ImageFeatureTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        sc = SparkSession.builder.getOrCreate().sparkContext

        def load_model():
            base_model = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
            for layer in base_model.layers:
                layer.trainable = False
            model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
            return model

        model = load_model()

        broadcast_weights = sc.broadcast(model.get_weights())

        def preprocess(content):
            img = Image.open(io.BytesIO(content)).resize([224, 224])
            arr = img_to_array(img)
            return preprocess_input(arr)

        def featurize(content):
            model.set_weights(broadcast_weights.value)
            arr = preprocess(content)
            preds = model.predict(np.expand_dims(arr, axis=0))
            return preds.flatten().tolist()

        featurize_udf = udf(featurize, ArrayType(FloatType()))
        return dataset.withColumn(self.getOutputCol(), featurize_udf(col(self.getInputCol())))


# In[86]:


#Transformation du tableau de features en vecteur de features
class ArrayToVectorTransformer(Transformer, HasInputCol, HasOutputCol,DefaultParamsReadable, DefaultParamsWritable):

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(ArrayToVectorTransformer, self).__init__()
        self._setDefault(inputCol=None, outputCol=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        toArray = udf(lambda x: Vectors.dense(x), VectorUDT())
        return dataset.withColumn(self.getOutputCol(), toArray(self.getInputCol()))


# In[87]:


images = spark.read.format("binaryFile")   .option("pathGlobFilter", "*.jpg")   .option("recursiveFileLookup", "true")   .load(PATH_Data)


# In[88]:


images.show(10)


# In[89]:


# Initialisation des transformateurs

path_to_label = PathToLabelTransformer(inputCol="path", outputCol="label")
label_indexer = StringIndexer(inputCol="label", outputCol="label_index") 
image_feature = ImageFeatureTransformer(inputCol="content", outputCol="features") 
array_to_vector = ArrayToVectorTransformer(inputCol="features", outputCol="features_vector")

# MinMaxScaler
minMaxScaler = MinMaxScaler(inputCol="features_vector", outputCol="scaled_features")

# PCA
pca = PCA(k=1000, inputCol="scaled_features", outputCol="pca_features")

# Pipeline
pipeline = Pipeline(stages=[path_to_label, image_feature, array_to_vector, label_indexer, minMaxScaler, pca])


# In[41]:


# sampled_data = images.limit(300)


# In[ ]:


# pipeline_model = pipeline.fit(sampled_data)
# transformed_df = pipeline_model.transform(sampled_data)


# In[90]:


pipeline_model = pipeline.fit(images)
transformed_df = pipeline_model.transform(images)


# In[91]:


# Sauvegarde des résultats
transformed_df.write.mode("overwrite").parquet(PATH_RESULTS)
pipeline_model.save(PATH_pipe_model)
pca_result = transformed_df.select("pca_features")
pca_result.write.mode("overwrite").parquet(PATH_Result)


# # Chargement des données enregistrées et validation du résultat

# In[92]:


df = pd.read_parquet(PATH_RESULTS, engine='pyarrow')


# In[93]:


df.tail(40)


# <!-- df1.pca_features -->

# In[94]:


# df_pca = pd.read_parquet(PATH_Result, engine='pyarrow')


# In[ ]:


# df_pca.head


# In[96]:


def sparse_vector_to_list(row):
    if isinstance(row, dict) and 'values' in row:
        return row['values']
    return row

df['pca_features_list'] = df['pca_features'].apply(sparse_vector_to_list)

df.to_csv("Result_PCA.csv", index=False)


# In[97]:


def sparse_vector_to_list(row):
   if isinstance(row, dict) and 'values' in row:
        return row['values']
   return row
df['pca_features_list'] = df['pca_features'].apply(sparse_vector_to_list)

df.to_csv("Result_PCA.csv", index=False)


# In[98]:


csv_file2 = "Result_PCA.csv"
df4 = pd.read_csv(csv_file2)
print(df.head())


# In[ ]:


PATH = os.getcwd()
pipe_mdel = Pipeline.load(PATH + '/pipeline_model')
                                                                                
exp_var_cumul = np.append(0, np.cumsum([0] + pipe_mdel.getStages()[-1].explainedVariance))
exp_var_cumul


# In[ ]:


df = pd.DataFrame({'pca': exp_var_cumul})
from plotly import express as px
fig = px.line(df, y="pca", title='Variance cumulée expliquée')
fig.update_layout({
    "width": 800,
    "height": 800,
    "xaxis_title": "Nb composantes PCA",
    "yaxis_title": "Ratio variance expliquée"
})

fig.show()


# In[168]:


elapsed_time = time.time() - t0


# In[169]:


print(f"durée d'execution: {time.time() - t0}")
w = True
while w:
    a = input("Appuyez sur Entrée pour arrêter: ")
    if a == '':
        
        w=False


# In[170]:


# spark.stop()


# In[ ]:




