import h2o
import pandas as pd
from h2o.estimators import H2OGradientBoostingEstimator
h2o.init(nthreads=-1, max_mem_size="1g", verbose=False)

df = pd.DataFrame({
    'f1': [1,2,3,4,5,6,7,8,9,10],
    'target': ['Y','N','Y','N','Y','N','Y','N','Y','N']
})
hf = h2o.H2OFrame(df)
hf['target'] = hf['target'].asfactor()

model = H2OGradientBoostingEstimator(ntrees=5)
model.train(x=['f1'], y='target', training_frame=hf)
perf = model.model_performance(hf)

print("AUC:", perf.auc())
print("Logloss:", perf.logloss())
try:
    print("Accuracy (binomial):", perf.accuracy()[0][1])
except Exception as e:
    print("Error accuracy:", e)
try:
    print("mean_per_class_error:", perf.mean_per_class_error())
except Exception as e:
    print("Error mean_per_class_error:", e)

h2o.cluster().shutdown()
