import pandas as pd
from sentence_transformers import SentenceTransformer

from simatcher.algorithm.beta import SentenceFaiss


intents = [
    ['正式服发布', 5],
    ['测试服重启', 1],
    ['测试服发布', 2],
    ['QA服务重启', 3],
    ['重启预发布环境', 4],
]
df = pd.DataFrame(intents, columns=['utterance', 'id'])
print(df)
print('---------------------')
text = df['utterance']
encoder = SentenceTransformer("./model/sbert-chinese-general-v2")
vectors = encoder.encode(text)
sf = SentenceFaiss(vectors)
sf.train()
query = '正式服重启'
query_vector = encoder.encode(query)
similarity = sf.process(query_vector, 5)
results = pd.DataFrame({
    'distances': similarity['distances'][0],
    'ann': similarity['ann'][0]
})
print(results)
print('---------------------')
merge = pd.merge(results, df, left_on='ann', right_index=True)
print(merge)
print('---------------------')
labels = df['id']
print(results['ann'])
id = labels[similarity['ann'][0][0]]
print(id)
