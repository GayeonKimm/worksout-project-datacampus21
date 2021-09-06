import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as c_s

df = pd.read_csv('../item_0810_final.csv', encoding='UTF8')
df.drop_duplicates('품번', inplace = True)
df = df.reset_index()
df_name = df[['품번', '품명']]

item_vec = pd.read_csv('../item_vector_0827.csv')
del item_vec['품번']
del item_vec['Unnamed: 0']

transformer = MinMaxScaler()
x_data = item_vec[['age', '최초판매가']]
transformer.fit(x_data)
x_data = transformer.transform(x_data)
item_vec[['age', '최초판매가']] = x_data



item_vec_array = np.array(item_vec)
df_name_array = np.array(df_name)


doc_list = []

for i in range(0, 53652):
    doc_list.append(df_name_array[i][1])

tfidf_vect_simple = TfidfVectorizer()
feature_vect_simple = tfidf_vect_simple.fit_transform(doc_list)

feature_vect_array = feature_vect_simple.todense()

result = np.hstack((item_vec_array,feature_vect_array))

#각 속성별 가중치 설정
w_age=0.24
w_price=0.183
w_cate1=0.192
w_cate2=0.192
w_brand=0.255
w_color=0.3
w_name=0.3

#가중합하여 행렬에 넣기
result[:,0]=result[:,0]*w_age
result[:,1]=result[:,1]*w_price
result[:,2:17]=result[:,2:17]*w_cate1
result[:,17:107]=result[:,17:107]*w_cate2
result[:,107:143]=result[:,107:143]*w_brand
result[:,143:175]=result[:,143:175]*w_color
result[:,175:]=result[:,175:]*w_name


# 메모리 부족으로 10000개만 계산
# 이 과정을 n=5번(사양에 따라) 반복
# 메모리가 충분하다면
# rerere_1 = c_s(result,result)
# 위의 방법 시행

rerere_1 = c_s(result,result[:10000])

np.save('./test_sim_mat', rerere_1)
# 이런식으로 분할해서 저장 or DB에 업로드!
