# model B
# model-b(user_based)는 DB에 안넣고 바로 사용해서 로딩시간이 꽤 소요됨

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as c_s

df_p = pd.read_csv('../dataset/purchase_0827.csv')
df_purchase = df_p[['고객코드', '품번', '수량']]
df_purchase = df_purchase.astype({'고객코드':int})
df_purchase_group = df_purchase.groupby(['고객코드', '품번'])['수량'].sum() # 0은 반품한 제품
df_purchase_group = df_purchase_group.to_frame()
df_purchase_group = df_purchase_group[df_purchase_group.수량 != 0] # 반품데이터 제거
df_purchase_group['수량'] = 1 # 수량 1로 변환
df_purchase_group.reset_index(inplace=True)
df_purchase_customer = df_purchase_group.set_index(['고객코드'])

df = pd.read_csv('../dataset/item_0810_final.csv', encoding='UTF8')

df.drop_duplicates('품번', inplace = True)
df = df.reset_index()
df_name = df[['품번', '품명']]

item_vec = pd.read_csv('../dataset/item_vector_final_0827.csv')




customer_code = int(input("고객 코드를 입력해주세요:"))

df_customer_item = df_purchase_customer[df_purchase_customer.index==customer_code]
df_customer_item = df_customer_item['품번']
item_list = df_customer_item.values

void_list = []
for item in item_list:
    void_list.append(item_vec.loc[item_vec['품번']==item].index)

df_int_index = pd.DataFrame(void_list).dropna().astype('int')

user_bought_item_index = df_int_index[0].values.tolist()

############################################################
del item_vec['품번']


transformer = MinMaxScaler()  #Min-Max-Scaler

x_data = item_vec[['age', '최초판매가']]
transformer.fit(x_data)
x_data = transformer.transform(x_data)
item_vec[['age', '최초판매가']] = x_data

item_vec_array = np.array(item_vec)

df_name_array = np.array(df_name)

doc_list = []

for i in range(0,53652):
    doc_list.append(df_name_array[i][1])

tfidf_vect_simple = TfidfVectorizer()
feature_vect_simple = tfidf_vect_simple.fit_transform(doc_list)
feature_vect_array = feature_vect_simple.todense()

result = np.hstack((item_vec_array,feature_vect_array))

#속성별 가중치 설정
w_age=0.24
w_price=0.183
w_cate1=0.192
w_cate2=0.192
w_brand=0.255
w_color=0.3
w_name=0.3

result[:,0]=result[:,0]*w_age
result[:,1]=result[:,1]*w_price
result[:,2:17]=result[:,2:17]*w_cate1
result[:,17:107]=result[:,17:107]*w_cate2
result[:,107:143]=result[:,107:143]*w_brand
result[:,143:175]=result[:,143:175]*w_color
result[:,175:]=result[:,175:]*w_name



bought_sum_result = np.zeros((1,result[0].size))

for index in user_bought_item_index:
    bought_sum_result += result[index]

bought_average_result = bought_sum_result / len(user_bought_item_index)

bought_average_sim_matrix = c_s(bought_average_result, result)

n=20 #상위 n개 뽑기

s = bought_average_sim_matrix.argsort()
s = s[0]
index_rank = s[::-1]

for index in user_bought_item_index:
    index_rank = np.delete(index_rank, index) #유저가 샀던 아이템 제외

top_n_rank = index_rank[0:n]

item_code_only = df['품번']
item_code_only.reset_index(drop=True, inplace=True)

for i in range(0,n):
    print(df[df['품번']==item_code_only[top_n_rank[i]]][['BRAND명', '품번', '품명', 'CATEGORY1', 'CATEGORY2', '최초판매가']])
