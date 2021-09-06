import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF
from sklearn.metrics import mean_absolute_error

####################
### 0. Read Dataset
print("Read Dataset")
## pull Dataset from DB
purchase = pd.read_csv('./purchase.csv')
customer = pd.read_csv('./customer.csv')


####################
### 1. preprocessing

# set merge : based on purchase
purchase = pd.merge(purchase, customer[['고객코드','성별','생년월일']], how='left', left_on='고객코드', right_on='고객코드')

# 1차 카테고리 column 생성
purchase.insert(2, 'CATEGORY1', purchase['CATEGORY2'].str[:2])

#반품 row -> /2 처리
purchase['수량'] = pd.to_numeric(purchase['수량'])
purchase['re_qty'] = np.where(purchase['구매구분'] == '반품', purchase['수량']/2, purchase['수량'])

#re_quntity 합계해서 5개에서 150개에 해당하는 고객 select
Cudf = pd.DataFrame()
sumCustomer = purchase.groupby('고객코드')['re_qty'].agg(**{'q_sum':'sum'}).reset_index()
sumCustomer2 = sumCustomer[5<=sumCustomer.q_sum]
selCustomer = sumCustomer2[sumCustomer2.q_sum<=150]
selCustomer = selCustomer.rename(columns={"고객코드":"customerCode"}).reset_index()
selCustomer = selCustomer.drop(['index'], axis=1)

# select한 고객 df 기반으로 해당 고객 row만 select
def customerFunction(i):
    cust = purchase[purchase['고객코드'] == selCustomer.loc[i].customerCode]
    global Cudf
    Cudf = Cudf.append(cust, ignore_index=True)

print("\nPreprocessing Start")
i = 0
for i in tqdm(selCustomer.index):
    customerFunction(i)
    i = i + 1

#필요없는 열 삭제, rename
#Cudf=Cudf.drop(['SEASONGROUP명','YEAR명','구매매장명','품번2','카테고리0','카테고리1','CATEGORY2'], axis=1)
Cudf.rename(columns={'고객코드':'customerCode','BRAND명':'BRAND' },inplace=True)

#반품 열 삭제
reData = Cudf[Cudf['구매구분'] != '반품']
finalData = pd.DataFrame(columns=range(0))
pc = finalData.append(reData,ignore_index=True)

#price 속성 추가
def custom(price):
    if (price <= 34000):
        return 0
    elif((price > 34000) &(price <= 45000)):
        return 1
    elif ((price > 45000) & (price <= 58000)):
        return 2
    elif ((price > 58000) & (price <= 73000)):
        return 3
    elif ((price > 73000) & (price <= 89000)):
        return 4
    elif ((price > 89000) & (price <= 108000)):
        return 5
    elif ((price > 108000) & (price <= 128000)):
        return 6
    elif ((price > 128000) & (price <= 149000)):
        return 7
    elif ((price > 149000) & (price <= 199000)):
        return 8
    elif ((price > 199000) & (price <= 348000)):
        return 9
    else:
        return 10

pc["price"] = pc.apply(lambda x: custom(x['금액']) , axis = 1 )
#pc.to_excel("test.xlsx")

#age 속성 추가
df_code_age = pc[["품번", "생년월일"]]
df_code_age["age"] = df_code_age["생년월일"].astype(str).str[0:4]
#df_code_age["age"] = df_code_age.loc[:,["생년월일"]].astype(str).str[0:4]
df_code_age["age"] = df_code_age["age"].astype(int)
#this_year = 2021
#df_code_age["age"] = this_year - df_code_age["age"]
price = df_code_age.groupby(['품번'], as_index=False).mean()
pc = pd.merge(pc, price[['품번','age']], how='left', left_on='품번', right_on='품번')


####################
### 2. make matrix

def makelayers(pc, att): #pc = purchase dataset

    # 고객코드 추출
    customercode = list(set(pc['customerCode']))

    # 속성 추출
    attribute = list(set(pc[att]))

    # 행렬곱 위한 왼쪽 matrix (user x attribute)
    array = [[0 for col in range(len(attribute))] for row in range(len(customercode))]   # 빈 matrix 준비

    #구매갯수만큼 채워넣기
    for i in tqdm(range(len(customercode))):
        for j in range(len(attribute)):
            sum = pc[(pc['customerCode'] == customercode[i]) & (pc[att] == attribute[j])]
            array[i][j] = sum['re_qty'].sum()

    leftdf = pd.DataFrame(data=array, index=customercode, columns=attribute)
    attribute.sort()
    aftersort = leftdf[attribute]

    left = aftersort
    np_left = left.to_numpy()

    #행렬곱위한 오른쪽 matrix (item x attribute)
    onehotencoding_c=pd.get_dummies(pc[['품번',att]], columns = [att])
    onehotencoding_c.to_excel
    drop_dup = onehotencoding_c.drop_duplicates('품번')
    rightdf = drop_dup.set_index('품번')

    right = rightdf.sort_values('품번')
    np_right = right.to_numpy()

    #행렬곱하여 최종 matrix 완성
    final = np.matmul(np_left, np.transpose(np_right))

    return final


# 품번 추출
exceptduplicates = pc.drop_duplicates(['품번'])  #중복 제품 삭제
itemcode = list(exceptduplicates['품번'])
itemcode.sort()


# 고객코드 추출
customercode = list(set(pc['customerCode']))

#행렬곱한 layer들 생성
print("\n** brand matrix **")
brandmatrix = makelayers(pc,'BRAND')
R_max = np.max(brandmatrix)
R_min = np.min(brandmatrix)
print("max of orgin matrix: ", R_max, "min of origin matrix:", R_min)

print("\n** category matrix **")
categorymatrix = makelayers(pc,'CATEGORY1')
R_max = np.max(categorymatrix)
R_min = np.min(categorymatrix)
print("max of orgin matrix: ", R_max, "min of origin matrix:", R_min)

print("\n** price matrix**")
pricematrix = makelayers(pc, 'price')
R_max = np.max(pricematrix)
R_min = np.min(pricematrix)
print("max of orgin matrix: ", R_max, "min of origin matrix:", R_min)

print("\n** color matrix **")
colormatrix = makelayers(pc, '색상')
R_max = np.max(colormatrix)
R_min = np.min(colormatrix)
print("max of orgin matrix: ", R_max, "min of origin matrix:", R_min)

print("\n** age matrix **")
agematrix = makelayers(pc, 'age')
R_max = np.max(agematrix)
R_min = np.min(agematrix)
print("max of orgin matrix: ", R_max, "min of origin matrix:", R_min)



####################
### 3. nomalization
scaler = MinMaxScaler()

def scaling(matrix):
    # column별로 scalar 되기에 transpose 해줌.
    scaled_matrix_T = scaler.fit_transform(matrix.T)
    scaled_matrix = scaled_matrix_T.T
    return scaled_matrix

scaled_brandmatrix = scaling(brandmatrix)
scaled_categorymatrix = scaling(categorymatrix)
scaled_colormatrix = scaling(colormatrix)
scaled_pricematrix = scaling(pricematrix)
scaled_agematrix = scaling(agematrix)



####################
### 4. weighted sum
# brand: 0.293, price: 0.21, category: 0.221, age: 0.276
R= (1.293 * scaled_brandmatrix) + (1.221 * scaled_categorymatrix) + (1.21 * scaled_pricematrix) + (1.276 * scaled_agematrix) + scaled_colormatrix


####################
### 5. matrix factorization

# NMF(n_components=None, *, init='warn', solver='cd', beta_loss='frobenius',
#   tol=0.0001, max_iter=200, random_state=None, alpha=0.0, l1_ratio=0.0,
#   verbose=0, shuffle=False, regularization='both')
print("\nProcessing, please wait")
model = NMF(n_components=200, init='random', random_state=0, max_iter=1000)
W = model.fit_transform(R)
H = model.components_
predict = np.matmul(W,H)


forDBinput = pd.DataFrame(predict, columns=itemcode, index=customercode)
forDBinput.to_excel('forDBinput.xlsx')

##push 'forDBinput' into DB


### 5.1. evaluation
## use mae to evaluation the model
# the boundary of the orgin matirx
R_max = np.max(R)
R_min = np.min(R)
boundary = R_max - R_min
print("max of orgin matrix: ",R_max ,"min of origin matrix:",R_min ,"boundary: ", boundary)
# calculate mae (origin matrix R & after mf matrix predict)
mae = mean_absolute_error(R, predict)
print("MAE: ", mae)


#####################################################################################################
# 그냥 결과 제대로 나오는지 확인용
### 6. sorting

# return index of sorting list
sorted_result = [[0 for col in range(R.shape[1])] for row in range(R.shape[0])]   # null matrix
for i in range(R.shape[0]):
    sorted_result[i] = sorted(range(len(R[i])), key=lambda k: R[i][k], reverse=True)
    #print(i, ' : ', sorted_result[i] )

sorted_result=np.array(sorted_result)

sorted_result_item = [[0 for col in range(R.shape[1])] for row in range(R.shape[0])]   # null matrix
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        sorted_result_item[i][j]= itemcode[sorted_result[i][j]]

sorted_result_item = np.array(sorted_result_item)


#("고객코드","제품코드" 나열하여 확인)
customercode = np.array(customercode)
customercode = customercode.reshape(len(customercode),1) #2차원 배열로 reshape
concatenate = np.concatenate((customercode, sorted_result_item), axis=1) #열방향 (좌 -> 우)로 연결
np.savetxt('result.csv',concatenate, fmt= '%s', delimiter=",")
print("Finished, result saved")


### 7. select top-N (DB에서 실행, 쿼리문으로 post-filtering)

# 1. raffle 상품 제외
# 2. 현재 판매하지 않는 상품 제외
# 3. 고객이 구매했었던 상품 제외