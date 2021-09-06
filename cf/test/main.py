import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

import matixFactorization as mf
import makeMatrix


### 0. Read Dataset
print("Read Dataset")
#purchase = pd.read_csv('C:/Users/User/PycharmProjects/collaborativeFiltering/dataset/purchase.csv')
purchase = pd.read_excel('C:/Users/User/PycharmProjects/collaborativeFiltering/dataset/purchase.xlsx', sheet_name = 'sample')
#purchase = pd.read_excel('C:/Users/User/PycharmProjects/collaborativeFiltering/dataset/test_purchase.xlsx')


### 1. preprocessing

# 1차 카테고리 column 생성
purchase.insert(2, 'CATEGORY1', purchase['CATEGORY2'].str[:2])

#반품 row -> /2 처리
purchase['수량'] = pd.to_numeric(purchase['수량'])
purchase['re_qty'] = np.where(purchase['구매구분'] == '반품', purchase['수량']/2, purchase['수량'])

#re_quntity 합계해서 5개에서 150개에 해당하는 고객 select
Cudf = pd.DataFrame()
sumCustomer = purchase.groupby('고객코드')['re_qty'].agg(**{'q_sum':'sum'}).reset_index()
#sumCustomer2 = sumCustomer[5<=sumCustomer.q_sum]
sumCustomer2 = sumCustomer[3<=sumCustomer.q_sum]
selCustomer = sumCustomer2[sumCustomer2.q_sum<=150]
selCustomer = selCustomer.rename(columns={"고객코드":"customerCode"}).reset_index()
selCustomer = selCustomer.drop(['index'], axis=1)

# select한 고객 df 기반으로 해당 고객 row만 select
def customerFunction(i):
    cust = purchase[purchase['고객코드'] == selCustomer.loc[i].customerCode]
    global Cudf
    Cudf = Cudf.append(cust, ignore_index=True)

print("Preprocessing Start")
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
print(pc)


### 2. make matrix
exceptduplicates = pc.drop_duplicates(['품번'])   #중복 제품 삭제
#exceptduplicates.insert(2, 'CATEGORY1', pc['CATEGORY2'].str[:2])


# 품번 추출
itemcode = list(exceptduplicates['품번'])
itemcode.sort()
print("main 품번", itemcode)

# 고객코드 추출
customercode = list(set(pc['customerCode']))
customercode.sort()
print("main customercode", customercode)

#행렬곱한 layer들 생성
print("\n** brand matrix **")
brandmatrix = makeMatrix.makelayers(pc,'BRAND')
print(brandmatrix)
print("\n** category matrix **")
categorymatrix = makeMatrix.makelayers(pc,'CATEGORY1')
print(categorymatrix)
#print("\n** color matrix **")
#colormatrix = makeMatrix.makelayers(pc, '색상')



### 3. nomalization
scaler = StandardScaler()

# column별로 scalar 되기에 transpose 해줌.
scaled_brandmatrix_T = scaler.fit_transform(brandmatrix.T)
scaled_categorymatrix_T = scaler.fit_transform(categorymatrix.T)
#scaled_colormatrix_T = scaler.fit_transform(colormatrix.T)

scaled_brandmatrix = scaled_brandmatrix_T.T
scaled_categorymatrix = scaled_categorymatrix_T.T
#scaled_colormatrix = scaled_colormatrix_T.T

print("scaled brandmatrix \n", scaled_brandmatrix)
print("scaled categorymatrix \n", scaled_categorymatrix)



### 4. weighted sum
# R= (1.293 * scaled_brandmatrix) + (1.221 * scaled_categorymatrix) + scaled_colormatrix
#R= (1.293 * scaled_brandmatrix) + (1.221 * scaled_categorymatrix)
   #+ scaled_colormatrix

R = (2*scaled_brandmatrix) + scaled_categorymatrix

print("\n** result ** \n", R)
print("shape: ", R.shape)
#np.savetxt('beforescalar.csv',R, fmt= '%f', delimiter=",")


#scaler = MinMaxScaler()
#scaled_result = scaler.fit_transform(R.T)

#MinMaxScaler = MinMaxScaler()
#scaled_result = MinMaxScaler.fit_transform(scaled_result)

#R = scaled_result.T
#np.savetxt('afterscalar.csv',R, fmt= '%f', delimiter=",")
print("afterscalar R: \n", R)

#print("\n** customercode ** \n", customercode)
#print("length: ", len(customercode))
#print("\n** itemcode ** \n", itemcode)
#print("length: ", len(itemcode))



### 5. matrix factorization

# initialize parameters
r_lambda = .1   # normalization parameter
nf = 5         # dimension of latent vector of each user and item
alpha = 40      # confidence level
EPOCH = 15

# R as the dataset (array)

# initialize user and item latent factor matrix
nu = R.shape[0] # number of users
ni = R.shape[1] # number of items

# initialize X and Y with very small values
X = np.random.rand(nu, nf) * 0.01
Y = np.random.rand(ni, nf) * 0.01

# initialize Binary rating matrix P: convert original rating matrix R to P
P = np.copy(R)
P[P > 0] = 1

# Initialize Confidence Matrix C
C = 1 + alpha * R   # confidence level of certain rating dataset

# Train
predict_errors = []
confidence_errors = []
regularization_list = []
total_losses = []

for i in tqdm(range(EPOCH)):
    if i != 0:
        mf.optimize_user(X, Y, C, P, nu, nf, r_lambda)
        mf.optimize_item(X, Y, C, P, ni, nf, r_lambda)
    predict = np.matmul(X, np.transpose(Y))
    #print(predict)
    predict_error, confidence_error, regularization, total_loss = mf.loss_function(C, P, predict, X, Y, r_lambda)

    predict_errors.append(predict_error)
    confidence_errors.append(confidence_error)
    regularization_list.append(regularization)
    total_losses.append(total_loss)

    print('\n ----------------step %d----------------' % int(i+1))
    print("predict error: %f" % predict_error)
    print("confidence error: %f" % confidence_error)
    print("regularization: %f" % regularization)
    print("total loss: %f" % total_loss)

predict = np.matmul(X, np.transpose(Y))
#print('final predict')
#np.set_printoptions(threshold=sys.maxsize)
print([predict])
#np.savetxt('aftermf.csv',predict, fmt= '%f', delimiter=",")



### 6. sorting

# return index of sorting list
sorted_result = [[0 for col in range(predict.shape[1])] for row in range(predict.shape[0])]   # null matrix
print(predict.shape[0])
for i in range(predict.shape[0]):
    sorted_result[i] = sorted(range(len(predict[i])), key=lambda k: predict[i][k], reverse=True)
    print(i, ' : ', sorted_result[i] )

sorted_result=np.array(sorted_result)

#print("\n** sorted_result **")
#print(sorted_result)

#return item code
sorted_result_item = [[0 for col in range(predict.shape[1])] for row in range(predict.shape[0])]   # null matrix
for i in range(predict.shape[0]):
    for j in range(predict.shape[1]):
        #print(itemcode[sorted_result[i][j]])
        sorted_result_item[i][j]= itemcode[sorted_result[i][j]]

sorted_result_item = np.array(sorted_result_item)
#print("\n** item code **")
#print(sorted_result_item)

##push 'sorted_result_item' into DB


#("고객코드","제품코드" 나열하여 확인)
customercode = np.array(customercode)
customercode = customercode.reshape(len(customercode),1) #2차원 배열로 reshape
concatenate = np.concatenate((customercode, sorted_result_item), axis=1) #열방향 (좌 -> 우)로 연결
#print(hstack)
np.savetxt('sample.csv',concatenate, fmt= '%s', delimiter=",")


### 7. select top-N

##pull 'sorted_result_item' from DB

#1.1. array에서 raffle 상품 제외
#1.2. array에서 현재 판매하지 않는 상품 제외
#1.3. 구매했었던 상품 제외
#2. 앞에서부터 20개 출력
#2.1. 만약 20개가 없다면 그 전까지만 출력하기
#3.