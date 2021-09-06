import pandas as pd
import numpy as np
from tqdm import tqdm

def makelayers(pc, att): #pc = purchase dataset

    # 고객코드 추출
    customercode = list(set(pc['customerCode']))

    # 카테고리 추출
    attribute = list(set(pc[att]))

    # 행렬곱 위한 왼쪽 matrix (user x category)
    array = [[0 for col in range(len(attribute))] for row in range(len(customercode))]   # 빈 matrix 준비

    #구매갯수만큼 채워넣기
    for i in tqdm(range(len(customercode))):
        for j in range(len(attribute)):
            array[i][j] = len(pc[(pc['customerCode']== customercode[i]) & (pc[att] == attribute[j])])

    leftdf = pd.DataFrame(data=array, index=customercode, columns=attribute)
    attribute.sort()
    aftersort = leftdf[attribute]

    left = aftersort
    np_left = left.to_numpy()


    #행렬곱위한 오른쪽 matrix (item x category)
    onehotencoding_c=pd.get_dummies(pc[['품번',att ]], columns = [att])
    drop_dup = onehotencoding_c.drop_duplicates()
    rightdf = drop_dup.set_index('품번')

    right = rightdf.sort_values('품번')
    np_right = right.to_numpy()


    print("left \n", np_left)
    print("right \n", np_right.T)

    #행렬곱하여 최종 matrix 완성
    final = np.matmul(np_left, np.transpose(np_right))

    print("makematrix customercode", customercode)
    print("makematrix attribute", attribute)

    return final