import random
import numpy as np
import copy
from itertools import product, combinations
import math

count = 0
x1min = -8
x1max = 5
x2min = -5
x2max = 7
x3min = -5
x3max = 10
x01 = (x1max + x1min)/2
x02 = (x2max + x2min)/2
x03 = (x3max + x3min)/2
x1del = x1max - x01
x2del = x2max - x02
x3del = x3max - x03

xSrmax = x1max + x2max + x3max / 3
xSrmin = x1min + x2min + x3min / 3

gt = {1: 0.6798, 2: 0.5157, 3: 0.4377, 4: 0.3910, 5: 0.3595, 6: 0.3362, 7: 0.3185, 8: 0.3043, 9: 0.2926, 10: 0.2829}
print("\nРівняння регресії ПФЕ")
print("y=b0+b1*x1+b2*x2+b3*x3+b12*x1*x2+b13*x1*x3+b23*x2*x3+b123*x1*x2*x3+b11*x1^2+b22*x2^2+b33*x3^2\n")
print("Нормована матриця")
print("{:5} {:5} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6}".format("№","X1","X2","X3","X4","X5","X6","X7","X8","X9","X10"))
X11 = [-1, -1, -1, -1, 1, 1, 1, 1, -1.215, 1.215, 0, 0, 0, 0, 0]
X22 = [-1, -1, 1, 1, -1, -1, 1, 1, 0, 0, -1.215, 1.215, 0, 0, 0]
X33 = [-1, 1, -1, 1, -1, 1, -1, 1, 0, 0, 0, 0, -1.215, 1.215, 0]

def sumkf2(x1,x2):
    xn = []
    for i in range(len(x1)):
        xn.append(x1[i]*x2[i])
    return xn
def sumkf3(x1,x2,x3):
    xn = []
    for i in range(len(x1)):
        xn.append(x1[i]*x2[i]*x3[i])
    return xn
def kv(x):
    xn = []
    for i in range(len(x)):
        xn.append(x[i]*x[i])
    return xn
X12 = sumkf2(X11,X22)
X13 = sumkf2(X11,X33)
X23 = sumkf2(X22,X33)
X123 = sumkf3(X11,X22,X33)
X8 = kv(X11)
X9 = kv(X22)
X10 = kv(X33)

X00 = [1]*15
Xnorm=[X00,X11,X22,X33,sumkf2(X11,X22),sumkf2(X22,X33),sumkf2(X22,X33),sumkf3(X11,X22,X33)]
for i in range(15):
    print("{:1} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6}".format(i+1,X11[i],X22[i],X33[i],X12[i],X13[i],X23[i],X123[i],round(X8[i],3),round(X9[i],3),round(X10[i],3)))
gt = { 12: {1: 0.5410, 2: 0.3924, 3: 0.3264, 4: 0.2880, 5: 0.2624, 6: 0.2439, 7:
0.2299, 8: 0.2187, 9: 0.2098, 10: 0.2020},
       15: {1: 0.4709, 2: 0.3346, 3: 0.2758, 4: 0.2419, 5: 0.2159, 6: 0.2034, 7:
0.1911, 8: 0.1815, 9: 0.1736, 10: 0.1671}}
matrix_with_min_max_x = np.array([[-8, 5], [-5, 7], [-5, 10]])
def make_plan_matrix_from_norm_matrix(norm_matrix):
    plan_matrix = np.empty((len(norm_matrix), len(norm_matrix[0])),
dtype=np.float)
    for i in range(len(norm_matrix)):
        for j in range(len(norm_matrix[i])):
            if norm_matrix[i, j] == -1:
                 plan_matrix[i, j] = matrix_with_min_max_x[j - 1][0]
            elif norm_matrix[i, j] == 1 and j != 0:
                 plan_matrix[i, j] = matrix_with_min_max_x[j - 1][1]
            elif norm_matrix[i, j] == 1 and j == 0:
                 plan_matrix[i, j] = 1
            else:
                 mean = np.mean(matrix_with_min_max_x[j - 1])
                 plan_matrix[i, j] = norm_matrix[i, j] * (matrix_with_min_max_x[j - 1][1] - mean) + mean
            return plan_matrix

def make_linear_equation():
    norm_matrix = np.array(list(product("01", repeat=3)), dtype=np.int)
    norm_matrix[norm_matrix == 0] = -1
    norm_matrix = np.insert(norm_matrix, 0, 1, axis=1)
    plan_matrix = make_plan_matrix_from_norm_matrix(norm_matrix)
    return norm_matrix, plan_matrix

def make_equation_with_quadratic_terms(current_norm_matrix):
    norm_matrix_second_part = np.empty((3, 7))
    key = 0
    for i in range(3):
        j = 0
        while j < 7:
            if j == key:
                norm_matrix_second_part[i][key] = -1.215
                norm_matrix_second_part[i][key + 1] = 1.215
                j += 1
            else:
                norm_matrix_second_part[i][j] = 0
                j += 1
            key += 2

ymax = int(200+xSrmax)
ymin = int(200+xSrmin)
print("\nМатриця плану експерименту")
print("{:3} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6}".format("№","X0","X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","Y1","Y2","Y3"))
X1 = [x1min, x1min, x1min, x1min, x1max, x1max, x1max, x1max, -1.215*x1del+x01, 1.215*x1del+x01, x01, x01, x01, x01, x01]
X2 = [x2min, x2min, x2max, x2max, x2min, x2min, x2max, x2max, x02, x02, -1.215*x2del+x02, 1.215*x2del+x02, x02, x02, x02]
X3 = [x3min, x3max, x3min, x3max, x3min, x3max, x3min, x3max, x03, x03, x03, x03, -1.215*x3del+x03, 1.215*x3del+x03, x03]
Y1 = [random.randrange(175,261, 1) for i in range(15)]
Y2 = [random.randrange(175,261, 1) for i in range(15)]
Y3 = [random.randrange(175,261, 1) for i in range(15)]
X12 = sumkf2(X1,X2)
X13 = sumkf2(X1,X3)
X23 = sumkf2(X2,X3)
X123 = sumkf3(X1,X2,X3)
X8 = kv(X1)
X9 = kv(X2)
X10 = kv(X3)
X0 = [1]*15
Xall=[X0,X1,X2,X3,X12,X13,X23,X123,X8,X9,X10]
for i in range(15):
    print("{:1} {:5} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} ".format(i+1,X0[i],round(X1[i],3),round(X2[i],3),round(X3[i],3),round(X12[i],3),round(X13[i],3),round(X23[i],3),round(X123[i],3),round(X8[i],3),round(X9[i],3),round(X10[i],3),Y1[i],Y2[i],Y3[i]))
m=3
N=15
print("\nМатриця відгуків")
Y_matrix = []
Y_matrix = np.random.randint(200 + np.mean(matrix_with_min_max_x,
axis=0)[0],
                             200 + np.mean(matrix_with_min_max_x,
axis=0)[1], size=(N, m))
print(Y_matrix)
def yav(i):
    return (Y1[i-1]+Y2[i-1]+Y3[i-1])/3
ySr1 = yav(1)
ySr2 = yav(2)
ySr3 = yav(3)
ySr4 = yav(4)
ySr5 = yav(5)
ySr6 = yav(6)
ySr7 = yav(7)
ySr8 = yav(8)
ySr9 = yav(9)
ySr10 = yav(10)
ySr11 = yav(11)
ySr12 = yav(12)
ySr13 = yav(13)
ySr14 = yav(14)
ySr15 = yav(15)

mx1 = sum(X1)/15
mx2 = sum(X2)/15
mx3 = sum(X3)/15

Yall=[ySr1, ySr2, ySr3, ySr4, ySr5, ySr6, ySr7, ySr8, ySr9, ySr10, ySr11, ySr12, ySr13, ySr14, ySr15]

def mni(j):
    s=0
    for i in range(len(Xall[j])):
        s=s+Xall[j][i]
    return s/len(Xall[j])

def mni2(j,k):
    s=0
    for i in range(len(Xall[j])):
        s=s+Xall[j][i]*Xall[k][i]
    return s/len(Xall[j])
Yalls=sum(Yall)/15
def kmn(j):
    s=0
    for i in range(len(Yall)):
        s=s+yav(i)*Xall[j][i]
    return s/len(Yall)
M, P = 11, 11
m = [[0 for y in range(M)] for x in range(P)]
m[0][0] = 1
for i in range(len(m)):
    for j in range(len(m[0])):
        if i == 0:
            m[i][j] = m[j][i] = mni(j)
        if (i != 0) and (j != 0):
            m[i][j] = mni2(i,j)

k = [Yalls]
for i in range(10):
    k.append(kmn(i+1))
b0m = []
b = []
for i in range(11):
    b0m.append(copy.deepcopy(m))
    b0m[i][i] = k
    b.append(np.linalg.det(b0m[i])/np.linalg.det(m))


print("\nПеревірка дисперсії на однорідність за критерієм Кохрена")
d=[]
for i in range(15):
    d.append(((Y1[i] - ySr1)**2 + (Y2[i] - ySr2)**2 + (Y3[i] - ySr3)**2)/3)
    print("Dispersion"+str(i+1)+"="+str(round(d[i],2)))

m = 3
Gp = max(d)/sum(d)
f1 = m-1
f2 = N = 15
Gt = 0.3346
if Gp < Gt:
    print("Дисперсія однорідна")
else:
    print("Дисперсія  неоднорідна")

print("\nКритерій Стьюдента")
sb = sum(d)/N
ssbs = sb/N*m
sbs = math.sqrt(ssbs)

beta0 = (ySr1*1+ySr2*1+ySr3*1+ySr4*1+ySr5*1+ySr6*1+ySr7*1+ySr8*1+ySr9*(-1.215)+ySr10*1.215+ySr11*0+ySr12*0+ySr13*0+ySr14*0+ySr15*0)/15
beta1 = (ySr1*(-1)+ySr2*(-1)+ySr3*(-1)+ySr4*(-1)+ySr5*1+ySr6*1+ySr7*1+ySr8*1+ySr9*0+ySr10*0+ySr11*(-1.215)+ySr12*1.215+ySr13*0+ySr14*0+ySr15*0)/15
beta2 = (ySr1*(-1)+ySr2*(-1)+ySr3*1+ySr4*1+ySr5*(-1)+ySr6*(-1)+ySr7*1+ySr8*1+ySr9*0+ySr10*0+ySr11*0+ySr12*0+ySr13*(-1.215)+ySr14*1.215+ySr15*0)/15
beta3 = (ySr1*(-1)+ySr2*1+ySr3*(-1)+ySr4*1+ySr5*(-1)+ySr6*1+ySr7*(-1)+ySr8*1)/15
beta4 = (ySr1*1+ySr2*1+ySr3*(-1)+ySr4*(-1)+ySr5*(-1)+ySr6*(-1)+ySr7*1+ySr8*1)/15
beta5 = (ySr1*1+ySr2*(-1)+ySr3*1+ySr4*(-1)+ySr5*(-1)+ySr6*1+ySr7*(-1)+ySr8*1)/15
beta6 = (ySr1*1+ySr2*(-1)+ySr3*(-1)+ySr4*1+ySr5*1+ySr6*(-1)+ySr7*(-1)+ySr8*1)/15
beta7 = (ySr1*(-1)+ySr2*1+ySr3*1+ySr4*(-1)+ySr5*1+ySr6*(-1)+ySr7*(-1)+ySr8*1)/15

beta8 = (ySr1*1+ySr2*1+ySr3*1+ySr4*1+ySr5*1+ySr6*1+ySr7*1+ySr8*1+ySr9*1.46723+ySr10*1.46723)/15
beta9 = (ySr1*1+ySr2*1+ySr3*1+ySr4*1+ySr5*1+ySr6*1+ySr7*1+ySr8*1+ySr11*1.46723+ySr12*1.46723)/15
beta10 =(ySr1*1+ySr2*1+ySr3*1+ySr4*1+ySr5*1+ySr6*1+ySr7*1+ySr8*1+ySr13*1.46723+ySr14*1.46723)/15
beta = [beta0,beta1,beta2,beta3,beta4,beta5,beta6,beta7,beta8,beta9,beta10]
t = []
for i in range(11):
    t.append(abs(beta[i])/sbs)


#print(t0,t1,t2,t3)

f3 = f1*f2
ttabl  = 2.042
print("f3 = f1*f2, з таблиці tтабл = 2.306")
#print(t0,t1,t2,t3)
d=11
for i in range(11):
    if (t[i]<ttabl):
        print("t"+str(i)+"<ttabl, b"+str(i)+" не значимий")
        b[i]=0
        d=d-1
yy1 = b[0]+b[1]*x1min+b[2]*x2min+b[3]*x3min+b[4]*x1min*x2min+b[5]*x1min*x3min+b[6]*x2min*x3min+b[7]*x1min*x2min*x3min+b[8]*x1min*x1min+b[9]*x2min*x2min+b[10]*x3min*x3min
yy2 = b[0]+b[1]*x1min+b[2]*x2min+b[3]*x3max+b[4]*x1min*x2min+b[5]*x1min*x3max+b[6]*x2min*x3max+b[7]*x1min*x2min*x3max+b[8]*x1min*x1min+b[9]*x2min*x2min+b[10]*x3max*x3max
yy3 = b[0]+b[1]*x1min+b[2]*x2max+b[3]*x3min+b[4]*x1min*x2max+b[5]*x1min*x3min+b[6]*x2max*x3min+b[7]*x1min*x2max*x3min+b[8]*x1min*x1min+b[9]*x2max*x2max+b[10]*x3min*x3min
yy4 = b[0]+b[1]*x1min+b[2]*x2max+b[3]*x3max+b[4]*x1min*x2max+b[5]*x1min*x3max+b[6]*x2max*x3max+b[7]*x1min*x2max*x3max+b[8]*x1min*x1min+b[9]*x2max*x2max+b[10]*x3max*x3max
yy5 = b[0]+b[1]*x1max+b[2]*x2min+b[3]*x3min+b[4]*x1max*x2min+b[5]*x1max*x3min+b[6]*x2min*x3min+b[7]*x1max*x2min*x3min+b[8]*x1max*x1max+b[9]*x2min*x2min+b[10]*x3min*x3min
yy6 = b[0]+b[1]*x1max+b[2]*x2min+b[3]*x3max+b[4]*x1max*x2min+b[5]*x1max*x3max+b[6]*x2min*x3max+b[7]*x1max*x2min*x3max+b[8]*x1max*x1max+b[9]*x2min*x2min+b[10]*x3min*x3max
yy7 = b[0]+b[1]*x1max+b[2]*x2max+b[3]*x3min+b[4]*x1max*x2max+b[5]*x1max*x3min+b[6]*x2max*x3min+b[7]*x1max*x2min*x3max+b[8]*x1max*x1max+b[9]*x2max*x2max+b[10]*x3min*x3min
yy8 = b[0]+b[1]*x1max+b[2]*x2max+b[3]*x3max+b[4]*x1max*x2max+b[5]*x1max*x3max+b[6]*x2max*x3max+b[7]*x1max*x2max*x3max+b[8]*x1max*x1max+b[9]*x2max*x2max+b[10]*x3min*x3max

yy9 = b[0]+b[1]*X1[8]+b[2]*X2[8]+b[3]*X3[8]+b[4]*X12[8]+b[5]*X13[8]+b[6]*X23[8]+b[7]*X123[8]+b[8]*X8[8]+b[9]*X9[8]+b[10]*X10[8]
yy10 = b[0]+b[1]*X1[9]+b[2]*X2[9]+b[3]*X3[9]+b[4]*X12[9]+b[5]*X13[9]+b[6]*X23[9]+b[7]*X123[9]+b[8]*X8[9]+b[9]*X9[9]+b[10]*X10[9]
yy11 = b[0]+b[1]*X1[10]+b[2]*X2[10]+b[3]*X3[10]+b[4]*X12[10]+b[5]*X13[10]+b[6]*X23[10]+b[7]*X123[10]+b[8]*X8[10]+b[9]*X9[10]+b[10]*X10[10]
yy12 = b[0]+b[1]*X1[11]+b[2]*X2[11]+b[3]*X3[11]+b[4]*X12[11]+b[5]*X13[11]+b[6]*X23[11]+b[7]*X123[11]+b[8]*X8[11]+b[9]*X9[11]+b[10]*X10[11]
yy13 = b[0]+b[1]*X1[12]+b[2]*X2[12]+b[3]*X3[12]+b[4]*X12[12]+b[5]*X13[12]+b[6]*X23[12]+b[7]*X123[12]+b[8]*X8[12]+b[9]*X9[12]+b[10]*X10[12]
yy14 = b[0]+b[1]*X1[13]+b[2]*X2[13]+b[3]*X3[13]+b[4]*X12[13]+b[5]*X13[13]+b[6]*X23[13]+b[7]*X123[13]+b[8]*X8[13]+b[9]*X9[13]+b[10]*X10[13]
yy15 = b[0]+b[1]*X1[14]+b[2]*X2[14]+b[3]*X3[14]+b[4]*X12[14]+b[5]*X13[14]+b[6]*X23[14]+b[7]*X123[14]+b[8]*X8[14]+b[9]*X9[14]+b[10]*X10[14]
print("Коефіціенти за критиріем Стьюдента:", t)
print("Коефіціенти t0, t8, t9, t10 є значимими, тому їх не виключаємо з рівняння " )
print("\nКритерій Фішера")
print(d," значимих коефіцієнтів")
f4 = N - d

sad = ((yy1-ySr1)**2+(yy2-ySr2)**2+(yy3-ySr3)**2+(yy4-ySr4)**2+(yy5-ySr5)**2+(yy6-ySr6)**2+(yy7-ySr7)**2+(yy8-ySr8)**2+ (yy9-ySr9)**2+(yy10-ySr10)**2+(yy11-ySr11)**2+(yy12-ySr12)**2+(yy13-ySr13)**2+(yy14-ySr14)**2+(yy15-ySr15)**2)*(m/(N-d))
Fp = sad/sb
#print("d1=", round(d1,2), "d2=", round(d2,2), "d3=", round(d3,2), "d4=", round(d4,2), "d5=", round(sb,2))
print("Fp=", round(Fp,3))
F=[4.2,3.3,2.9,2.7,2.5,2.4,2.1,1.9]
Fi=[1,2,3,4,5,6,12,24]
dif=100
counter=0
for i in range(len(F)):
    if abs(Fi[i]-(i+1))<dif:
        dif= abs(Fi[i]-i)
        counter=i
Ft=F[counter]
#print("Ft берем із таблиці 16 рядяк ",f4," стовпець Ft = ",Ft)
if Fp>Ft:
    print("Fp=",round(Fp,2),">Ft",Ft,"Рівняння неадекватно оригіналу")
else:
    count = count + 1
    print("Fp=",round(Fp,2),"<Ft",Ft,"Рівняння адекватно оригіналу")

