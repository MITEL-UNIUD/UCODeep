from fastai import *
from fastai.tabular import * 
import numpy as np
from zipfile import ZipFile
import datetime
import argparse
import torch
import sys

print("UCOD - recoded version")
parser = argparse.ArgumentParser(description='Train and predict UCOD with recoded set')
parser.add_argument("-t", "--train", dest='train',  nargs='?', required=True, help="path to training and validation set")
parser.add_argument("-e", "--test", dest='test',  nargs='?', required=True, help="path to test set")
parser.add_argument("-a", "--first", dest='l1', default=5000, type=int, help="First layer")
parser.add_argument("-b", "--second", dest='l2', default=5000, type=int, help="Second layer")
parser.add_argument("-c", "--third", dest='l3', default=5000, type=int, help="Third layer")
parser.add_argument("-d", "--fourth", dest='l4', default=5000, type=int, help="Fourth layer")
parser.add_argument("-o", "--output", dest='out', default="output", help="Name of the output file")
args = parser.parse_args()

print(args.train)
print(args.test)
print(args.l1,args.l2,args.l3,args.l4)
basename=str(args.l1)+"-"+str(args.l2)+"-"+str(args.l3)+"-"+str(args.l4)

print("Reading training set...")
print(datetime.datetime.now())

df = pd.read_csv(args.train, na_filter=False, sep='\t',dtype={'c1': str, 'p1': str, 'par1.1': str, 'par1.2': str, 'par1.3': str, 'c2': str, 'p2': str, 'par2.1': str, 'par2.2': str, 'par2.3': str, 'c3': str, 'p3': str, 'par3.1': str, 'par3.2': str, 'par3.3': str, 'c4': str, 'p4': str, 'par4.1': str, 'par4.2': str, 'par4.3': str, 'c5': str, 'p5': str, 'par5.1': str, 'par5.2': str, 'par5.3': str, 'c6': str, 'p6': str, 'par6.1': str, 'par6.2': str, 'par6.3': str, 'c7': str, 'p7': str, 'par7.1': str, 'par7.2': str, 'par7.3': str, 'c8': str, 'p8': str, 'par8.1': str, 'par8.2': str, 'par8.3': str, 'c9': str, 'p9': str, 'par9.1': str, 'par9.2': str, 'par9.3': str, 'c10': str, 'p10': str, 'par10.1': str, 'par10.2': str, 'par10.3': str,                          'c11': str, 'p11': str, 'par11.1': str, 'par11.2': str, 'par11.3': str, 'c12': str, 'p12': str, 'par12.1': str, 'par12.2': str, 'par12.3': str, 'c13': str, 'p13': str, 'par13.1': str, 'par13.2': str, 'par13.3': str, 'c14': str, 'p14': str, 'par14.1': str, 'par14.2': str, 'par14.3': str, 'c15': str, 'p15': str, 'par15.1': str, 'par15.2': str, 'par15.3': str, 'sex': int, 'age': int, 'UCOD': str})

print(df.head())

print("Test set", flush=True)
testdf = pd.read_csv(args.test, na_filter=False, sep='\t',dtype={'c1': str, 'p1': str, 'par1.1': str, 'par1.2': str, 'par1.3': str, 'c2': str, 'p2': str, 'par2.1': str, 'par2.2': str, 'par2.3': str, 'c3': str, 'p3': str, 'par3.1': str, 'par3.2': str, 'par3.3': str, 'c4': str, 'p4': str, 'par4.1': str, 'par4.2': str, 'par4.3': str, 'c5': str, 'p5': str, 'par5.1': str, 'par5.2': str, 'par5.3': str, 'c6': str, 'p6': str, 'par6.1': str, 'par6.2': str, 'par6.3': str, 'c7': str, 'p7': str, 'par7.1': str, 'par7.2': str, 'par7.3': str, 'c8': str, 'p8': str, 'par8.1': str, 'par8.2': str, 'par8.3': str, 'c9': str, 'p9': str, 'par9.1': str, 'par9.2': str, 'par9.3': str, 'c10': str, 'p10': str, 'par10.1': str, 'par10.2': str, 'par10.3': str,                          'c11': str, 'p11': str, 'par11.1': str, 'par11.2': str, 'par11.3': str, 'c12': str, 'p12': str, 'par12.1': str, 'par12.2': str, 'par12.3': str, 'c13': str, 'p13': str, 'par13.1': str, 'par13.2': str, 'par13.3': str, 'c14': str, 'p14': str, 'par14.1': str, 'par14.2': str, 'par14.3': str, 'c15': str, 'p15': str, 'par15.1': str, 'par15.2': str, 'par15.3': str, 'sex': int, 'age': int, 'UCOD': str})
print(testdf.head())

#del df['CYear']
#del testdf['CYear']
test_truth = testdf.pop('UCOD')

print("Setting things...", flush=True)
print(datetime.datetime.now())

procs = [FillMissing, Categorify, Normalize]
valid_idx = range(len(df)-2000000, len(df))
dep_var = 'UCOD'
cat_names = ['sex', 'age', 'c1', 'p1', 'par1.1', 'par1.2', 'par1.3', 'c2', 'p2', 'par2.1', 'par2.2', 'par2.3', 'c3', 'p3', 'par3.1', 'par3.2', 'par3.3', 'c4', 'p4', 'par4.1', 'par4.2', 'par4.3', 'c5', 'p5', 'par5.1', 'par5.2', 'par5.3', 'c6', 'p6', 'par6.1', 'par6.2', 'par6.3', 'c7', 'p7', 'par7.1', 'par7.2', 'par7.3', 'c8', 'p8', 'par8.1', 'par8.2', 'par8.3', 'c9', 'p9', 'par9.1', 'par9.2', 'par9.3', 'c10', 'p10', 'par10.1', 'par10.2', 'par10.3', 'c11', 'p11', 'par11.1', 'par11.2', 'par11.3', 'c12', 'p12', 'par12.1', 'par12.2', 'par12.3', 'c13', 'p13', 'par13.1', 'par13.2', 'par13.3', 'c14', 'p14', 'par14.1', 'par14.2', 'par14.3', 'c15', 'p15', 'par15.1', 'par15.2', 'par15.3']

# input example:
# c1/15 are ICD-10 codes, p1/15 specify the position in the certificate: xy, where x is the line and y is the position of the code in the line. x is in range value from 1 to 6, where from 1 to 5 represent the Part1 of the certificate (5 lines), and 6 represent Part2, single line. Par1, Par2, Par3 are the parents in the ICD-10 classification hierarchy. 
# c1  p1  par1.1  par1.2  par1.3  c2  p2  par2.1  par2.2  par2.3  c3  p3  par3.1  par3.2  par3.3  c4  p4  par4.1  par4.2  par4.3  c5  p5  par5.1  par5.2  par5.3  c6  p6  par6.1  par6.2  par6.3  c7  p7  par7.1  par7.2  par7.3  c8  p8  par8.1  par8.2  par8.3  c9  p9  par9.1  par9.2  par9.3  c10 p10 par10.1 par10.2 par10.3 c11 p11 par11.1 par11.2 par11.3 c12 p12 par12.1 par12.2 par12.3 c13 p13 par13.1 par13.2 par13.3 c14 p14 par14.1 par14.2 par14.3 c15 p15 par15.1 par15.2 par15.3 sex age UCOD
# R13	10	XVIII	R10-R19		J690	20	X	J60-J70	J69	T179	60	XIX	T15-T19	T17	W80	61	XX	V01-X59	W00-X59	J90	62	X	J90-J94		J939	63	X	J90-J94	J93																																														1	39	J690

path='.'
data = TabularDataBunch.from_df(path, df, dep_var, valid_idx=valid_idx, test_df=testdf, procs=procs, cat_names=cat_names, bs=2048)


learn = tabular_learner(data, layers=[args.l1,args.l2,args.l3,args.l4], metrics=accuracy)
print(learn)
#sys.exit("Error message")

learn.model.cuda()

print("LR finder", flush=True)
print(datetime.datetime.now())
learn.lr_find()
learn.recorder.plot(suggestion=True)
min_grad_lr = learn.recorder.min_grad_lr
print(min_grad_lr)
print(min_grad_lr*0.9)

print("Training", flush=True)
print(datetime.datetime.now())
learn.fit_one_cycle(5, min_grad_lr*0.9)

modelname=args.out+".pkl"
learn.export(modelname)

print("Predict", flush=True)
print(datetime.datetime.now())

preds = learn.get_preds(ds_type=DatasetType.Test)

out=args.out+".txt"
f=open(out,'a')
f.write('n,truth,pred1,pred2,pred3,pred4,pred5,prob1,prob2,prob3,prob4,prob5,eq1,eq3,eq5\n')
r = range(0,len(testdf))
equal=0
equal3=0
equal5=0
for count in r:
  f1=0
  f3=0
  f5=0
  truth=test_truth[count]
  ind = np.argpartition(preds[0][count,:], -5)[-5:]
  ind2=ind[np.argsort(preds[0][count,ind])]
  pred1=learn.data.classes[ind2[4]]
  pred2=learn.data.classes[ind2[3]]
  pred3=learn.data.classes[ind2[2]]
  pred4=learn.data.classes[ind2[1]]
  pred5=learn.data.classes[ind2[0]]
  f.write(str(count))
  f.write(',')
  f.write(truth)
  f.write(',')
  f.write(pred1)
  f.write(',')
  f.write(pred2)
  f.write(',')
  f.write(pred3)
  f.write(',')
  f.write(pred4)
  f.write(',')
  f.write(pred5)
  f.write(',')
  f.write(str(preds[0][count,ind2][4].item()))
  f.write(',')
  f.write(str(preds[0][count,ind2][3].item()))
  f.write(',')
  f.write(str(preds[0][count,ind2][2].item()))
  f.write(',')
  f.write(str(preds[0][count,ind2][1].item()))
  f.write(',')
  f.write(str(preds[0][count,ind2][0].item()))
  f.write(',')

  if truth == pred1: 
      equal=equal+1
      f1=1
  if truth==pred1 or truth==pred2 or truth==pred3:
      equal3=equal3+1
      f3=1
  if truth==pred1 or truth==pred2 or truth==pred3 or truth==pred4 or truth==pred5:
      equal5=equal5+1
      f5=1
  f.write(str(f1))
  f.write(',')
  f.write(str(f3))
  f.write(',')
  f.write(str(f5))
  f.write('\n')
  
  if count % 5000 == 0 and count>0:
    acc=equal/count
    acc3=equal3/count
    acc5=equal5/count
    print(count,acc,acc3,acc5)
f.close()
print(acc)
print(acc3)
print(acc5)


print("End")
print(datetime.datetime.now())

