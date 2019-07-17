#!/usr/bin/env python3

from numpy import genfromtxt
import numpy as np
import argparse
import utils
import csv
import matplotlib.pyplot as plt
import pandas as pd
#import numpy as np

# Open the desired file for reading

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")

args = parser.parse_args()

model_dir = utils.get_model_dir(args.model)
print(model_dir)
file_name =model_dir+"/log.csv"
print(file_name)

# data = np.loadtxt(file_name, delimiter=',', dtype=str)
# print(data)


# for i in range(1,data.shape[1]):
    # plt.plot(data[:,0], data[:,i], label='id %s' %i)

# plt.legend()
# plt.show()

df = pd.read_csv(file_name, delimiter=',', names = ['update', 'frames', 'rreturn_mean', 'rreturn_std', 'entropy', 'return_mean'])
sub_df = df[['rreturn_mean','rreturn_std']]
# print(sub_df)
# print(df[['frames']])
# idx_array = np.array(list(df['update']),dtype=float)
idx_array = np.arange(len(list(df['frames'])))
print(idx_array)
# idx_array = np.array(list(df['frames']))
rr_array = np.array(list(df['return_mean']))
# rr_array = np.array(list(df['rreturn_mean']))
# print(df.columns[5])
# print(len(df.columns[5,:]))


# plt.scatter(idx_array, rr_array)
plt.plot(idx_array, rr_array)


ax = plt.axes()
ax.yaxis.set_major_locator(plt.NullLocator())
ax.xaxis.set_major_formatter(plt.NullFormatter())

plt.xlabel('# of frames')
plt.ylabel('reward_mean')
plt.show()
# pd.tseries.plotting.pylab.show()




#  = [row[0] for row in data]
# b = [row[1] for row in data]
# c = [row[2] for row in data]


# fig = plt.figure()
# ax = fig.add_subplot(111, axisbg = 'w')
# ax.plot(a,b,'g',lw=1.3)
# ax.plot(a,c,'r',lw=1.3)
# plt.show()
