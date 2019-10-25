import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import csv
import math
import os


path_list = ["results/rs/input/adaptive/", "results/rs/input/fixed/",
             "results/ss/input/adaptive/", "results/ss/input/fixed/"]

save_to = ["rs/adaptive/", "rs/fixed/",
           "ss/adaptive/", "ss/fixed/"]

models = ["25","45","99"]

for i in range(len(path_list)):
    for j in range(1,4):
        if j == 3:
            weight_files = 10
        else:
            weight_files = 4
        print("=========>",j)
        print(path_list[i]+models[j-1])

        if j == 1:
            upper = 2
        else:
            upper = 4

        for jj in range(1,upper):
            freq = []
            best_labels = []
            for k in range(weight_files):

                dataset_file = "avg_indices_reflectance_1617.csv"
                dataset = pd.read_csv(dataset_file, header = 0)
                dataset_labels = list(dataset.columns.values)

                lab25 = dataset_labels[8:47]
                lab31 = dataset_labels[47:86]
                lab45 = dataset_labels[86:125]


                target_file = "Rice_AgronomicTraits_1617.csv"
                target = pd.read_csv(target_file, header = 0)
                target_labels = list(target.columns.values)
                target25 = target_labels[8:12]
                target45 = target_labels[16:20]
                target99 = target_labels[20:len(target_labels)]

                targets = [target25]
                targets.append(target45)
                targets.append(target99)

                sample_file = path_list[i]+models[j-1]+"/Source_Weights_model"+str(jj)+"_"+str(k)+".csv"
                sample = pd.read_csv(sample_file, header = None)
                sample_data = sample._get_numeric_data()
                sample_np = sample_data[0:len(sample_data)].as_matrix()
                sample_np = np.asarray(sample_np)

                if sample_np.shape[0] == 80:
                    temp1 = sample_np[0:40,:]
                    temp2 = sample_np[40:80,:]
                    sample_np = np.concatenate((temp1,temp2),axis = 1)
                elif sample_np.shape[0] == 120:
                    temp1 = sample_np[0:40,:]
                    temp2 = sample_np[40:80,:]
                    temp3 = sample_np[80:120,:]
                    sample_np = np.concatenate((temp1,temp2,temp3),axis = 1)

                weights = sample_np
                """
                indexes = np.zeros((weights.shape))
                threshold = 0.95
                for idx1 in range(weights.shape[0]):
                    sum = 0
                    temp1 = weights[idx1,:]
                    temp2 = sorted(temp1, reverse = True)
                    idx2 = 0
                    while sum < 0.95:
                        sum = sum + temp2[idx2]
                        idx2 = idx2 + 1
                    for ll in range(idx2):
                        index = np.where(temp1 == temp2[ll])
                        for lll in range(index[0].shape[0]):
                            indexes[idx1,index[lll]]  = 1

                for idx1 in range(weights.shape[0]):
                    for idx2 in range(weights.shape[1]):
                        if indexes[idx1,idx2] == 0:
                            weights[idx1,idx2] = 0
                frequency = np.zeros(weights.shape[1])
                for idx1 in range(len(frequency)):
                    frequency[idx1] = np.sum(indexes[:,idx1])
                """
                frequency = np.zeros(weights.shape[1])
                for col in range(weights.shape[1]):
                    sum = 0
                    for row in range(weights.shape[0]):
                        sum = sum + weights[row,col]

                    frequency[col] = sum

                frequency = frequency/40

                for ll in range(frequency.shape[0]):
                    frequency[ll] = round(frequency[ll],4)

                data = weights
                with open('weights/'+save_to[i]+models[j-1] + '/New_weights'+str(jj)+'_'+ str(k) + '.csv', 'w+') as csvFile:
                    writer = csv.writer(csvFile,delimiter=',')
                    writer.writerows(data)
                    writer.writerow(frequency)
                csvFile.close()

                f1 = np.zeros(weights.shape[1])
                f2 = np.zeros(weights.shape[1])
                thres = 0.75
                error = 1e-2

                indexes = []
                for ll in range(frequency.shape[0]):
                    if frequency[ll]>thres-error:
                        f1[ll] = frequency[ll]
                        indexes.append(ll)
                    else:
                        f2[ll] = frequency[ll]

                labels = []
                if jj == 1:
                    labels = lab25
                    labels.append('Whole_set')
                    plt.bar(range(0,len(f1)),f1, tick_label = labels, color = "green")
                    plt.bar(range(0,len(f2)),f2, tick_label = labels, color = "#CC0000")
                    plt.xticks(rotation = 90, fontsize = 5)
                    plt.ylabel("Total contribution in current output")
                    plt.axhline(thres-error, color="grey", linestyle = "--", label = 'threshold = '+str(thres)+'\n(error: '+str(error)+')')
                    plt.legend(loc = 0, fontsize = "xx-small")
                    plt.title(targets[j-1][k])
                    plt.savefig('weights/'+save_to[i]+models[j-1] + '/Frequency'+str(jj)+'_'+ str(k)+'.png', bbox_inches='tight')
                    plt.clf()

                elif jj == 2:
                    lab25.append('Whole_set')
                    lab45.append('Whole_set')
                    labels = np.concatenate((lab25,lab45), axis = 0)
                    labels = labels.tolist()
                    labels.append('Whole_set')

                    plt.bar(range(0,40),f1[0:40], tick_label = labels[0:40], color = "green")
                    plt.bar(range(0,40),f2[0:40], tick_label = labels[0:40], color = "#CC0000")
                    plt.ylabel("Total contribution in current output")
                    plt.axhline(thres-error, color="grey", linestyle = "--", label = 'threshold = '+str(thres)+'\n(error: '+str(error)+')')
                    plt.legend(loc = 0, fontsize = "xx-small")
                    plt.xticks(rotation = 90, fontsize = 5)
                    plt.title(targets[j-1][k]+"_a")
                    plt.savefig('weights/'+save_to[i]+models[j-1] + '/Frequency'+str(jj)+'_'+ str(k)+'a.png', bbox_inches='tight')
                    plt.clf()

                    plt.bar(range(40,80),f1[40:80], tick_label = labels[40:80], color = "green")
                    plt.bar(range(40,80),f2[40:80], tick_label = labels[40:80], color = "#CC0000")
                    plt.ylabel("Total contribution in current output")
                    plt.axhline(thres-error, color="grey", linestyle = "--", label = 'threshold = '+str(thres)+'\n(error: '+str(error)+')')
                    plt.legend(loc = 0, fontsize = "xx-small")
                    plt.xticks(rotation = 90, fontsize = 5)
                    plt.title(targets[j-1][k]+"_b")
                    plt.savefig('weights/'+save_to[i]+models[j-1] + '/Frequency'+str(jj)+'_'+ str(k)+'b.png', bbox_inches='tight')
                    plt.clf()

                elif jj == 3:
                    lab25.append('Whole_set')
                    lab31.append('Whole_set')
                    lab45.append('Whole_set')
                    labels = np.concatenate((lab25,lab31,lab45), axis = 0)
                    labels = labels.tolist()
                    labels.append('Whole_set')

                    plt.bar(range(0,40),f1[0:40], tick_label = labels[0:40], color = "green")
                    plt.bar(range(0,40),f2[0:40], tick_label = labels[0:40], color = "#CC0000")
                    plt.ylabel("Total contribution in current output")
                    plt.axhline(thres-error, color="grey", linestyle = "--", label = 'threshold = '+str(thres)+'\n(error: '+str(error)+')')
                    plt.legend(loc = 0, fontsize = "xx-small")
                    plt.xticks(rotation = 90, fontsize = 5)
                    plt.title(targets[j-1][k]+"_a")
                    plt.savefig('weights/'+save_to[i]+models[j-1] + '/Frequency'+str(jj)+'_'+ str(k)+'a.png', bbox_inches='tight')
                    plt.clf()

                    plt.bar(range(40,80),f1[40:80], tick_label = labels[40:80], color = "green")
                    plt.bar(range(40,80),f2[40:80], tick_label = labels[40:80], color = "#CC0000")
                    plt.ylabel("Total contribution in current output")
                    plt.axhline(thres-error, color="grey", linestyle = "--", label = 'threshold = '+str(thres)+'\n(error: '+str(error)+')')
                    plt.legend(loc = 0, fontsize = "xx-small")
                    plt.xticks(rotation = 90, fontsize = 5)
                    plt.title(targets[j-1][k]+"_b")
                    plt.savefig('weights/'+save_to[i]+models[j-1] + '/Frequency'+str(jj)+'_'+ str(k)+'b.png', bbox_inches='tight')
                    plt.clf()

                    plt.bar(range(80,120),f1[80:120], tick_label = labels[80:120], color = "green")
                    plt.bar(range(80,120),f2[80:120], tick_label = labels[80:120], color = "#CC0000")
                    plt.ylabel("Total contribution in current output")
                    plt.axhline(thres-error, color="grey", linestyle = "--", label = 'threshold = '+str(thres)+'\n(error: '+str(error)+')')
                    plt.legend(loc = 0, fontsize = "xx-small")
                    plt.xticks(rotation = 90, fontsize = 5)
                    plt.title(targets[j-1][k]+"_c")
                    plt.savefig('weights/'+save_to[i]+models[j-1] + '/Frequency'+str(jj)+'_'+ str(k)+'c.png', bbox_inches='tight')
                    plt.clf()
                temp = []
                for idx in indexes:
                    temp.append(labels[idx])
                best_labels.append([temp])
                freq.append(frequency)
            data = freq
            with open('weights/'+save_to[i] + '/Frequency_'+models[j-1]+'_'+str(jj)+'.csv', 'w+') as csvFile:
                writer = csv.writer(csvFile,delimiter=',')
                writer.writerow(labels)
                writer.writerows(data)
            csvFile.close()

            best_labels = np.asarray(best_labels)
            best_labels.reshape(-1,1)
            data = best_labels
            with open('weights/'+save_to[i] + '/Most_important_'+models[j-1]+'_'+str(jj)+'.csv', 'w+') as csvFile:
                writer = csv.writer(csvFile,delimiter=',')
                writer.writerows(data)
            csvFile.close()
