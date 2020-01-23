import pandas as pd
import numpy as np
import math
import random
import statistics
import matplotlib.pyplot as plt

torch = [6, 4, 10, 2, 9, 2, 2, 2, 12, 2, 4, 4, 9, 9, 1, 4, 3, 5, 9, 6, 7]
cnt2 = {}
nn = ["class", "cap-shape", "cap-surface", "cap-color", "bruises?", "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above", "stalk-surface-below", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat" ]

#divide the data to training set and testing set with ratio 7:3
class Naive_Bayes:
    
    def __init__(self, file, file_name):
        self.file_name = file_name
        self.file = file
        if file_name=="mushroom":
            self.first = 1
        elif file_name=="iris":
            self.first = 0

    def Separate_By_Class(self, train_set, train_label):
        data = {}
     

        for x in range(len(train_set)):
            typ = train_label[x]
            if typ not in data:
                data[typ] = []
            li = []
            
            for info in train_set.iloc[x,:]:
                li.append(info)
                
            data[typ].append(li)
        
        return data

    def summarize_numerical(self, dataset):
        summaries = [(statistics.mean(attr), statistics.stdev(attr)) for attr in zip(*dataset)]
        return summaries
    
    def summarize_categorical(self, instances, classValue):
        global cnt2 
        prob = []
        prob_with_laplace = []
        k = 3
        idx = 0;
        cnt2[classValue] = []
        ans_with_laplace = {}
        for attr in zip(*instances):
            ans = {}
            for at in attr:
                if at not in ans:
                    ans[at] = 0
                ans[at] += 1
            #copy
            for x in ans.keys():
                if self.way == "normal":
                    ans[x] /= float(len(attr))
                elif self.way == "smooth":
                    ans[x] = (ans[x] + k) / float(len(attr) + k * torch[idx])

            prob.append(ans)            
            cnt2[classValue].append((len(attr)))
            idx += 1;

        return prob
    def summarizeByClass(self, train_set, train_label):
        sep = self.Separate_By_Class(train_set, train_label)
        summ = {}
        for classValue, instances in sep.items():
    #         print(instances)
            if self.file_name == "iris":
                summ[classValue] = self.summarize_numerical(instances)
            elif self.file_name == "mushroom":
                summ[classValue] = self.summarize_categorical(instances, classValue)
        return summ


    def Gaussian_Probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean, 2) / (2*math.pow(stdev, 2))))
        a = ((1 / (math.sqrt(2*math.pi) * stdev)) * exponent)
       
        return ((1 / (math.sqrt(2*math.pi) * stdev)) * exponent)

    def cal_numerical_probability(self, summaries, features):
        prob = {}
        #print(features)
        for classValue, classSummaries in summaries.items():
            prob[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = features[i]
                probability = self.Gaussian_Probability(x, mean, stdev)
                if probability > 0:
                    prob[classValue] += math.log(probability)
#                 else:
#                     prob[classValue] -= 99999
#                 prob[classValue] *= (probability)

        return prob  

    def cal_categorical_probability(self, summ, features):
        prob = {}
        total = len(summ['p']) + len(summ['e'])
        for classValue in summ.keys():
            prob[classValue] = 0
            for j in range(len(features)):
                x = features[j]
                if x not in summ[classValue][j]:
                    if self.way == "smooth":
                        prob[classValue] += math.log( (3 / float(cnt2[classValue][j] + 3 * (torch[j]))) )
#                     prob[classValue] = -9999999;
                else:
#                     prob[classValue] *= (summ[classValue][j][x])
                    prob[classValue] += math.log((summ[classValue][j][x]))

    #         prob[classValue] = probability
            prob[classValue] += math.log(len(summ[classValue]) / float(total))
#             prob[classValue] *= len(summ[classValue]) / float(total)

        return prob

    def predict(self, summaries, features):
        if self.file_name == "iris":
            prob = self.cal_numerical_probability(summaries, features)
        elif self.file_name == "mushroom":
            prob = self.cal_categorical_probability(summaries, features)

        max_prob = -10000000
        best_label = None
        for classValue, probability in prob.items():
    #         print(probability)
            if probability > max_prob:
                max_prob = probability
                best_label = classValue
        return best_label          

    def get_Prediction(self, summaries, testSet):
        pred = []
        for i in range(len(testSet)):
            result = self.predict(summaries, list(testSet.iloc[i, :]))
            pred.append(result)
        return pred
    
    def execute(self, train_set, train_label, test_set, test_label):
        global cnt2
        cnt2 = {}
        summaries = self.summarizeByClass(train_set, train_label)
        predictions = self.get_Prediction(summaries, test_set)

        return predictions

    def k_fold(self, k, idx):
        self.file = self.file.reindex(np.random.permutation(self.file.index))

        seg_size = int(len(self.file) / k)

        set1 = self.file[:seg_size] 
        set2 = self.file[seg_size : 2*seg_size]
        set3 = self.file[2*seg_size:]
        set_ = [set1, set2, set3]

        train_set = pd.DataFrame()
        #allocate the trainset and testset
        for i in range(3):
            if i==idx:
                test_set = set_[i]
            else:
                train_set = train_set.append(set_[i])

        train_label = list(train_set.loc[:, "class"])
        train_set = train_set.drop("class", axis=1)

        test_label = list(test_set.loc[:, "class"])
        test_set = test_set.drop("class", axis=1)

        return [train_set, train_label, test_set, test_label]
    
    def Holdout_Handle(self, ratio):
        self.file = self.file.reindex(np.random.permutation(self.file.index))
#         self.file = self.file.reindex(np.random.permutation(self.file.index))

        train_size = int(len(self.file) * ratio)

        train_set = self.file[:train_size]
        train_label = list(train_set.loc[:, "class"])
        train_set = train_set.drop("class", axis=1)

        test_set = self.file[train_size:]
        test_label = list(test_set.loc[:, "class"])
        test_set = test_set.drop("class", axis=1)

        return [train_set, train_label, test_set, test_label]        
        
    def Holdout_validation(self, ratio, way = "normal"):
        self.way = way
        train_set, train_label, test_set, test_label = self.Holdout_Handle(ratio)

        pred = self.execute(train_set, train_label, test_set, test_label)
        acc = self.cal_result(test_set, test_label, pred)
        if(self.way == 'normal'):
            print('Data {0} Accuracy by Holdout_validation: {1}% \n'.format(self.file_name, acc))
        elif(self.way == 'smooth'):
            print('Data {0} Accuracy by Holdout_validation with laplace smooth: {1}% \n'.format(self.file_name, acc))
    def k_fold_validation(self, k, way = "normal"):
        self.way = way
        acc = 0.0
        acc_list = []
        idx = k
        index = self.first
        while idx:
            idx-=1
            train_set, train_label, test_set, test_label = self.k_fold(k, idx)

            pred = self.execute(train_set, train_label, test_set, test_label)
            acc += self.cal_result(test_set, test_label, pred)
        acc /= float(k)
        if(self.way == 'normal'):
            print('Data {0} Average Accuracy by K-fold validation: {1}% \n'.format(self.file_name, acc))
        elif(self.way=='smooth'):
            print('Data {0} Average Accuracy by K-fold validation with laplace smooth: {1}% \n'.format(self.file_name, acc))

        idx = k

    def cal_result(self, test_set, test_label, prediction):
        #calculate confusion matrix, accuracy, precision, recall
        correct = 0
        label = {}
        precision = {}
        for i in range(len(test_set)):
            T = test_label[i]
            P = prediction[i]
            if T not in label:
                label[T] = {}
            if P not in label[T]:
                label[T][P] = 0
            label[T][P] += 1
            
            if T == P:
                correct += 1
        print('Confusion Matrix parameters: ')
        print(label)
        incorrect = len(test_set) - correct;
       
       
            
        acc = (correct / float(len(test_set))) * 100.0
        return acc

    def plot_feature_mushroom(self):
        #value frequency
       
        for x in self.file:
            count_feat = {}

            for feat in self.file.loc[:,x]:
                if feat not in count_feat:
                    count_feat[feat] = 0
                count_feat[feat] += 1
            plt.title(x)
            plt.bar(count_feat.keys(), count_feat.values(), color='brown')
            for x, y in enumerate(count_feat.values()):
                plt.text(x, y, '%s' %y)
            plt.show()


    def plot_feature_iris(self):
        #value frequency with binning
        means = {}
        stdevs = {}
        for x in self.file:
            if x=="class":
                continue
            count_feat = {}
            for feat in self.file.loc[:, x]:
                if feat not in count_feat:
                    count_feat[feat] = 0
                count_feat[feat] += 1
            means[x] = statistics.mean(self.file.loc[:, x])
            stdevs[x] = statistics.stdev(self.file.loc[:, x])
            #show value frequency
            
            plt.title(x + " value frequency")
            count, binnn, patch = plt.hist(list(self.file.loc[:, x]), facecolor='red',edgecolor='black')

            dis = (binnn[1]-binnn[0])/2
            for (x, y) in zip(binnn, count):
                plt.text(x+dis, y, '%s' %y, ha='center')
            plt.xticks(binnn)
            plt.xlabel("cm")
            plt.ylabel("num")
            plt.show()
            
        
        #show means
        
        plt.title("feature's" + " mean")
        
        plt.bar(means.keys(), means.values(), color='green')
        for x, y in enumerate( means.values() ):
            plt.text(x, y+0.1, '%.5s' %y, ha='left')
        plt.show()
        #show standard deviation
        plt.title("feature's" + " stdev")
        plt.bar(stdevs.keys(), stdevs.values(), color='aquamarine')
        for x, y in enumerate( stdevs.values() ):
            plt.text(x, y, '%.5s' %y, ha='left')
        plt.show()
    

if __name__ == '__main__':
    #read file
   
    mushroom_file = pd.read_csv("data/mushroom.data.csv", header=None, names=nn) #categorical data
    mushroom_file = mushroom_file.dropna()

    #missing value
    mushroom_file = mushroom_file.drop("stalk-root", axis=1)      

    iris_file = pd.read_csv("data/iris.data.csv", header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])  #numerical data

    A = Naive_Bayes(iris_file, "iris")
    A.Holdout_validation(0.7)
    A.k_fold_validation(3)
    A.plot_feature_iris()

    A = Naive_Bayes(mushroom_file, "mushroom")
    A.Holdout_validation(0.7)
    A.Holdout_validation(0.7, "smooth")
    A.k_fold_validation(3)
    A.k_fold_validation(3, "smooth")
    A.plot_feature_mushroom()

    


    
