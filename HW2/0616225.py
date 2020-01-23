import pandas as pd
import numpy as np
import math
import random
        
class DecisionTree:
    def __init__(self, min_num_of_node = 10, max_depth = 5):
        self.min_num_of_node = min_num_of_node
        self.max_depth = max_depth
        self.root = {}
    
    def Entropy(self, tar):
        s = sum(tar.values())
        g = 0
        for t in tar.values():
            t /= s
            g += -(t * math.log2(t) )
        return g
    def Gini(self, tar):
        s = sum(tar.values())
        g = 0
        for t in tar.values():
            t /= s
            g += t*t
        return (1.0 - g)
            
    def categorical_split(self, attr, X_idx, data_X, target):
        level = {}
        length = {}
        sp = {}
        T = {}
        for dx in X_idx:
                
            #levels
            att = data_X.loc[dx, attr]
            if att == ' ?':
                continue
            if att not in level:
                level[att] = {}
                length[att] = 0
                sp[att] = []
                
            length[att] += 1
            sp[att].append(dx)
            tar = target.loc[dx,'Category']
            if tar not in T:
                T[tar] = 0
            T[tar] += 1
            
            if tar not in level[att]:
                level[att][tar] = 0
            level[att][tar] += 1
        
        HD = self.Entropy(length)
        if HD == 0:
            return -999999, {}
        
        len_sum = sum(length.values())
        for a in length.keys():
            length[a] /= len_sum
            
        rem = 0
        for att in level.keys():
#             part_g = self.Gini(level[att])
            part_g = self.Entropy(level[att])
            rem += ( part_g * length[att] )
        if len(sp.keys())==0:
            print('Categorical only')
            print(sp)
            return -999999, {}
        n = {'attr':attr, 'value':0,'childs':sp}
        
        H = self.Entropy(T)
#         GR = (H - rem) / HD
        GR = H - rem
        return GR, n
    
    def continuous_split(self, attr, X_idx, data_X, target):

        nums = [ data_X.loc[dx, attr] for dx in X_idx ] 
        up , down = max(nums), min(nums)

        leng = up - down
        div = int(leng / 10)
        
        threshold = []
        if div == 0:
            threshold.append(down)
        else:
            threshold = [i for i in range(int(down)+div, int(up), div)]
#             threshold = [(up + down) / 2]    
        max_GR = -999999
        max_sp = {}
        max_value = -1
        T = {}
        for dx in X_idx:
            tar = target.loc[dx, 'Category']
            if tar not in T:
                T[tar] = 0
            T[tar] += 1
        H = self.Entropy(T)
        signal = 0
        for t in threshold:
            level = {'bigger':{}, 'smaller':{}}
            length = {'bigger':0, 'smaller':0}
            sp = {'bigger':[], 'smaller':[]}
            tmp = []
            signal = 0
            for dx in X_idx:
                att = data_X.loc[dx, attr]
                tmp.append(att)
                if att == ' ?':
                    continue
                
                tar = target.loc[dx, 'Category']
                
                if t < att:
                    length['bigger'] += 1
                    if tar not in level['bigger']:
                        level['bigger'][tar] = 0
                    level['bigger'][tar] += 1
                    sp['bigger'].append(dx)
                else:
                    length['smaller'] += 1
                    if tar not in level['smaller']:
                        level['smaller'][tar] = 0
                    level['smaller'][tar] += 1
                    sp['smaller'].append(dx)
                    
            if ( length['bigger'] == 0 or length['smaller'] == 0 ):
                continue
            
            HD = self.Entropy(length)
            
            len_sum = sum(length.values())
            for a in length.keys():
                length[a] /= len_sum
            
            rem = 0
            for att in level.keys():
#                 part_g = self.Gini(level[att])
                part_g = self.Entropy(level[att])

                rem += ( part_g * length[att] )
        
#             GR = (H - rem) / HD
            GR = H - rem
            if GR > max_GR:
                max_GR = GR
                max_sp = sp
                max_value = t
        if len(max_sp) == 0:
            return -999999, {}
        
        n = {'attr':attr, 'value':max_value,'childs':max_sp}
        return max_GR, n
    
    def select_feature(self, X_idx, data_X, target):
        max_gr = -999999
        max_node = {}
        for attr in data_X:
            if attr == 'Id':
                continue
            typ = data_X.loc[:, attr]
            
            if type(typ.iloc[0]) == np.str:
                GR, nod = self.categorical_split(attr, X_idx, data_X, target)
            else:
                GR, nod = self.continuous_split(attr, X_idx, data_X, target)

            if (GR > max_gr):
                max_node = nod
                max_gr = GR
        return max_node

    def terminal(self, nod, X_idx):
        outcome = []
        for idx in X_idx:
            outcome.append(data_Y.loc[idx, 'Category'])
        out_class = max(set(outcome), key=outcome.count)
        nod['class_'] = out_class
        
    def check_terminal(self, nod, X_idx, data_X, data_Y):
        a = data_Y.loc[X_idx, 'Category']
        if len(set(a)) == 1:
            return True
        else:
            return False
        
    def split(self, nod, X_idx, data_X, data_Y):
        #check whether all category are the same
        if self.check_terminal(nod, X_idx, data_X, data_Y):
            self.terminal(nod, X_idx)
            return
        
        #check whether there are sufficient attribute to divide
        if len(data_X.columns) == 1:
            self.terminal(nod, X_idx)
            return

        #pre-pruning
        if len(X_idx) <= self.min_num_of_node:
            self.terminal(nod, X_idx)
            return
        #pre-pruning
        if nod['depth'] >= self.max_depth:
            self.terminal(nod, X_idx)
            return
        #check whether all attribute are the same (entropy=0)
        
        tmp_nod = self.select_feature(X_idx, data_X, data_Y)
        #cannot split (all value in a feature are the same)
        if len(tmp_nod.keys())==0:
            self.terminal(nod, X_idx)
            return
        
        #pop class
        p = []
        for dx in X_idx:
            p.append( data_Y.loc[dx, 'Category'] )
        if p.count(0) == p.count(1):
            pop_class = ""
        else:
            pop_class = max(set(p), key = p.count)
        
        nod['class_'] = pop_class
        nod['attr'] = tmp_nod['attr']
        nod['childs'] = tmp_nod['childs']
        nod['value'] = tmp_nod['value']
        
        if len(nod['childs']) == 0:
            self.terminal(nod, X_idx)
            return
        
        data_X_tmp = data_X.drop(nod['attr'], axis=1)
        
        for att in nod['childs'].keys():
            d = nod['depth']+1
            nod['child_node'][att] = {'attr':"", 'value':0, 'childs':{}, 'child_node':{}, 'depth':d, 'class_':""}
            self.split(nod['child_node'][att], nod['childs'][att], data_X_tmp, data_Y)
        return
            
    def build_tree(self, data_X, X_idx, data_Y):
        if 'Id' in data_X.columns:
            data_X = data_X.drop(columns='Id')
        self.root= {'attr':"", 'value':0, 'childs':{}, 'child_node':{}, 'depth':0, 'class_':""}
        
        self.split(self.root, X_idx, data_X, data_Y)
        return self.root
    
    def post_prune(self, nod, valid_set, X_idx):
        if len(nod['childs']) ==0:
            return
        
        #popular class
        pop_class = nod['class_']
        #iterate child node
        attr = nod['attr']
        childs_error = 0
        seg_idx = {}
        for att in nod['child_node'].keys():
            p_class = nod['child_node'][att]['class_']
            if p_class == "":
                continue
            
            v = nod['value']
            if att=='bigger' or att=='smaller':
                if att == 'bigger':
                    s = valid_set.loc[ valid_set[attr] > v]
                    seg_idx[att] = list(s.index)
                if att == 'smaller':
                    s = valid_set.loc[ valid_set[attr] <= v]
                    seg_idx[att] = list(s.index)
            else:
                #categorical
                s = valid_set.loc[ valid_set[attr] == att ]
                seg_idx[att] = list(s.index)
            
            for row in seg_idx[att]:
                if data_Y.loc[row, 'Category'] != p_class:
                    childs_error += 1
        
        parent_error = 0
        for dx in X_idx:
            if data_Y.loc[dx, 'Category'] != pop_class:
                parent_error += 1
        
        if childs_error >= parent_error:
            #prune
            nod['child_node'] = {}
            nod['childs'] = {}
        else:
            tmp = valid_set.drop(columns = attr)
            for att in nod['child_node'].keys():
                self.post_prune(nod['child_node'][att], tmp, seg_idx[att])
                   
    def predict(self, nod, row, data_Y, print_time):
        if len(nod['child_node']) == 0:
            if print_time > 0:
                print('reach leaf node, stop at level {1} and the predicted value is {0}'.format(nod['class_'], nod['depth']))
                print('The real value is {0}.'.format( data_Y.loc[row['Id'], 'Category'] ))
                print()
            return nod['class_']
        else: 
            sub_attr = row[nod['attr']]
            
            if type(sub_attr) == str:
                if sub_attr not in nod['child_node']:
                    ran_class = random.randint(0, 1)
                    if print_time > 0:
                        print('This attribute is not in {0}, so randomly predicted value is {1}'.format(nod['attr'], ran_class))
                        print()
                    return ran_class
                next_node = nod['child_node'][sub_attr]
                if print_time > 0:
                    print('In level {2} Split by feature {0}, the sample\'s attribute in {0} is {1}'.format(nod['attr'], sub_attr, nod['depth']) )
            else:
                if nod['value'] < sub_attr:
                    next_node = nod['child_node']['bigger']
                    if print_time > 0:
                        print('In level {2} Split by feature {0}, the sample\'s attribute in {0} is larger than threshold {1}'.format(nod['attr'], nod['value'], nod['depth']), end='\n' )

                else:
                    next_node = nod['child_node']['smaller']
                    if print_time > 0:
                        print('In level {2} Split by feature {0}, the sample\'s attribute in {0} is smaller than threshold {1}'.format(nod['attr'], nod['value'], nod['depth']), end='\n' )

            ans = self.predict(next_node,row, data_Y,print_time)
            return ans
        
    def pred(self, test_X, data_Y, print_or_not):
        y = pd.DataFrame(columns=['Id', 'Category'])
        if print_or_not == 1:
            print_time = 10
        else:
            print_time = 0
        for row in test_X.index:
            if print_time > 0:
                print('ID {0}: start prediction '.format(row))
            ans = self.predict( self.root, test_X.loc[row, :], data_Y, print_time )
            print_time -= 1
            y.loc[row] = [test_X.loc[row,'Id'], ans]

        return y
    
    def print_tree(self, nod, data_X, data_Y):
        if len(nod['child_node'])==0:
            return
        for b in range( nod['depth'] ):
                print(' ', end='')
        print('Split by {}'.format(nod['attr']))
        print()
        tmp = data_X.drop(columns=nod['attr'])
        for a in nod['child_node'].keys():
            for b in range( nod['depth'] ):
                print(' ', end='')
            print('attr {}'.format(a))
            print()
            for b in range( nod['depth'] ):
                print(' ', end='')
            print(nod['childs'][a])
            for b in range( nod['depth'] ):
                print(' ', end='')
            print(data_Y.loc[nod['childs'][a], 'Category' ].values)
            print()
        
            self.print_tree(nod['child_node'][a], tmp, data_Y)
        return
    
    def acc(self, y, data_Y):
        correct = 0
        total = 0
        confusion = pd.DataFrame(0, index= ['real 0', 'real 1'] , columns=['predict 0', 'predict 1'])
        
        for a in y.index:
            if y.loc[a]['Category'] == data_Y.loc[a,'Category']:
                correct += 1
                
                if y.loc[a]['Category'] == 1:
                    confusion.iloc[1,1] += 1
                else:
                    confusion.iloc[0, 0] += 1
            else:
                if y.loc[a]['Category'] == 1:
                    confusion.iloc[0, 1] += 1
                else:
                    confusion.iloc[1, 0] += 1
            total += 1
        
        return correct / total, confusion
    
    def sensitivity(self, confusion):
        sensitivity = {}
        sensitivity['0'] = confusion.iloc[0, 0] / sum(confusion.iloc[0, :])
        sensitivity['1'] = confusion.iloc[1, 1] / sum(confusion.iloc[1, :])
        sen = pd.Series(sensitivity)
        return sen
    
    def precision(self, confusion):
        precision = {}
        precision['0'] = confusion.iloc[0, 0] / sum(confusion.iloc[:, 0]) 
        precision['1'] = confusion.iloc[1, 1] / sum(confusion.iloc[:, 1]) 
        pre = pd.Series(precision)
        return pre
    def print_metric(self, confusion, acc, n):
        
        print('accuracy: {0}'.format(acc / n), end='\n\n')
        print('confusion matrix: ')
        print(confusion, end='\n\n')
        print('sensitivity: ')
        print(self.sensitivity(confusion), end='\n\n')
        print('precision: ')    
        print(self.precision(confusion), end='\n\n')
        
    def K_fold(self, data_X, data_Y, n_fold, idx):
        total_idx = [i for i in data_X.index]

        seg = []
        fold_siz = int(data_X.shape[0] / n_fold )
        now = 0
        for i in range(n_fold):
            seg.append(data_X.iloc[now : now + fold_siz,:])
            now += fold_siz

        valid_set = seg[idx]
        train_set = pd.DataFrame(columns=data_X.columns)

        for j in range(len(seg)):
            if j != idx:
                train_set = pd.concat( [seg[j], train_set], ignore_index=False )
        return train_set, valid_set

    
    def Holdout_Validation(self, data_X, ratio):

        fold_siz = int(data_X.shape[0] * ratio)

        train_set = data_X.iloc[:fold_siz, : ]
        valid_set = data_X.iloc[fold_siz:, : ]
        
        return train_set, valid_set
        
class RandomForest:
    def __init__(self, tree_num):
        self.tree_num = tree_num
    def subsample(self, data_X, ratio = 1.0):
        num = int(data_X.shape[0] * ratio)
        s = random.choices( data_X.index, k=num)
        ans = set(s)
        return list(ans)
    
    def sample_feature(self, columns, ratio = 1.0):
        num = int(data_X.shape[1] * ratio)
        c = random.choices( data_X.columns, k=num)
        c = list( set(c) )
        if 'Id' not in c:
            c.append('Id')
        return c
        
    def predict(self, IDs, predictions):
#         rows = predictions[0].index
        rows = len(predictions[0])
        
        ans = pd.DataFrame(columns = ['Id', 'Category'])
        ans['Id'] = IDs
        ansc = []
        for i in range(rows):
            p = [ predictions[idx][i] for idx in range(len(predictions)) ]
            ansp = max(set(p) , key=p.count)
            ansc.append(ansp)
        
        ans['Category'] = ansc
        return ans
        
    def build(self, data_X, data_Y, test_X):
        T = DecisionTree()

        preds = []
        predss = []
        
        #K_fold
        print('===========================================')
        print('Start implementing k_fold:')
        print()

        n_fold = 3
        total_acc = 0
        total_confusion = pd.DataFrame(0, index= ['real 0', 'real 1'] , columns=['predict 0', 'predict 1'])
        
        for idx in range(n_fold):
            
            train_set, test_set = T.K_fold(data_X.iloc[:,:], data_Y, n_fold, idx)
            
            for i in range(self.tree_num):
                sample_X_idx = self.subsample(train_set, 0.7)
    #             sample_feature = self.sample_feature(train_set.columns[1:], 0.7)

                tree = T.build_tree(train_set.loc[:,:], sample_X_idx, data_Y)

                y = T.pred(test_set, data_Y, 0)
                y = y.sort_values(by='Id')
                preds.append(list(y['Category']))
                
            test_set = test_set.sort_values(by='Id')
            IDs = test_set['Id']
            y = self.predict(IDs, preds)
            
            accuracy, confusion = T.acc(y, data_Y)
            
            total_confusion = total_confusion.add(confusion) 
            total_acc += accuracy
        T.print_metric(total_confusion, total_acc, 3)
        
        #Holdout_validation
        preds = []
        print('-------------------------------------------')
        print('Start implementing holdout_validation:')
        print()

        train_set, test_set = T.Holdout_Validation(data_X, 0.7)
        idx = 1
        
        for i in range(self.tree_num):
            sample_X_idx = self.subsample(train_set, 0.7)
#             sample_feature = self.sample_feature(train_set.columns[1:], 0.7)
            tree = T.build_tree(train_set.iloc[:,:], sample_X_idx, data_Y)
            
            y = T.pred(test_set, data_Y, 0)
            y = y.sort_values(by='Id')
            preds.append(list(y['Category']))
            
            y1 = T.pred(test_X, data_Y, 0)
            y1 = y1.sort_values(by='Id')
            predss.append(list(y1['Category']))
            
        test_set = test_set.sort_values(by='Id')
        IDs = test_set['Id']
        y = self.predict(IDs, preds)
        
        accuracy, confusion = T.acc(y, data_Y)
        T.print_metric(confusion, accuracy, 1)
        
        IDs = test_X['Id']
        y = self.predict(IDs, predss)
        y.to_csv("test.csv",index=False, sep=',')
        print('save file finish')
        
data_X = pd.read_csv("data/X_train.csv")
data_Y = pd.read_csv("data/y_train.csv")
test_X = pd.read_csv("data/X_test.csv")

data_X = data_X.sample(frac=1)

#pre-processing
for att in data_X.columns:
    tmp = data_X.loc[ data_X[att] == ' ?']
    if tmp.empty:#dataframe is empty
        continue
    else:
        f = data_X[att].value_counts()
        most = f.index[0]
        data_X.loc[tmp.index, att] = most


R = RandomForest(5)
print('===========================================')
print('Model Random Forest:')
R.build(data_X.iloc[:,:], data_Y, test_X)
# specified = [2, 4, 6, 7, 8, 9, 10, 14]

T = DecisionTree()
print('===========================================')

print('Model Decision Treee:')
#Implement k_fold
print('===========================================')
print('Start implementing k_fold')
print()
n_fold = 3
total_acc = 0
total_confusion = pd.DataFrame(0, index= ['real 0', 'real 1'] , columns=['predict 0', 'predict 1'])

for idx in range(n_fold):
    train_set, test_set = T.K_fold(data_X.iloc[:,:], data_Y, n_fold, idx)
    total_idx = [i for i in train_set.index]
    
    T.build_tree(train_set, total_idx, data_Y)
    
    y = T.pred(test_set, data_Y, 0)

    accuracy, confusion = T.acc(y, data_Y)
    total_confusion = total_confusion.add(confusion) 
    total_acc += accuracy
#     print('{0} fold accuracy: {1}'.format(idx+1, accuracy))
T.print_metric(total_confusion, total_acc, 3)
    
    
#implement Holdout_validation
print('-------------------------------------------')
print('Start implementing Holdout_validation:')
print()
train_set, test_set = T.Holdout_Validation(data_X.iloc[:,:], 0.7)
total_idx = [i for i in train_set.index]

T.build_tree(train_set.iloc[:,:], total_idx, data_Y)
# print("Build finish")

# T.post_prune(T.root, test_set, test_set.index)
# T.print_tree(T.root, train_set.iloc[:,:], data_Y)

# y = T.pred(train_set)
# accuracy = T.acc(y, data_Y)
# print(accuracy)
print('Printing prediction procedure of 10 samples:', end='\n\n')
y = T.pred(test_set, data_Y, 1)
accuracy , confusion = T.acc(y, data_Y)
T.print_metric(confusion, accuracy, 1)
# y = T.pred(test_X)
# y.to_csv("test.csv",index=False, sep=',')
# print('save file finish')

