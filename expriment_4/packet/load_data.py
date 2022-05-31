from csv import reader
import random
# load the csv data
def load_the_csv(filename):
    dataset =[]
    labelset=[]
    with open(filename,'r') as file:
        csv_reader =reader(file)
        for i,row in enumerate(csv_reader):
            if not row :
                continue
                # check if the row is empty ,like []
            dataset.append(row[0:-1])
            # dataset.append(i)
            labelset.append(row[-1])
            # labelset.append(i)
        temp_random = list(zip(dataset, labelset))
        random.shuffle(temp_random)
        dataset, labelset = zip(*temp_random)
        dataset=list(dataset)
        labelset=list(labelset)
    return dataset,labelset

# convert the string to float
def conv_string_to_float(dataset,column) :
    for row in dataset:
        row[column]=float(row[column])

def conv_string_map_integer(label_set):
    type_set=set(label_set)
    type_dic=dict()

    for i,item in enumerate(type_set):
        type_dic[item]=i
    for i in range(len(label_set)):
        label_set[i]=type_dic[label_set[i]]
    return type_dic


def normalize(dataset):
    minmax = []
    for i in range(len(dataset[0]) - 1):
        item_list = [row[i] for row in dataset]
        item_minmax = ([min(item_list), max(item_list)])
        minmax.append(item_minmax)
    for row in dataset:
        for i in range(len(dataset[0])-1):
            row[i]=(row[i]-minmax[i][0])/(minmax[i][1]-minmax[i][0])