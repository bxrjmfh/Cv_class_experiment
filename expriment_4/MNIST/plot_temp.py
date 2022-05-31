import matplotlib.pyplot as plt
import pickle

def Draw_each_time(acc,val_acc,name):
    numbers = range(1,len(acc)+1)
    ave_acc = sum(acc)/len(acc)
    ave_val_acc = sum(val_acc)/len(val_acc)
    plt.title(name+"_acc")
    plt.plot(numbers, acc, 'bo', label="train_acc")
    plt.plot(numbers, val_acc, 'go', label="validation_acc")
    plt.axhline(y=ave_acc,c='b',label="ave_train_acc_{:.3f}".format(ave_acc))
    plt.axhline(y=ave_val_acc,c='g',label="ave_val_acc_{:.3f}".format(ave_val_acc))
    plt.xlabel("numbers")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig(name)
    plt.show()

with open('res_2_3.pkl', 'rb') as infile:
    result = pickle.load(infile)

[acc_300_10,val_accs_300_10,accs_300_100_10,val_accs_300_100_10] = result
Draw_each_time(acc_300_10,val_accs_300_10,'300x10')
Draw_each_time(accs_300_100_10,val_accs_300_100_10,'300x100x10')
