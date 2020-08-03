"""
Main
====

**Summary** Main launch file for the DNA project.

Details
    This contains a launch file for the DNA project.

.. moduleauthor::
    Ryan J. McCarty <rmccarty@uci.edu> <http://ryanjmccarty.com> ORCID: `0000-0002-2417-8917 <http://https://orcid.org/0000-0002-2417-8917>`_

.. topic:: Funding

    This research was made possible by funding from the University of California President's Postdoctoral Fellowship awarded to Ryan J. McCarty, enabling Ryan to perform this work.

.. topic:: Internal Log

    **2020-05-09** Started work on the main launch file

Todo:
    * Everything

"""

import numpy as np
import time
import Import_Data
from sklearn import svm
from scipy.stats import pearsonr
from Feature_generation.Composition import nucleotide_composition, nucleotide_diversity


"""
X = [[0], [1], [2], [3], [2]]
Y = [0, 1, 2, 3, 3]
clf = svm.SVC(decision_function_shape='ovr', kernel='rbf')
clf.fit(X, Y)
print(clf.score(X, Y))
print(clf.predict(X))
exit(9)
dec = clf.decision_function([[1]])
print(dec)

exit(9)
"""

start = time.time()
file_location='E:\Research\Spectra_Analysis\DNA\Datasets\Copp_et_al_2020\All_Data_Copp_with_note.txt'
DNA_strings, color_class, color_sum, peak_counts, peak_lists = Import_Data.parse_data_in_return_fulllist(file_location, splitpeaks=True)
print(start-time.time())

print(type(DNA_strings))
print(type(color_class))

#5 color class
dark_class = color_class.copy()
for index, item in enumerate(dark_class):
    if item == 0:
        dark_class[index] = 0
    else:
        dark_class[index] = 1

#3 color class, dr
drop = True
if drop:
    droplist=[]
    for index, color in enumerate(color_class):
        if color == 0:
            continue
        elif color == 1:
            continue
        elif color == 2:
            droplist.append(index)
        elif color == 3:
            color_class[index] = 2
        elif color == 4:
            color_class[index] = 2
    droplist.reverse()
    for index in droplist:
        DNA_strings = np.delete(DNA_strings, index)
        color_class = np.delete(color_class, index)
        peak_counts = np.delete(peak_counts, index)
        peak_lists = np.delete(peak_lists, index)

#2 color class, drop blacks
drop = True
if drop:
    droplist=[]
    for index, color in enumerate(color_class):
        if color == 0:
            droplist.append(index)
        elif color == 1:
            color_class[index] = 0
        elif color == 2:
            droplist.append(index)
        elif color == 3:
            color_class[index] = 1
        elif color == 4:
            color_class[index] = 1
    droplist.reverse()
    for index in droplist:
        DNA_strings = np.delete(DNA_strings, index)
        color_class = np.delete(color_class, index)
        peak_counts = np.delete(peak_counts, index)
        peak_lists = np.delete(peak_lists, index)

dark_class = color_class.copy()
for index, item in enumerate(dark_class):
    if item == 0:
        dark_class[index] = 0
    else:
        dark_class[index] = 1

firsttime=True
for DNA_sequence in DNA_strings:
    ratio_array, sum_array = nucleotide_composition(DNA_sequence)
    #print(ratio_array)
    
    nucleotide_div = nucleotide_diversity(DNA_sequence, ratio_array)
    #print(ratio_array)
    #exit(9)
    if firsttime:
        firsttime = False
        ratio_array_full = np.asarray([ratio_array])
        sum_array_full = np.asarray([sum_array])
        nuc_array_full = np.asarray([nucleotide_div])
    else:
        ratio_array_full =  np.append(ratio_array_full, np.asarray([ratio_array]), axis=0)
        sum_array_full = np.append(sum_array_full, np.asarray([sum_array]), axis=0)
        nuc_array_full = np.append(nuc_array_full, np.asarray([nucleotide_div]), axis=0)

#color_class = color_class.astype(float)

short_array = ratio_array_full[:,0]
short_array = short_array.astype(float)
#print(pearsonr(short_array, dark_class))
correlation, pvalue = pearsonr(short_array, color_class)
print(correlation, pvalue)

short_array = ratio_array_full[:,1]
short_array = short_array.astype(float)
#print(pearsonr(short_array, dark_class))
correlation, pvalue = pearsonr(short_array, color_class)
print(correlation, pvalue)

print('')
short_array = ratio_array_full[:,2]
short_array = short_array.astype(float)
correlation, pvalue = pearsonr(short_array, color_class)
print(correlation, pvalue)

short_array = ratio_array_full[:,3]
short_array = short_array.astype(float)
correlation, pvalue = pearsonr(short_array, color_class)
print(correlation, pvalue)


exit(9)

clf = svm.SVC(decision_function_shape='ovr', kernel='rbf')
clf.fit(nuc_array_full, color_class)

print(clf.score(nuc_array_full, color_class))

score_matrix = np.zeros((5,5), dtype=float)
predicted_list = clf.predict(nuc_array_full)

#print('Paramaters:')
#print(clf.get_params())
#print('')


for index, predicted in enumerate(predicted_list):
    score_matrix[color_class[index]][predicted] = score_matrix[color_class[index]][predicted] + 1



#
True_Positive = np.zeros((5), dtype=float)
False_Positive = np.zeros((5), dtype=float)
True_Negative = np.zeros((5), dtype=float)
False_Negative = np.zeros((5), dtype=float)
total_matrix_sum = np.sum(score_matrix)

for index in range(0,5, 1):
    TP = score_matrix[index][index]
    True_Positive[index] = TP
    False_Negative[index] = np.sum(score_matrix[index])-TP
    False_Positive[index] = np.sum(score_matrix[:,index])-TP
precision = True_Positive/(True_Positive + False_Positive)
recall = True_Positive/(True_Positive + False_Negative)
Fone = (precision*recall)/(precision+recall)

percent_right = np.sum(True_Positive)/total_matrix_sum

print(precision)
print(recall)
print(Fone)




print('SVM :    ', str((percent_right-0.2)/.80))





summatrix = (np.sum(score_matrix))/25



import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

actual = ['Black','Green','Transition','Red','Very Red']
predicted = ['Black','Green','Transition','Red','Very Red']

fig, ax = plt.subplots()
im = ax.imshow(score_matrix)

# We want to show all ticks...
ax.set_xticks(np.arange(len(predicted)))
ax.set_yticks(np.arange(len(actual)))
# ... and label them with the respective list entries
ax.set_xticklabels(predicted)
ax.set_yticklabels(actual)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(actual)):
    for j in range(len(predicted)):
        text = ax.text(j, i, score_matrix[i, j], ha="center", va="center", color="w")

ax.set_title("Actual and Predicted")
fig.tight_layout()
plt.show()









