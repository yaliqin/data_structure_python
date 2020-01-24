

import random

import numpy as np

uniform_values = [int(random.uniform(0,25))
                  for _ in range(1000)]       #1000 random values in range 0-25



uni_array = np.array(uniform_values)
values,frequencies = np.unique(uni_array,return_counts = True)


import seaborn as sns
sns.set_style('whitegrid')
axes = sns.barplot(x = values, y = frequencies, color = 'gray')
axes.set(xlabel = 'Values', ylabel = 'Frequency')
axes.set_title('Uniformly distributed random values')

uniform_values = [int(random.uniform(0,25))
                  for _ in range(10000)]       #10k random values in range 0-25
values_2,frequencies_2 = np.unique(uni_array,return_counts = True)
sns.set_style('whitegrid')
axes = sns.barplot(x = values_2, y = frequencies_2, color = 'gray')
axes.set(xlabel = 'Values', ylabel = 'Frequency')
axes.set_title('Uniformly distributed random values with 10k samples')

### Expand data value range to 0~100

uniform_values = [int(random.uniform(0,100))
                  for _ in range(10000)]       #10k random values in range 0-25
values_3,frequencies_3 = np.unique(uni_array,return_counts = True)

sns.set_style('whitegrid')
axes = sns.barplot(x = values_2, y = frequencies_2, color = 'gray')
axes.set(xlabel = 'Values', ylabel = 'Frequency')
axes.set_title('Uniformly distributed random values with 10k samples and big range')

import matplotlib.pyplot as plt

fig, axs = plt.subplots(ncols=3)
sns.regplot(x='1k samples value', y='frequency', ax=values, data=frequencies)
sns.regplot(x='10k samples value', y='frequency', ax=values_2, data=frequencies_2)
sns.boxplot(x='10k samples& 100 range value', y='frequency', ax=values_3, data=frequencies_3)

