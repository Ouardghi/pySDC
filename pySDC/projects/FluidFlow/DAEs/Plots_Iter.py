import matplotlib.pyplot as plt
import numpy as np

import dolfin as df


df.info(df.LinearVariationalSolver.default_parameters(), True)

df.list_linear_solver_methods()
df.list_krylov_solver_preconditioners()

exit()


def addlabels4(x, y):
    for i in range(len(x)):
        plt.text(i - 0.0 * 0.15, 3, y[i], ha='center', rotation=90)


def addlabels5(x, y):
    for i in range(len(x)):
        plt.text(i + 1.0 * 0.15, 3, y[i], ha='center', rotation=90)


T1 = (15.5, 50, 50, 0, 24.7)
T2 = (15.5, 50, 50, 0, 22.7)
T3 = (19.8, 50, 50, 0, 28)
T4 = (10.9, 50, 50, 0, 15.6)


IT1 = (15.5, 15.5, 19.8, 10.9)
IT2 = (50, 50, 50, 50)
IT3 = (50, 50, 50, 50)
IT4 = (0, 0, 0, 0)
IT5 = (24.7, 22.7, 28, 15.6)

last_week_cups = (20, 35, 30, 35, 27)
this_week_cups = (25, 32, 34, 20, 25)
# names = ['LU', 'IEpar', 'MIN', 'MIN_SR_NS', 'MIN_SR_S']
Times = [2, 4, 5.49, 7.2]


ToIt1 = 14.98
ToIt2 = 50
ToIT3 = 50
ToIt4 = 22.38

fig = plt.figure(figsize=(6.95, 5), dpi=200)
left, bottom, width, height = 0.1, 0.3, 0.8, 0.6
ax = fig.add_axes([left, bottom, width, height])

width = 0.15
ticks = np.arange(len(Times))
# ax.bar(ticks, last_week_cups, width, label='Last week')
ax.bar(ticks - 0.0 * width, IT1, width, align="center", label='LU')
# ax.bar(ticks - 0.5*width, IT2, width, align="center",label='IEpar')
# ax.bar(ticks + 0.5*width, IT3, width, align="center",label='MIN')
# ax.bar(ticks + 1.5*width, IT4, width, align="center",label='MIN_SR_NS')
ax.bar(ticks + 1.0 * width, IT5, width, align="center", label='MIN_SR_S')

addlabels4(ticks, IT1)
addlabels5(ticks, IT5)

ax.set_ylabel('# Iterations')
ax.set_xlabel('Times')
# ax.set_title('Iterations')
ax.set_xticks(ticks + width / 2)
ax.set_xticklabels(Times)

ax.set_ylim(0, 60)

ax.legend(ncol=5, loc='best')
plt.savefig('Num_of_Iter', bbox_inches='tight')


def addlabels1(x, y):
    for i in range(len(x)):
        plt.text(i - 0.5 * 0.15, 3, y[i], ha='center', rotation=90)


def addlabels2(x, y):
    for i in range(len(x)):
        plt.text(i + 0.5 * 0.15, 3, y[i], ha='center', rotation=90)


def addlabels3(x, y):
    for i in range(len(x)):
        plt.text(i + 1.5 * 0.15, 3, y[i], ha='center', rotation=90)


def addlabels4(x, y):
    for i in range(1):
        plt.text(i - 0.0 * 0.15, 3, y[i], ha='center', rotation=90)


def addlabels5(x, y):
    for i in range(1):
        plt.text(i + 1.0 * 0.15, 3, y[i], ha='center', rotation=90)


"""        
def addlabels6(x,y):
    for i in range(1):
        plt.text(i+1+1.5*0.15, 3, y[i], ha = 'center', rotation = 90)
"""
ToIt1 = (6.23, 0)
ToIt2 = (8.19, 0)
ToIT3 = (43.07, 0)
ToIt4 = (0, 0)
ToIt5 = (7.67, 0)


IT1 = (6, 7, 6, 6)
IT2 = (8.8, 7, 7.9, 8)
IT3 = (46.6, 41.7, 44.1, 43.3)
IT4 = (0, 0, 0, 0)
IT5 = (8, 8, 8, 6.4)

Times = [2, 4, 5.49, 7.2]

fig = plt.figure(figsize=(6.95, 5), dpi=200)
left, bottom, width, height = 0.1, 0.3, 0.8, 0.6
ax = fig.add_axes([left, bottom, width, height])

width = 0.15
ticks = np.arange(len(Times))
ax.bar(ticks - 0.5 * width, IT1, width, align="center", label='LU')
ax.bar(ticks + 0.5 * width, IT2, width, align="center", label='IEpar')
# ax.bar(ticks + 0.5*width, IT3, width, align="center",label='MIN')
# ax.bar(ticks + 1.5*width, IT4, width, align="center",label='MIN_SR_NS')
ax.bar(ticks + 1.5 * width, IT5, width, align="center", label='MIN_SR_S')

addlabels1(ticks, IT1)
addlabels2(ticks, IT2)
addlabels3(ticks, IT5)

ax.set_ylabel('# Iterations')
ax.set_xlabel('Times')
# ax.set_title('Iterations')
ax.set_xticks(ticks + width / 2)
ax.set_xticklabels(Times)

ax.set_ylim(0, 12)

ax.legend(ncol=5, loc='best')
plt.savefig('Num_of_Iter2', bbox_inches='tight')


"""
fig = plt.figure(figsize=(6.95,5), dpi=200)
left, bottom, width, height = 0.1, 0.3, 0.8, 0.6
ax = fig.add_axes([left, bottom, width, height]) 

width = 0.15   
ticks = 1    
ax.bar(ticks - 0.5*width, ToIt1, width, align="center",label='LU')
ax.bar(ticks + 0.5*width, ToIt2, width, align="center",label='IEpar')
#ax.bar(ticks + 0.5*width, ToIt3, width, align="center",label='MIN')
#ax.bar(ticks + 1.5*width, ToIt4, width, align="center",label='MIN_SR_NS')
ax.bar(ticks + 1.5*width, ToIt5, width, align="center",label='MIN_SR_S')

addlabels4(ticks, ToIt1)
addlabels5(ticks, ToIt2)
addlabels6(ticks, ToIt5)

ax.set_xlim(0,2)
ax.set_ylim(0,12)
ax.set_ylabel('# Iterations')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    
ax.legend(ncol=5, loc='best')
plt.savefig('Num_of_Iter_2',bbox_inches='tight')
"""
plt.show()
