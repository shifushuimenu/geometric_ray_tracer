import numpy as np
import matplotlib.pyplot as plt 


num_surfs = 10

num_aberrations = 5
barWidth = 1.0/(num_aberrations+1)
fig, axs = plt.subplots(figsize =(12, 8)) 

spherical = np.random.randint(-10,10, size=num_surfs+1)
coma = np.random.randint(-10,10, size=num_surfs+1) 
astigmatism = np.random.randint(-10,10, size=num_surfs+1)
field_curvature = np.random.randint(-10,10, size=num_surfs+1)
distortion = np.random.randint(-10,10, size=num_surfs+1)

br1 = np.arange(len(spherical)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
br4 = [x + barWidth for x in br3] 
br5 = [x + barWidth for x in br4] 
br_sep = [x + barWidth for x in br5] 

fig.axes[0].bar(br1, spherical, color ='red', width = barWidth, 
        edgecolor ='grey', label ='spherial') 
fig.axes[0].bar(br2, coma, color ='green', width = barWidth, 
        edgecolor ='grey', label ='coma') 
fig.axes[0].bar(br3, astigmatism, color ='magenta', width = barWidth, 
        edgecolor ='grey', label ='astigmatism') 
fig.axes[0].bar(br4, field_curvature, color ='cyan', width = barWidth, 
        edgecolor ='grey', label ='field curvature') 
fig.axes[0].bar(br5, distortion, color ='yellow', width = barWidth, 
        edgecolor ='grey', label ='distortion') 
ymin, ymax = fig.axes[0].get_ylim()
fig.axes[0].vlines(br_sep, ymin, ymax, linewidth=2, color="black")
fig.axes[0].tick_params(top=True, labeltop=True, bottom=True, labelbottom=True)

# plt.xlabel('Surface', fontweight ='bold', fontsize = 15) 
fig.axes[0].set_ylabel('Seidel aberrations', fontweight ='bold', fontsize = 15) 
fig.axes[0].set_xticks([r + 2*barWidth for r in range(num_surfs+1)], 
       [str(r) for r in range(num_surfs)]+["SUM"])

fig.axes[0].set_title("Surfaces", fontsize=15)
fig.axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5, fontsize=15)

plt.show()