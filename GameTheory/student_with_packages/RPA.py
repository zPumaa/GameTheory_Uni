import pandas as pd
import matplotlib.pyplot as plt


file_path = 'data.xlsx'  
sheet_name = 'Regression Profit Analysis'
data = pd.read_excel(file_path, sheet_name=sheet_name)


data.set_index('Follower', inplace=True)

# Plotting
ax = data.plot(kind='bar', figsize=(10, 7), width=0.8)
plt.title('Profit Comparison Across Different Regression Models')
plt.ylabel('Profit')
plt.xlabel('Follower')
plt.xticks(rotation=0)  
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(title='Regression Type')
plt.tight_layout()


for p in ax.patches:  
    ax.annotate(format(p.get_height(), '.2f'),  
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center',  
                va = 'center',  
                xytext = (0, 9),  
                textcoords = 'offset points')  

plt.show()
