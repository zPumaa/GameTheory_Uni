import pandas as pd
import matplotlib.pyplot as plt


file_path = 'data.xlsx' 
sheet_name = 'Follower 2 Test'
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Set the index to the regression method names
data.set_index(data.columns[0], inplace=True)  

# Plotting
ax = data.plot(kind='bar', figsize=(10, 6), width=0.8, color=['#1f77b4', '#ff7f0e'])
plt.title('Profit Comparison for Follower 2 with Different Regression Models')
plt.ylabel('Profit')
plt.xlabel('Regression Method')
plt.xticks(rotation=0)  
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(title='Data Set', labels=['Whole Dataset', 'Without Day 4'])
plt.tight_layout()

# Annotate each bar with its value
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', 
                va='center', 
                xytext=(0, 10), 
                textcoords='offset points')

plt.show()
