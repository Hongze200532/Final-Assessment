'''
Question 3 - Agent-based modeling
Hongze Lin
'''

# ==================== Import packages ====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from OpinionModel import OpinionWorld

# Random Seed
SEED = 42

# ==================== Q3(a) ====================
model_a = OpinionWorld(seed=SEED) # Default confidence_threshold = 2.0 [cite: 62]

# 50 steps
for _ in range(50):
    model_a.step()

# Extract opinions Data
df_a = model_a.datacollector.get_agent_vars_dataframe()
opinions_a = df_a.unstack('AgentID')['opinion']

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(opinions_a, alpha=0.3, linewidth=1)
plt.title('Q3(a): Agent Opinions Over Time (Default d=2.0)')
plt.xlabel('Time Step')
plt.ylabel('Opinion')
plt.ylim(-1.1, 1.1)
plt.grid(True, alpha=0.5)
plt.show()

# ==================== Q3(b)(i) ====================
d_values_test = [0.5, 0.2]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for i, d in enumerate(d_values_test):
    model_b = OpinionWorld(confidence_threshold=d, seed=SEED)
    for _ in range(50):
        model_b.step()
    
    df_b = model_b.datacollector.get_agent_vars_dataframe()
    opinions_b = df_b.unstack('AgentID')['opinion']
    
    axes[i].plot(opinions_b, alpha=0.3, linewidth=1)
    axes[i].set_title(f'Agent Opinions (d={d})')
    axes[i].set_xlabel('Time Step')
    axes[i].set_ylabel('Opinion')
    axes[i].set_ylim(-1.1, 1.1)
    axes[i].grid(True, alpha=0.5)

plt.tight_layout()
plt.show()

# ==================== Q3(b)(ii) ====================
import numpy as np
import matplotlib.pyplot as plt
from OpinionModel import OpinionWorld

SEED = 42 

# Fucntion
def ret_num_cliques(opinions, confidence_threshold):
    num_cliques = 0
    sorted_constrained_opinions = sorted([opinion for opinion in opinions if abs(opinion) <= (1 - (0.5 * confidence_threshold))])
    
    if not sorted_constrained_opinions: 
        return 0
        
    x_min = sorted_constrained_opinions[0]
    num_cliques += 1
    
    for opinion in sorted_constrained_opinions:
        if opinion > (x_min + confidence_threshold):
            x_min = opinion
            num_cliques += 1
            
    return num_cliques

# Define d searching range [0.1, 0.9]
d_range = np.arange(0.1, 1.0, 0.1)
clique_counts = []

for d in d_range:
    model_b2 = OpinionWorld(confidence_threshold=d, seed=SEED)
    for _ in range(100): # 100 steps
        model_b2.step()
    final_opinions = [a.opinion for a in model_b2.agents]
    clique_counts.append(ret_num_cliques(final_opinions, d))

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(d_range, clique_counts, marker='o', linestyle='-', color='purple')
plt.title('Q3(b)(ii): Number of Cliques vs Confidence Threshold')
plt.xlabel('Confidence Threshold (d)')
plt.ylabel('Number of Cliques')
plt.grid(True, alpha=0.5)
plt.show()

# ==================== Q3(c)(i) ====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from OpinionModel import OpinionWorld

SEED = 42
fixed_d = 0.3
test_neighborhoods = [1, 3, 10]

# 1. Plotting opinion trajectories for different neighborhood sizes
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, n_size in enumerate(test_neighborhoods):
    model_c = OpinionWorld(confidence_threshold=fixed_d, neighbourhood_size=n_size, seed=SEED)
    for _ in range(60):
        model_c.step()
    df_c = model_c.datacollector.get_agent_vars_dataframe().unstack('AgentID')['opinion']
    axes[i].plot(df_c, alpha=0.3)
    axes[i].set_title(f'Neighbourhood Size = {n_size}')
    axes[i].set_ylim(-1.1, 1.1)
    axes[i].grid(True, alpha=0.5)

plt.tight_layout()
plt.show()

# 2. Plotting boxplots of extremist proportions for multiple runs
runs_per_condition = 10  # Runs per condition (10 times) for boxplot sampling
n_sizes_box = [1, 2, 3, 5, 10, 21]
extremist_data = []

for n_size in n_sizes_box:
    for run in range(runs_per_condition):
        # Use random seed for each run
        model_box = OpinionWorld(confidence_threshold=fixed_d, neighbourhood_size=n_size, seed=SEED+run)
        for _ in range(50):
            model_box.step()
        df_model = model_box.datacollector.get_model_vars_dataframe()
        final_extremists = df_model['proportion_of_extremists'].iloc[-1]
        extremist_data.append({'Neighbourhood Size': n_size, 'Proportion of Extremists': final_extremists})

    # Plotting (boxplots)
df_box = pd.DataFrame(extremist_data)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Neighbourhood Size', y='Proportion of Extremists', data=df_box)
plt.title('Q3(c)(i): Extremist Proportion vs Neighbourhood Size')
plt.grid(True, alpha=0.3)
plt.show()

# ==================== Q3(d) ====================
import numpy as np
import matplotlib.pyplot as plt
from OpinionModel import OpinionWorld

SEED = 42

# Running model with normal distribution initialization
model_d = OpinionWorld(initialisation_type="normal", seed=SEED)

for _ in range(50):
    model_d.step()

df_d = model_d.datacollector.get_agent_vars_dataframe().unstack('AgentID')['opinion']

plt.figure(figsize=(10, 6))
plt.plot(df_d, alpha=0.3)
plt.title('Q3(d): Opinions Over Time (Normal Distribution)')
plt.xlabel('Time Step')
plt.ylabel('Opinion')
plt.ylim(-1.1, 1.1)
plt.grid(True, alpha=0.5)
plt.show()

# ==================== Q3(e) ====================
import numpy as np
import matplotlib.pyplot as plt
from OpinionModel import OpinionWorld

SEED = 42

# Assuming 20% of the population are stubborn negative voters (stubborn_proportion=0.2)
model_e = OpinionWorld(initialisation_type="stubborn", stubborn_proportion=0.2, seed=SEED)

for _ in range(100):
    model_e.step()

df_e = model_e.datacollector.get_agent_vars_dataframe().unstack('AgentID')['opinion']

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df_e, alpha=0.3)
plt.axhline(y=-1, color='red', linestyle='--', linewidth=2, label='Stubborn Negative Opinion')
plt.title('Q3(e): Impact of Stubborn Negative Voters (20% proportion)')
plt.xlabel('Time Step')
plt.ylabel('Opinion')
plt.ylim(-1.1, 1.1)
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()