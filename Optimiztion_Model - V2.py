import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize
import matplotlib.pyplot as plt
import seaborn as sns
import random
# Define the problem
prob = LpProblem("Demand_Optimization", LpMinimize)

# Define parameters
number_days = 5
number_rooms = 12
number_slots_per_day = 16  # 30-minute slots from 8 AM to 4 PM
lunch_break_slots = [6, 7, 8, 9] # Assuming slot indices 8 and 9 correspond to 12 PM - 12:30 PM and 12:30 PM - 1 PM
number_providers = 6
number_nurses = 8
max_hours_per_week = 40

# Define random fluctuation function
def add_random_fluctuation(demand, fluctuation_range=2):
    return [[d + random.randint(-fluctuation_range, fluctuation_range) for d in day] for day in demand]

# Original demand data
original_demand_endocrinology = [
    [4, 5, 6, 5, 4, 4, 3, 3, 3, 4, 5, 6, 4, 4, 3, 3],  # Monday
    [3, 4, 5, 5, 4, 4, 3, 3, 3, 4, 5, 6, 4, 4, 3, 3],  # Tuesday
    [4, 5, 6, 6, 4, 4, 3, 3, 4, 5, 6, 6, 5, 4, 3, 3],  # Wednesday
    [5, 6, 6, 5, 5, 5, 4, 4, 4, 5, 6, 6, 5, 5, 4, 4],  # Thursday
    [6, 6, 6, 5, 5, 5, 4, 4, 5, 6, 6, 6, 5, 5, 5, 4]  # Friday
]

original_demand_diabetes = [
    [8, 7, 6, 7, 8, 8, 9, 9, 9, 8, 7, 6, 8, 8, 9, 9],  # Monday
    [9, 8, 7, 7, 8, 8, 9, 9, 9, 8, 7, 6, 8, 8, 9, 9],  # Tuesday
    [8, 7, 6, 6, 8, 8, 9, 9, 8, 7, 6, 6, 7, 8, 9, 9],  # Wednesday
    [7, 6, 6, 7, 7, 7, 8, 8, 8, 7, 6, 6, 7, 7, 8, 8],  # Thursday
    [6, 6, 6, 7, 7, 7, 8, 8, 7, 6, 6, 6, 7, 7, 7, 8]  # Friday
]

# Apply random fluctuations
demand_endocrinology = add_random_fluctuation(original_demand_endocrinology)
demand_diabetes = add_random_fluctuation(original_demand_diabetes)


assign_endocrinology = [
    [
        [LpVariable(f'assign_endocrinology_{i+1}_{j+1}_{k+1}', cat='Binary')
         for i in range(number_rooms)]
        for j in range(number_days)]
    for k in range(number_slots_per_day)
]

assign_diabetes = [
    [
        [LpVariable(f'assign_diabetes_{i+1}_{j+1}_{k+1}', cat='Binary')
         for i in range(number_rooms)]
        for j in range(number_days)]
    for k in range(number_slots_per_day)
]
assign_providers_endocrinology = [
    [
        [LpVariable(f'assign_providers_endocrinology_{i+1}_{j+1}_{k+1}', cat='Binary')
         for i in range(number_providers)]
        for j in range(number_days)]
    for k in range(number_slots_per_day)
]

assign_providers_diabetes = [
    [
        [LpVariable(f'assign_providers_diabetes_{i+1}_{j+1}_{k+1}', cat='Binary')
         for i in range(number_providers)]
        for j in range(number_days)]
    for k in range(number_slots_per_day)
]

assign_nurses_endocrinology = [
    [
        [LpVariable(f'assign_nurses_endocrinology_{i+1}_{j+1}_{k+1}', cat='Binary')
         for i in range(number_nurses)]
        for j in range(number_days)]
    for k in range(number_slots_per_day)
]

assign_nurses_diabetes = [
    [
        [LpVariable(f'assign_nurses_diabetes_{i+1}_{j+1}_{k+1}', cat='Binary')
         for i in range(number_nurses)]
        for j in range(number_days)]
    for k in range(number_slots_per_day)
]

shortfall_endocrinology = [
    [LpVariable(f'shortfall_endocrinology_{j+1}_{k+1}', lowBound=0) 
     for k in range(number_slots_per_day)]
    for j in range(number_days)
]

shortfall_diabetes = [
    [LpVariable(f'shortfall_diabetes_{j+1}_{k+1}', lowBound=0) 
     for k in range(number_slots_per_day)]
    for j in range(number_days)
]
for j in range(number_days):
    for k in range(number_slots_per_day):
        prob += shortfall_endocrinology[j][k] >= demand_endocrinology[j][k] - lpSum(assign_endocrinology[k][j])
        prob += shortfall_diabetes[j][k] >= demand_diabetes[j][k] - lpSum(assign_diabetes[k][j])
for j in range(number_days):
    for k in range(number_slots_per_day):
        for i in range(number_rooms):
            prob += assign_endocrinology[k][j][i] + assign_diabetes[k][j][i] <= 1, f"Room_Usage_Constraint_{i+1}_{j+1}_{k+1}"
for j in range(number_days):
    for k in range(number_slots_per_day):
        prob += lpSum(assign_endocrinology[k][j]) <= number_rooms, f"Total_Rooms_Endocrinology_{j+1}_{k+1}"
        prob += lpSum(assign_diabetes[k][j]) <= number_rooms, f"Total_Rooms_Diabetes_{j+1}_{k+1}"
        
# Provider hours constraint
for p in range(number_providers):
    prob += lpSum(assign_providers_endocrinology[k][j][p] for k in range(number_slots_per_day) for j in range(number_days)) <= max_hours_per_week, \
        f"Max_Weekly_Hours_Providers_Endocrinology_{p+1}"
    prob += lpSum(assign_providers_diabetes[k][j][p] for k in range(number_slots_per_day) for j in range(number_days)) <= max_hours_per_week, \
        f"Max_Weekly_Hours_Providers_Diabetes_{p+1}"
    
# Nurse hours constraint
for n in range(number_nurses):
    prob += lpSum(assign_nurses_endocrinology[k][j][n] for k in range(number_slots_per_day) for j in range(number_days)) <= max_hours_per_week, \
        f"Max_Weekly_Hours_Nurses_Endocrinology_{n+1}"
    prob += lpSum(assign_nurses_diabetes[k][j][n] for k in range(number_slots_per_day) for j in range(number_days)) <= max_hours_per_week, \
        f"Max_Weekly_Hours_Nurses_Diabetes_{n+1}"

# Provider lunch break constraint
for j in range(number_days):
    for p in range(number_providers):
        prob += lpSum(assign_providers_endocrinology[lunch_slot][j][p] for lunch_slot in lunch_break_slots) == 0, f"Provider_{p+1}_Lunch_Break_Endocrinology_{j+1}_{p+1}"
        prob += lpSum(assign_providers_diabetes[lunch_slot][j][p] for lunch_slot in lunch_break_slots) == 0, f"Provider_{p+1}_Lunch_Break_Diabetes_{j+1}_{p+1}"

# Nurse lunch break constraint
for j in range(number_days):
    for n in range(number_nurses):
        prob += lpSum(assign_nurses_endocrinology[lunch_slot][j][n] for lunch_slot in lunch_break_slots) == 0, f"Nurse_{n+1}_Lunch_Break_Endocrinology_{j+1}_{n+1}"
        prob += lpSum(assign_nurses_diabetes[lunch_slot][j][n] for lunch_slot in lunch_break_slots) == 0, f"Nurse_{n+1}_Lunch_Break_Diabetes_{j+1}_{n+1}"

    
#Provider & nurse assignment constraints
for j in range(number_days):
    for k in range(number_slots_per_day):
        # Providers constraint
        prob += lpSum(assign_providers_endocrinology[k][j][i] for i in range(number_providers)) <= number_providers, f"Providers_Endocrinology_{j+1}_{k+1}"
        prob += lpSum(assign_providers_diabetes[k][j][i] for i in range(number_providers)) <= number_providers, f"Providers_Diabetes_{j+1}_{k+1}"
        
        # Nurses constraint
        prob += lpSum(assign_nurses_endocrinology[k][j][i] for i in range(number_nurses)) <= number_nurses, f"Nurses_Endocrinology_{j+1}_{k+1}"
        prob += lpSum(assign_nurses_diabetes[k][j][i] for i in range(number_nurses)) <= number_nurses, f"Nurses_Diabetes_{j+1}_{k+1}"
        
        # Constraints to ensure no room is over-utilized by providers or nurses (assigned > 1 provider or assigned > 1 nurse)
        for i in range(number_rooms):
            prob += lpSum(assign_providers_endocrinology[k][j][p] for p in range(number_providers)) + \
                    lpSum(assign_providers_diabetes[k][j][p] for p in range(number_providers)) <= 1, \
                f"Provider_Usage_Constraint_{i+1}_{j+1}_{k+1}"
            prob += lpSum(assign_nurses_endocrinology[k][j][n] for n in range(number_nurses)) + \
                    lpSum(assign_nurses_diabetes[k][j][n] for n in range(number_nurses)) <= 1, \
                f"Nurse_Usage_Constraint_{i+1}_{j+1}_{k+1}"
            
# Add dynamic break variables
lunch_break_var_providers = [
    [LpVariable(f"Provider_{p+1}_Lunch_Day_{j+1}", 0, 1, cat="Binary")
     for p in range(number_providers)]
    for j in range(number_days)
]

lunch_break_var_nurses = [
    [LpVariable(f"Nurse_{n+1}_Lunch_Day_{j+1}", 0, 1, cat="Binary")
     for n in range(number_nurses)]
    for j in range(number_days)
]
            
# Add lunch break constraints
for j in range(number_days):
    for p in range(number_providers):
        for k in lunch_break_slots:
            prob += assign_providers_endocrinology[k][j][p] + assign_providers_diabetes[k][j][p] <= (1 - lunch_break_var_providers[j][p]), \
                    f"Provider_{p+1}_Lunch_Break_Slot_{k}_Day_{j+1}"
        prob += lpSum(lunch_break_var_providers[j][p] for k in lunch_break_slots) == 1, \
                f"Provider_{p+1}_One_Lunch_Break_Day_{j+1}"

    for n in range(number_nurses):
        for k in lunch_break_slots:
            prob += assign_nurses_endocrinology[k][j][n] + assign_nurses_diabetes[k][j][n] <= (1 - lunch_break_var_nurses[j][n]), \
                    f"Nurse_{n+1}_Lunch_Break_Slot_{k}_Day_{j+1}"
        prob += lpSum(lunch_break_var_nurses[j][n] for k in lunch_break_slots) == 1, \
                f"Nurse_{n+1}_One_Lunch_Break_Day_{j+1}"
        
#Objective function and solving the problem
prob += lpSum(shortfall_endocrinology[j][k] for j in range(number_days) for k in range(number_slots_per_day)) + \
        lpSum(shortfall_diabetes[j][k] for j in range(number_days) for k in range(number_slots_per_day))
prob.solve()

# Print results
total_shortfall_endocrinology = sum(s.value() for j in range(number_days) for s in shortfall_endocrinology[j])
total_shortfall_diabetes = sum(s.value() for j in range(number_days) for s in shortfall_diabetes[j])
valid_solution = True

for j in range(number_days):
    for k in range(number_slots_per_day):
        rooms_endocrinology = [i+1 for i in range(number_rooms) if assign_endocrinology[k][j][i].value() == 1]
        rooms_diabetes = [i+1 for i in range(number_rooms) if assign_diabetes[k][j][i].value() == 1]
        
        # Calculate additional rooms needed
        assigned_endocrinology = sum(assign_endocrinology[k][j][i].value() for i in range(number_rooms))
        assigned_diabetes = sum(assign_diabetes[k][j][i].value() for i in range(number_rooms))
        
        additional_endocrinology = max(demand_endocrinology[j][k] - assigned_endocrinology, 0)
        additional_diabetes = max(demand_diabetes[j][k] - assigned_diabetes, 0)

        if additional_endocrinology > 0 or additional_diabetes > 0:
            valid_solution = False
        
        print(f"Day {j+1}, Slot {k+1}:")
        print(f"  Rooms assigned to Endocrinology: {rooms_endocrinology}")
        print(f"  Rooms assigned to Diabetes: {rooms_diabetes}")
        print(f"  Additional Rooms Needed for Endocrinology: {additional_endocrinology}")
        print(f"  Additional Rooms Needed for Diabetes: {additional_diabetes}")

print(f"Total Shortfall Endocrinology = {total_shortfall_endocrinology}")
print(f"Total Shortfall Diabetes = {total_shortfall_diabetes}")

if valid_solution:
    print("Model has successfully met all constraints!")
else:
    print("Model did not meet all constraints.")

# Example visualization of shortfall over days for endocrinology and diabetes
days = range(1, number_days + 1)
shortfall_endocrinology_values = [sum(shortfall_endocrinology[j][k].value() for k in range(number_slots_per_day)) for j in range(number_days)]
shortfall_diabetes_values = [sum(shortfall_diabetes[j][k].value() for k in range(number_slots_per_day)) for j in range(number_days)]

plt.figure(figsize=(12, 6))

# Plot for endocrinology
plt.subplot(1, 2, 1)
plt.plot(days, shortfall_endocrinology_values, marker='o', color='blue', label='Endocrinology')
plt.title('Shortfall in Endocrinology over Days')
plt.xlabel('Day')
plt.ylabel('Shortfall')
plt.xticks(days)
plt.grid(True)
plt.legend()

# Plot for diabetes
plt.subplot(1, 2, 2)
plt.plot(days, shortfall_diabetes_values, marker='o', color='green', label='Diabetes')
plt.title('Shortfall in Diabetes over Days')
plt.xlabel('Day')
plt.ylabel('Shortfall')
plt.xticks(days)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

