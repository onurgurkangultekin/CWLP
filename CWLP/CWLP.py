#lecture : Combinatorial Optimization
#project : Capacitated Warehouse Location Problem
#instructor : Dr.  Ali Hayrdar OZER
#author : Onur Gurkan GULTEKIN (o524115005)
#created : 12:40 PM, 28.04.2018
from enum import Enum
from io import StringIO
import random as rnd
import pandas as pd
import numpy as np
import gurobipy as grb
import time
import operator
import matplotlib.pyplot as plt

#variables
warehouse_limits = "warehouse_limits"
customer_demans = "customer_demans"
fixed_costs = "fixed_costs"
cost_matrix = "cost_matrix"
test_size = 30
size = 30
best_sample = 10
lucky_few = 10
number_of_child = 2
number_of_generation = 50
chance_of_mutation = 10
solution_dic = {}

class Size(Enum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3

# n is warehouse count, m is customer count
# a is warehouse limits, d is customer demands
def generate_test_case(size, k):
    m = 0
    n = 0
    if(size == Size.SMALL):
        m = 1000
        n = 10
    elif(size == Size.MEDIUM):
        m = 2000
        n = 20
    elif(size == Size.LARGE):
        m = 5000
        n = 50
    #ai = Maximum capacity (in units) of the warehouse i
    a = [0 for x in range(n)]
    #dj = Demand (in units) of customer j
    d = [0 for x in range(m)]
    #fi = fixed cost of operating warehouse i
    f = [0 for x in range(n)]
    #cij = cost of transporting one unit of good from warehouse i to customer j
    c = [[0.0 for x in range(m)] for y in range(n)]
    while(sum(a) <= sum(d)):
        for i in range(0, n):
           a[i] = rnd.randrange(3000, 21000)
        for i in range(0, m):
            d[i] = rnd.randrange(10, 110)

    for i in range(0, n):
        f[i] = rnd.randrange(3600, 7200)

    for i in range(0, len(c)):
        for j in range(0, len(c[i])):
            c[i][j] = round(rnd.uniform(0.2, 1.0), ndigits = 2)
    path = size.name + "\\" + str(k) + "_"
    write_to_csv(a, path + warehouse_limits)
    write_to_csv(d, path + customer_demans)
    write_to_csv(f, path + fixed_costs)
    write_to_csv(c, path + cost_matrix)

def write_to_csv(list, path):
    path = path + '.csv'
    np.savetxt(path, list, delimiter=',', fmt="%f")

def get_test_cases(size, k):
    path = size.name + "/" + str(k) + "_"
    a = np.loadtxt(path + warehouse_limits + ".csv",delimiter=",", dtype= int)
    d = np.loadtxt(path + customer_demans + ".csv",delimiter=",", dtype= int)
    f = np.loadtxt(path + fixed_costs + ".csv", delimiter=",", dtype= int)
    c = np.loadtxt(path + cost_matrix + ".csv", delimiter=",", dtype = float)
    return a, d, f, c

def generate_all_test_cases():
    for k in range(1,31):
        generate_test_case(Size.SMALL, k)
        generate_test_case(Size.MEDIUM, k)
        generate_test_case(Size.LARGE, k)

def create_lp_file(a, d, f, c, size, k):
    name = "lp\\" + size.name + "_" + str(k) + ".lp"
    n = len(a) # warehouse size
    m = len(d) # customer size
    file_str = StringIO("")
    file_str.write("Minimize\n")
    #create objective function "obj"
    
    for i in range(0, n):
        file_str.write(str(f[i]))
        file_str.write(" y")
        file_str.write(str(i))
        file_str.write(" + ")

    for i in range(0, n):
        for j in range(0, m):
            file_str.write(str(c[i][j]))
            file_str.write(" x")
            file_str.write(str(i))
            file_str.write("_")
            file_str.write(str(j))
            if not(i == n - 1 and j == m - 1):
                file_str.write(" + ")

    file_str.write("\nSubject To\n")
    #create constraints for customer demands
    for j in range(m):
        file_str.write("d")
        file_str.write(str(j))
        file_str.write(": ")
        for i in range(n):
            file_str.write("x")
            file_str.write(str(i))   
            file_str.write("_")
            file_str.write(str(j))
            if(i != n - 1):
                file_str.write(" + ")
        file_str.write(" = ")
        file_str.write(str(d[j]))
        file_str.write("\n")

    #create constraints for warehouse limits
    for i in range(n):
        file_str.write("a")
        file_str.write(str(i))
        file_str.write(": ")
        for j in range(m):
            file_str.write("x")
            file_str.write(str(i))
            file_str.write("_")
            file_str.write(str(j))
            if(j != m - 1):
                file_str.write(" + ")
        file_str.write(" - ")
        file_str.write(str(a[i]))
        file_str.write(" y")
        file_str.write(str(i))
        file_str.write(" <= 0\n")

    file_str.write("Integers\n")
    #integer variables
    for i in range(n):
        for j in range(m):
            file_str.write("x")
            file_str.write(str(i))
            file_str.write("_")
            file_str.write(str(j))
            file_str.write(" ")
    
    file_str.write("\nBinaries\n")
    #binary variables
    for i in range(n):
        file_str.write("y")
        file_str.write(str(i))
        file_str.write(" ")

    f = open(name, "w+")
    f.write(file_str.getvalue())
    f.close()

def create_all_lp_files():
    for size in Size:
        for k in range(1, 31):
            a, d, f, c = get_test_cases(size, k)
            create_lp_file(a, d, f, c, size, k)

def solve_lp_problem(size, k):
    lp = "lp/" + size.name + "_" + str(k) + ".lp"
    sol = "sol/" + size.name + "_" + str(k) + ".sol"
    model = grb.Model("CWLP")
    model = grb.read(lp)
    model.setParam("TimeLimit", 10 * 60) # time limit : 10 minutes
    model.optimize()
    
def solve_lp_problem_own_write(size, k):
    lp = "lp/" + size.name + "_" + str(k) + ".lp"
    sol = "sol/" + size.name + "_" + str(k) + ".sol"
    model = grb.Model("CWLP")
    model = grb.read(lp)
    model.setParam("TimeLimit", 10 * 60) # time limit : 10 minutes
    model.optimize()
    f = open(sol, "w")
    f.write("objVal {0}\n".format(model.objVal))
    f.write("RunTime {0}\n".format(round(model.runtime, 2)))
    if(model.runtime > 10 * 60):
        f.write("isOptimal False\n")
    else:
        f.write("isOptimal True\n")
    for var in model.getVars():
        if(var.varName.startswith("y")):
            f.write("{0} {1}\n".format(var.varName, abs(var.X)))

def solve_all_lp_problems():
    for size in Size:
        for k in range(1, 31):
            solve_lp_problem_own_write(size, k)

def is_feasible(vars, y):
    a = vars[0]
    d = vars[1]
    demands = sum(d)
    y = [int(s) for s in y]
    capacity = sum(a * y)
    return capacity >= demands

def generate_first_random_feasible_sol(vars):
    a = vars[0]
    d = vars[1]
    n = len(a)
    sol_feasible = False
    while sol_feasible == False:
        y = [str(rnd.randint(0,1)) for i in range(n)]
        sol_feasible = is_feasible(vars, y)
    return ''.join(y)

def generate_best_random_feasible_sol(vars):
    a = vars[0]
    d = vars[1]
    n = len(a)
    sample = [i for i in range(n)]
    y = ['0' for i in range(n)]
    for i in range(1,n + 1):
        indexes = rnd.sample(sample, i)
        y0 = y.copy()
        for j in indexes:
            y0[j] = '1'
        if(is_feasible(vars, y0)):
            return ''.join(y0)

def evaluate_fitness(vars, y):
    a = vars[0]
    d = vars[1]
    f = vars[2]
    c = vars[3]
    n = len(a) #warehouse
    m = len(d) #customer
    warehouses = range(n)
    customers = range(m)
    model = grb.Model("facility")
    x = model.addVars(warehouses, customers, vtype=grb.GRB.INTEGER, obj=c, name="x")
    model.modelSense = grb.GRB.MINIMIZE
    #demand constraints
    model.addConstrs((x.sum('*', c) == d[c] for c in customers),"Demand")
    #capacity constraints
    model.addConstrs((x.sum(i) <= a[i] * int(y[i]) for i in warehouses), "Capacity")
    model.setParam("Presolve", 0) 
    model.optimize()
    cost = model.objVal
    fixed = 0
    for i in range(n):
        fixed += int(y[i]) * f[i]
    total = cost + fixed
    return total

def fitness(vars, y):
    a = vars[0].copy()
    d = vars[1].copy()
    f = vars[2]
    c = vars[3].copy()
    n = len(a) #warehouse
    m = len(d) #customer
    total = 0
    for i in range(n):
        if y[i] == '0':
            a[i] = 0
            c[i, :] = 2.0
        else:
            total += f[i]
    for i in range(m):
        demand_satisfied = False
        while demand_satisfied == False:
            ind = np.argmin(c[:, i])
            if d[i] <= a[ind]:
                total += c[ind][i] * d[i]
                a[ind] -= d[i]
                demand_satisfied = True
            else:
                total += c[ind][i] * a[ind]
                d[i] -= a[ind]
                a[ind] = 0
                c[ind, :] = 2
                demand_satisfied = d[i] == 0
    print(total)
    return total

def generate_first_population(size, vars):
    population = []
    for i in range(size):
        #always try to create a unique random solution
        sol_exists = True
        while(sol_exists): 
            new_sol = generate_first_random_feasible_sol(vars)
            sol_exists = new_sol in population
        population.append(new_sol)
    return population

def compute_fitness_population(population, vars):
    population_fitness = {}
    for individual in population:
        if(individual in solution_dic):
            population_fitness[individual] = solution_dic[individual]
        else:
            population_fitness[individual] = fitness(vars, individual)
            solution_dic[individual] = population_fitness[individual]
    return sorted(population_fitness.items(), key = operator.itemgetter(1), reverse=False) #best is first

def select_from_population(population_sorted, best_sample, lucky_few):
    next_generation = []
    population_array = np.array(population_sorted)[:,0].tolist()
    population_array.reverse()
    for i in range(best_sample):
        next_generation.append(population_array.pop())
    for i in range(lucky_few):
        selected_sol = rnd.choice(population_array)
        next_generation.append(selected_sol)
        population_array.remove(selected_sol)
    rnd.shuffle(next_generation)
    return next_generation

def create_child(indivudual1, indivudual2, vars):
    child_is_feasible = False
    while child_is_feasible == False:
        child = ''
        for i in range(len(indivudual1)):
            if int(100 * rnd.random()) < 50:
                child += indivudual1[i]
            else:
                child += indivudual2[i]
        child_is_feasible = is_feasible(vars, child)
    return child

def create_children(breeders, number_of_child, vars):
    next_population = []
    next_population.extend(breeders[:best_sample])
    for i in range(int(len(breeders) / 2)):
        for j in range(number_of_child):
            child_exists = True
            while child_exists:
                random_breeders = rnd.choices(breeders, k=2)
                new_child = create_child(random_breeders[0], random_breeders[1], vars)
                child_exists = new_child in next_population
            next_population.append(new_child)
    return next_population

def mutate_solution(y, vars):
    mutated_sol_feasible = False
    while mutated_sol_feasible == False:
        y0 = list(y)
        for i in range(len(y0)):
            #index = int(rnd.random() * len(y))
            y0[i] = str(rnd.randint(0,1))
        #else:
        #    y0[index] = '1'
        y0 = ''.join(y0)
        mutated_sol_feasible = is_feasible(vars, y0)
    return y0

def mutate_better(y, vars):
    mutated_sol_feasible = False
    while mutated_sol_feasible == False:
        y0 = list(y)
        open_found = False
        while open_found == False:
            index = int(rnd.random() * len(y))
            open_found = y0[index] == '1'
        if rnd.random() * 100 < 100 - chance_of_mutation:
            y0[index] = '0'
        else:
            y0[index] = '1'
        y0 = ''.join(y0)
        mutated_sol_feasible = is_feasible(vars, y0)
    return y0

def mutate_population(population, chance_of_mutation, vars):
    for i in range(len(population)):
        if rnd.random() * 100 < chance_of_mutation:
            new_sol = mutate_solution(population[i], vars)
            mutated_sol_exist = new_sol in population
            if mutated_sol_exist == False:
                population[i] = new_sol
    return population

def next_generation(first_generation, vars, best_sample, lucky_few, number_of_child, chance_of_mutation):
    population_sorted = compute_fitness_population(first_generation, vars)
    next_breeders = select_from_population(population_sorted, best_sample, lucky_few)
    next_population = create_children(next_breeders, number_of_child, vars)
    next_generation = mutate_population(next_population, chance_of_mutation, vars)
    return next_generation

def multiple_generation(number_of_generation,  vars, size, best_sample, lucky_few, number_of_child, chance_of_mutation):
    historic = []
    historic.append(generate_first_population(size, vars))
    for i in range(number_of_generation):
        historic.append(next_generation(historic[i], vars, best_sample, lucky_few, number_of_child, chance_of_mutation))
    return historic

def get_best_individual_from_population(population, vars):
    return compute_fitness_population(population, vars)[0]

def get_list_best_individual_from_history(historic, vars):
    best_individuals = []
    for population in historic:
        best_individuals.append(get_best_individual_from_population(population, vars))
    return best_individuals

def evolution_best_fitness(historic,best_sol):
    evolution_fitness = []
    for population in historic:
        evolution_fitness.append(get_best_individual_from_population(population, vars)[1])
    plt.title("Best solution : " + best_sol[0])
    plt.axis([0, len(historic), min(evolution_fitness) - 1000 , max(evolution_fitness) + 1000])
    evolution_fitness.sort(reverse=True)
    plt.plot(evolution_fitness)
    plt.ylabel("fitness best individual")
    plt.xlabel("generation")
    plt.show()

def evolution_avarage_fitness(historic, size):
    evolution_fitness = []
    for population in historic:
        population_fitness = compute_fitness_population(population, None)
        average_fitness = 0
        for individual in population_fitness:
            average_fitness += individual[1]
        evolution_fitness.append(average_fitness / size)
    
    plt.title("Avarage Fitness")
    plt.axis([0, len(historic), min(evolution_fitness) - 1000 , max(evolution_fitness) + 1000])
    plt.plot(evolution_fitness)
    plt.ylabel("Average Fitness")
    plt.xlabel("Generation")
    plt.show()

def print_simple_result(historic, number_of_generation, vars):
    result = get_list_best_individual_from_history(historic, vars)[number_of_generation - 1]
    print("solution: " + str(result[0]) + "fitness: " + str(result[1]))
    return result

def solve_all_with_genetic_algorithm():
    for size in Size:
        for k in range(1, 31):
            genetic_algorithm(size, k)

def genetic_algorithm(test_size, k):
    a, d, f, c = get_test_cases(test_size, k)
    vars = []
    vars.append(a)
    vars.append(d)
    vars.append(f)
    vars.append(c)
    t0 = time.time()
    historic = multiple_generation(number_of_generation, vars, size, best_sample, lucky_few, number_of_child, chance_of_mutation)
    t1 = time.time()
    best_sol = print_simple_result(historic, number_of_generation, vars)
    runtime = t1 - t0
    print("time : ", runtime)
    sol = "sol/" + test_size.name + "_" + str(k) + "_gen.sol"
    f = open(sol, "w")
    f.write("objVal {0}\n".format(best_sol[1]))
    f.write("RunTime {0}\n".format(round(runtime, 2)))
    for i in range(len(best_sol[0])):
            f.write("y{0} {1}\n".format(str(i), best_sol[0][i]))
    #evolution_best_fitness(historic, best_sol)
    #evolution_avarage_fitness(historic, size)
            
def write_results_to_file():
    for s in Size:
        path = 'compare/' + s.name + '.csv'
        file = open(path, 'w')
        headers = 'Instance,Is Optimal,Gurobi Time,Gurobi Obj Value,GA Time,GA Obj Val'
        file.write(headers)
        file.write('\n')
        for k in range(1,31):
            sol_path = 'sol_/' + s.name + '_' + str(k) + '.sol'
            gen_path = 'sol_/' + s.name + '_' + str(k) + '_gen.sol'
            sol_file = open(sol_path, 'r')
            gen_file = open(gen_path, 'r')
            sol_line = sol_file.readlines()
            gen_line = gen_file.readlines()
            gurobi_obj_value = sol_line[0].split()[1]
            gurobi_time = sol_line[1].split()[1]
            gurobi_is_optimal = sol_line[2].split()[1]
            gen_obj_value = gen_line[0].split()[1]
            gen_time = gen_line[1].split()[1]
            content = ''.join(s.name + '_' + str(k) + ',' + gurobi_is_optimal + ',' + gurobi_time + ',' + gurobi_obj_value + ',' + gen_time + ',' + gen_obj_value)
            file.write(content)
            file.write('\n')

            sol_file.close()
            gen_file.close()
        file.close()

def plot_compare_time(s):
    # j == 2 time compare
    # j == 4 objective value compare
    file = open('compare/' + s.name + '.csv', 'r')
    lines = file.readlines()
    sol_time, gen_time = read_results(s, 2)
    plt.axis([0, len(gen_time) + 1, 0, max(sol_time) + 1])
    plt.plot(sol_time)
    plt.plot(gen_time)
    plt.title("Time Compare -" + s.name)
    plt.ylabel("Time")
    plt.xlabel("Instance")
    plt.show()

def plot_compare_obj(s):
    # j == 2 time compare
    # j == 4 result compare
    file = open('compare/' + s.name + '.csv', 'r')
    lines = file.readlines()
    sol_obj = []
    gen_obj = []
    for i in range(1, len(lines) - 1):
        sol_obj.append(lines[i].split(',')[3])
        gen_obj.append(lines[i].split(',')[5])
    gen_obj = [float(f) for f in gen_obj]
    sol_obj = [float(f) for f in sol_obj]
    plt.axis([0, len(gen_obj) + 1, min(sol_obj) - 2000, max(gen_obj) + 2000])
    plt.plot(sol_obj)
    plt.plot(gen_obj)
    plt.title("Obj Value Compare -" + s.name)
    plt.ylabel("Fitness")
    plt.xlabel("Instance")
    plt.show()

def find_average_results():
    avg_sol_time = []
    avg_gen_time = []
    avg_sol_obj = []
    avg_gen_obj = []
    sol_time, gen_time = read_results(Size.SMALL, 2)
    avg_sol_time.append(sum(sol_time) / test_size)
    avg_gen_time.append(sum(gen_time) / test_size)
    sol_time, gen_time = read_results(Size.MEDIUM, 2)
    avg_sol_time.append(sum(sol_time) / test_size)
    avg_gen_time.append(sum(gen_time) / test_size)
    sol_time, gen_time = read_results(Size.LARGE, 2)
    avg_sol_time.append(sum(sol_time) / test_size)
    avg_gen_time.append(sum(gen_time) / test_size)

    sol_obj, gen_obj = read_results(Size.SMALL, 3)
    avg_sol_obj.append(sum(sol_obj) / test_size)
    avg_gen_obj.append(sum(gen_obj) / test_size)
    sol_obj, gen_obj = read_results(Size.MEDIUM, 3)
    avg_sol_obj.append(sum(sol_obj) / test_size)
    avg_gen_obj.append(sum(gen_obj) / test_size)
    sol_obj, gen_obj = read_results(Size.LARGE, 3)
    avg_sol_obj.append(sum(sol_obj) / test_size)
    avg_gen_obj.append(sum(gen_obj) / test_size)
    
    plt.plot(avg_sol_time, label='Solver', linewidth=1)
    plt.plot(avg_gen_time, label='GA')
    plt.title("Average Time Compare - Solver vs GA")
    plt.ylabel("Time")
    plt.xlabel("Instance")
    plt.show()
    
    plt.plot(avg_sol_obj, label='Solver')
    plt.plot(avg_gen_obj, label='GA')
    plt.title("Average Obj Value Compare - Solver vs GA")
    plt.ylabel("Time")
    plt.xlabel("Instance")
    plt.show()

def read_results(test_size, j):
    # j == 2 for time, j == 3 for objVal
    file = open('compare/' + test_size.name + '.csv', 'r')
    lines = file.readlines()
    sol_time = []
    gen_time = []
    for i in range(1, len(lines) - 1):
        sol_time.append(lines[i].split(',')[j])
        gen_time.append(lines[i].split(',')[j + 2])
    sol_time = [float(f) for f in sol_time]
    gen_time = [float(f) for f in gen_time]
    return sol_time, gen_time
def main():
    #generate_all_test_cases()
    #create_all_lp_files()
    #solve_all_lp_problems()
    #genetic_algorithm(Size.LARGE, 30)
    solve_all_with_genetic_algorithm()
    #solve_lp_problem(Size.SMALL, 30)
    #find_average_results();
    #write_results_to_file()
    #plot_compare_time(Size.SMALL)
    #plot_compare_time(Size.MEDIUM)
    #plot_compare_time(Size.LARGE)
    #plot_compare_obj(Size.SMALL)
    #plot_compare_obj(Size.MEDIUM)
    #plot_compare_obj(Size.LARGE)
if __name__ == "__main__":
    main()