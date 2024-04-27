# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 13:10:06 2024

@author: jrjol
"""
import random
import numpy as np
import math as m
import matplotlib.pyplot as plt
import math as m

def binary_number_conversion(binary): 
    '''

    Parameters
    ----------
    binary : string inp. e.g. ['0011001100101....']
        

    Returns: scaled number (float)
    -------
    rescaled_number : TYPE
        This function converts a binary string, into an integer, then converts
        it into a continous variable.
        
        131071 is the greatest number achieved from a 17 bit length binary 
        string.
        
        The x10 multipler is to scale the number back up where the upper limit
        is 10.
        
        the variable 'number' will be summed to for each of the binary values
        in the binary string.
        
        'l' is simply the number of bits present in the binary string, a loop
        is then formed to check over each of these values, to check if they
        are ==1 or ==0.

    '''
    number = 0
    l = len(binary)
    for i in range(0, l):
        if int(binary[i])==0:
            number += 0
        else:
            number += 2**(l-1-i)
    rescaled_number = number/131071*10
    return rescaled_number

def tournament_selection(Population, subset_size, N, dimensions):
    '''
    

    Parameters
    ----------
    Population : Scaler (integer)
        This is the population size of each generation. 
    subset_size : Scaler (integer)
        A subset of the Population will be picked based on the subset size. 
        The selected parents will be used for the tournament selection proces.
    N : Not specifically required.

    dimensions : scaler (integer)
        simply tells the function how many chromosomes are present, this is 
        useful later performing an objective function evaluation.

    Returns
    -------
    binary_string_parent_one : String
        string of the selected 'best fitness' parent one
    binary_string_parent_two : String
        string of the selected '2nd best fitness' parent two

    '''
    
    l=N
    w = dimensions
    store=[] # storing all selected chromosomes for the subset
    picked_chromosomes = set() # ensuring no resampling
    subset_size_counter=0
    ### this loops picks enough parents from the generation to provide a 
    ### subset, for picking optimal parents. (Picks through random sampling).
    while subset_size_counter <subset_size:
        random_number = random.randint(0, l-1)
        if random_number not in picked_chromosomes: # no resampling
            picked_chromosomes.add(random_number)
            store.append(Population[random_number])
            subset_size_counter+=1
        
    
    ## This loop performs the tournament selection algorithm, picking based
    ## on best fitness, from an obj_eval.
    fitness = 0
    best_fitness=0
    second_best_fitness=0
    index_best=0
    index_2nd=0
    for i in range(0, subset_size):
        x = binary_strings_to_x_values(store[i][0], dimensions)
        fitness = Objective_function(x, w) # x_values inputted as list
        if fitness < best_fitness:  # inverted the max problem
            best_fitness=fitness
            index_best = i
        if best_fitness < fitness < second_best_fitness:
            second_best_fitness=fitness
            index_2nd=i
    binary_string_parent_one = store[index_best]
    binary_string_parent_two = store[index_2nd]
    return binary_string_parent_one, binary_string_parent_two

def Roulette_Wheel(Population, subset_size, N, dimensions):
    '''
    This process differs from tournament selection, as the subset selected 
    parents, further undergo a probability based selection for being the 
    optimal parent, simulating a roulette wheel.

    The greater the obj func val of the parent, the greater proportion of the
    roulette wheel they take up, increases their chances of selection for 
    breeding. 

    Parameters
    ----------
    Population : Scaler (integer)
        This is the population size of each generation. 
    subset_size : Scaler (integer)
        A subset of the Population will be picked based on the subset size. 
        The selected parents will be used for the tournament selection proces.
    N : Not specifically required.

    dimensions : scaler (integer)
        simply tells the function how many chromosomes are present, this is 
        useful later performing an objective function evaluation.

    Returns
    -------
    binary_string_parent_one : String
        string of the selected 'best fitness' parent one
    binary_string_parent_two : String
        string of the selected '2nd best fitness' parent two

    '''

    l=N
    w = dimensions
    store=[] # storing all selected chromosomes for the subset
    picked_chromosomes = set() # no resampling when building the subset.
    subset_size_counter=0
    ### building the subset ###
    while subset_size_counter <subset_size:
        random_number = random.randint(0, l-1)
        if random_number not in picked_chromosomes: # no resampling
            picked_chromosomes.add(random_number)
            store.append(Population[random_number])
            subset_size_counter+=1

    ## calculating each of the parents in the subsets probability of selection.
    fitnesses=[]
    probability=[]
    cumulative_probability=[]
    for i in range(0, subset_size):
        x = binary_strings_to_x_values(store[i][0], dimensions)
        fitness = Objective_function(x, w) 
        fitnesses.append(fitness)
    fitnesses = np.array(fitnesses)**5
    Sum = sum(fitnesses)
    cum_prob=0
    for i in range(0, subset_size):
        prob = fitnesses[i]/Sum
        probability.append(prob)
        cum_prob+=prob
        cumulative_probability.append(cum_prob)

    ## probability based parent selection ##
    indexes_used = set()
    I=0
    index_best_parents = []
    while I <= 1:
        random_float = random.random()
        lower_limit = 0
        for i in range(0, subset_size):
            #print(lower_limit)
            if lower_limit < random_float < cumulative_probability[i]:
                if i not in index_best_parents: # no resampling
                    indexes_used.add(i)
                    index_best_parents.append(store[i])
                    I+=1
            else:
                lower_limit += cumulative_probability[i]    
    binary_string_parent_one = index_best_parents[0]
    binary_string_parent_two = index_best_parents[1]
    return binary_string_parent_one, binary_string_parent_two

def Stochastic_Universal_Sampling(Population, subset_size, N, dimensions):
    '''
    This process differs from roulette wheel selection, as two parents will be 
    picked simultaneously from a single roulette wheel spin, hence there are
    essentially two balls or two pointers on the roulette wheel.
    The greater the obj func val of the parent, the greater proportion of the
    roulette wheel they take up, increases their chances of selection for 
    breeding. 

    Parameters
    ----------
    Population : Scaler (integer)
        This is the population size of each generation. 
    subset_size : Scaler (integer)
        A subset of the Population will be picked based on the subset size. 
        The selected parents will be used for the tournament selection proces.
    N : Not specifically required.

    dimensions : scaler (integer)
        simply tells the function how many chromosomes are present, this is 
        useful later performing an objective function evaluation.

    Returns
    -------
    binary_string_parent_one : String
        string of the selected 'best fitness' parent one
    binary_string_parent_two : String
        string of the selected '2nd best fitness' parent two

    '''

    l=N
    w = dimensions
    store=[] # storing all selected chromosomes for the subset
    picked_chromosomes = set()
    subset_size_counter=0
    
    while subset_size_counter <subset_size:
        random_number = random.randint(0, l-1)
        if random_number not in picked_chromosomes: # no resampling
            picked_chromosomes.add(random_number)
            store.append(Population[random_number])
            subset_size_counter+=1
    
    ##sizing the roulette wheel proportions based on their fitnesses.
    fitnesses=[]
    probability=[]
    cumulative_probability=[]
    for i in range(0, subset_size):
        x = binary_strings_to_x_values(store[i][0], dimensions)
        fitness = Objective_function(x, w) # x_values inputted as list
        fitnesses.append(fitness)
    fitnesses = np.array(fitnesses)**12
    Sum = sum(fitnesses)
    cum_prob=0
    for i in range(0, subset_size):
        prob = fitnesses[i]/Sum
        probability.append(prob)
        cum_prob+=prob
        cumulative_probability.append(cum_prob)
    
    ## now picking the two parents at random, simultaneously
    indexes_used = set()
    index_best_parents = []
    random_float_1 = random.random()
    random_float_2 = random.random()
    lower_limit = 0
    for i in range(0, subset_size):
        if lower_limit < random_float_1 or random_float_2 < cumulative_probability[i]:
            if i not in index_best_parents: # no resampling
                indexes_used.add(i)
                index_best_parents.append(store[i])
        else:
            lower_limit += cumulative_probability[i]    
    binary_string_parent_one = index_best_parents[0]
    binary_string_parent_two = index_best_parents[1]
    return binary_string_parent_one, binary_string_parent_two


def single_point_crossover(binary_string_parent_one,\
                           binary_string_parent_two, Pc, dimensions):
    '''
    The two parents inputted here should come from a selection method e.g.
    Tournamnet selection, rouletter wheel, or stochastic universal sampling.
    
    A single section of the RHS of the parents chromosome is swapped via 
    single point crossover. 
    

    Parameters
    ----------
    binary_string_parent_one : List of strings
        should be in the format of ['0101001...', '0010001010...', ...]
    binary_string_parent_two : List of strings
        should be in the format of ['0101001...', '0010001010...', ...]
    Pc : float
        Probability of crossover for the parent chromosomes.
    dimensions : interger
        How many chromosomes are present for each parent. e.g. 8D, would mean 
        8 chromosomes.

    Returns
    -------
    binary_string_parent_one : List of binary strings
        Returns a list of binary string, where each string (chromosome) may 
        have undergone some crossover with the other optimal parent. 
    binary_string_parent_two : List of binary strings
        Returns a list of binary string, where each string (chromosome) may 
        have undergone some crossover with the other optimal parent. 

    '''
    w = dimensions
    binary_string_parent_one = binary_string_parent_one[0]
    binary_string_parent_one_copy = binary_string_parent_one
    binary_string_parent_two = binary_string_parent_two[0]
    binary_string_parent_two_copy = binary_string_parent_two
    ## probability of cross over between 2 parents
    random_float = random.random()
    if random_float<Pc: # probability of crosover has been satisfied.
        #print('breed')
        new_parent_one=[] # new string ready for the newly mixed offspring chromosomes
        new_parent_two=[] # new string ready for the newly mixed offspring chromosomes
        for i in range(0, w):

            binary_p1 = list(binary_string_parent_one[i])
            binary_p2 = list(binary_string_parent_two[i])
            
            random_number = random.randint(0, 16)  # Generates a random number between 0 and 16 (inclusive)
            
            # Swap substrings based on random_number
            swap = binary_p1[random_number:]
            binary_p1[random_number:] = binary_p2[random_number:]
            binary_p2[random_number:] = swap
            
            # Convert lists back to strings
            binary_string_parent_one_swapped = ''.join(binary_p1)
            binary_string_parent_two_swapped = ''.join(binary_p2)
            
            new_parent_one.append(binary_string_parent_one_swapped)   # Updating each column (each x value)
            new_parent_two.append(binary_string_parent_two_swapped)   # Updating each column (each x value)
               
    ## feasibility checker ##
        x = binary_strings_to_x_values(new_parent_one, dimensions)
        fitness = Objective_function(x, dimensions)
        if fitness==np.inf:
            binary_string_parent_one=binary_string_parent_one_copy
            #Rejected
        else:
            binary_string_parent_one = new_parent_one
            #Accepted
        x = binary_strings_to_x_values(new_parent_two, dimensions)
        fitness = Objective_function(x, dimensions)
        if fitness==np.inf:
            #Rejected
            binary_string_parent_two = binary_string_parent_two_copy
        else:
            binary_string_parent_two = new_parent_two
            #accepted    
    return binary_string_parent_one, binary_string_parent_two

def double_point_crossover(binary_string_parent_one,\
                           binary_string_parent_two, Pc, dimensions):
    '''
    
    The two parents inputted here should come from a selection method e.g.
    Tournamnet selection, rouletter wheel, or stochastic universal sampling.
    
    A section, usually in the middle of the parent chromosome is selected and
    swappend. The location of the seciton comes from the 'two (double)' point 
    selection method.

    Parameters
    ----------
    binary_string_parent_one : List of strings
        should be in the format of ['0101001...', '0010001010...', ...]
    binary_string_parent_two : List of strings
        should be in the format of ['0101001...', '0010001010...', ...]
    Pc : float
        Probability of crossover for the parent chromosomes.
    dimensions : interger
        How many chromosomes are present for each parent. e.g. 8D, would mean 
        8 chromosomes.

    Returns
    -------
    binary_string_parent_one : List of binary strings
        Returns a list of binary string, where each string (chromosome) may 
        have undergone some crossover with the other optimal parent. 
    binary_string_parent_two : List of binary strings
        Returns a list of binary string, where each string (chromosome) may 
        have undergone some crossover with the other optimal parent. 

    '''
    w = dimensions
    binary_string_parent_one = binary_string_parent_one[0]
    binary_string_parent_one_copy = binary_string_parent_one
    binary_string_parent_two = binary_string_parent_two[0]
    binary_string_parent_two_copy = binary_string_parent_two
    ## probability of cross over between 2 parents
    random_float = random.random()
    if random_float<Pc: # probability of parent crossover satisfied.
        new_parent_one=[]
        new_parent_two=[]
        for i in range(0, w):
            binary_p1 = list(binary_string_parent_one[i])
            binary_p2 = list(binary_string_parent_two[i])
            
            # finding where on the chromosome (binary string) the crossover 
            # should occur.
            random_number_one = random.randint(0, 15)  # Generates a random number between 0 and 16 (inclusive)
            random_number_two = random.randint(random_number_one, 16)
            
            # Swap substrings based on random_number
            swap = binary_p1[random_number_one:random_number_two + 1]
            binary_p1[random_number_one:random_number_two + 1] = binary_p2[random_number_one:random_number_two + 1]
            binary_p2[random_number_one:random_number_two + 1] = swap
            
            # Convert lists back to strings
            binary_string_parent_one_swapped = ''.join(binary_p1)
            binary_string_parent_two_swapped = ''.join(binary_p2)
            #print(binary_string_parent_one_swapped, binary_string_parent_two_swapped)
                        
            new_parent_one.append(binary_string_parent_one_swapped)   # Updating each column (each x value)
            new_parent_two.append(binary_string_parent_two_swapped)   # Updating each column (each x value)
    ## feasibility checker ##
        x = binary_strings_to_x_values(new_parent_one, dimensions)
        fitness = Objective_function(x, dimensions)
        if fitness==np.inf:
            binary_string_parent_one=binary_string_parent_one_copy
            #Rejected
        else:
            binary_string_parent_one = new_parent_one
            #Accepted
        
        x = binary_strings_to_x_values(new_parent_two, dimensions)
        fitness = Objective_function(x, dimensions)
        if fitness==np.inf:
            #Rejected
            binary_string_parent_two = binary_string_parent_two_copy
        else:
            binary_string_parent_two = new_parent_two
            #Accepted
    
    return binary_string_parent_one, binary_string_parent_two
    
        

def mutation(binary_string, dimensions, Pm):
    '''
    Mutation of a single parent. If the probability of a mutation at a gene 
    within a chromosome is satisfied, then the allele will be change 'mutated'
    from 0->1 or 1->0.

    Parameters
    ----------
    binary_string : List of strings
        should be in the format of ['0101001...', '0010001010...', ...]
    dimensions : Integer
        8D = 8 dimensions, 2D = 2 dimensions
    Pm : Float
        Probability of mutation.

    Returns
    -------
    binary_string_parent_one : List of strings
        The newly mutated offspring chromosomes as binary strings.
    fitness_return : Scaler (float)
        The corresponding fitness
    x_values_return : List of floats
        x values (not requred, but can be useful for debug purposes.)

    '''

    #mutation for a single solution (parent DNA)
    binary_string_copy = binary_string
    x_values_original = binary_strings_to_x_values(binary_string_copy, dimensions)
    fitness_original = Objective_function(x_values_original, dimensions)    
    w = dimensions

    new_parent_one=[] # ready to store the new mutated chromosomes
    for i in range(0, w):

        binary_p1 = list(binary_string[i])        
        for j in range(len(binary_p1)):
            random_float = random.random()  # Generates a random number between 0 and 16 (inclusive)
            if random_float < Pm: # Probability of mutation satisfied.
                if int(binary_p1[j])==0:
                    
                    binary_p1[j]=1
                else:
                    
                    binary_p1[j]=0
        
        # Convert lists back to strings
        binary_string_parent_one_swapped = ''.join(map(str,binary_p1))        
        new_parent_one.append(binary_string_parent_one_swapped)   # Updating each column (each x value)
 
## feasibility checker ##
    x = binary_strings_to_x_values(new_parent_one, dimensions)
    fitness = Objective_function(x, dimensions)
    if fitness==np.inf or m.isnan(fitness):
        binary_string_parent_one=binary_string_copy
        fitness_return = fitness_original
        x_values_return = x_values_original
    else:
        binary_string_parent_one = new_parent_one
        fitness_return = fitness
        x_values_return = x

    return binary_string_parent_one, fitness_return, x_values_return
    


random_number = random.randint(0, 131071)  # 131071 is 2^17 - 1
binary_string = format(random_number, '017b')  # '017b' formats the number with 17 bits
number = binary_number_conversion(binary_string)

def binary_strings_to_x_values(x_values_bin_strings, dimensions):
    '''
    

    Parameters
    ----------
    x_values_bin_strings : This converts a list of chromosomes into x values.
        List of binary strings
    dimensions : Integer
        How many chromosomes need to be analysed for. 

    Returns
    -------
    x_values : List
        Returns the x-values corresponding to each chromosome, this is required 
        when inputting into the objective function for evaluation.

    '''
    x_values=[]
    for i in range(0, dimensions):
        rescaled_number = binary_number_conversion(x_values_bin_strings[i])
        x_values.append(rescaled_number)
    return x_values
    
def Initial_value(dimensions):
    '''
    Used to generate starting values.

    Parameters
    ----------
    dimensions : Integer
        How many x values (chromosomes need to be created)

    Returns
    -------
    list2 : List of binary strings
        List of binary strings for a feasible start point
    fitness : Float
        Scaler, proving feasible starting point.
    x_values : List
        List of x values for the starting point.

    '''
    list1=[]
    list2=[]
    feasible=0
    while feasible==0:
        for i in range(0, dimensions):
            # Generate a random integer between 0 and 131071 (2^17 - 1)
            random_number = random.randint(0, 131071)  # 131071 is 2^17 - 1
    
            # Convert the random number to a binary string with 17 bits
            binary_string = format(random_number, '017b')  # '017b' formats the number with 17 bits
            list1.append(binary_string)
            
        ### feasible starting point checker ###
        x_values = binary_strings_to_x_values(list1, dimensions)
        fitness = Objective_function(x_values, dimensions)
        
        if abs(fitness) != np.inf:  ### feasible starting location
            list2.append(list1)
            feasible=1
            #feasible            
        else:  ### infeasible starting coordinates, regenerate new inital solution
            list1=[]
            #infeasible

    return list2, fitness, x_values

def Objective_function(x, n):
    '''
    
    Fitness Function
    
    ### Parameters ###
    ----------
    x : vector of inputs (the dataset must be )
        the x coordinates.
    n : scaler
        Length of x.

    Returns
    Fitness of the system (Scalar)

    '''
    x = np.array(x)
    
    stiffness=-1e-8 # allowing for solutions very close to the constraints.
    sum1=np.sum(np.cos(x)**4)
    
    sum2=0
    for i in range(1,n+1):
        sum2+=i*x[i-1]**2
    if np.any(x<0) or np.any((10-x)<0):
        # infeasible
         B=np.inf
    else:    
         B=np.sum(stiffness*np.log(x)+stiffness*np.log(10-x))
    
    sum4 = np.sum(x)
    
    prod1=np.prod(np.cos(x)**2)
    
    prod2=np.prod(x)
    A=-abs((sum1-2*prod1)/m.sqrt(sum2))
    if prod2-0.75<=0:
        C=np.inf # infeasible
    else:
        C=stiffness*np.log(prod2-0.75)
    if 15*n/2-sum4<=0:
        D=np.inf # infeasible
    else:
        D=stiffness*np.log(15*n/2-sum4)
    Function_output = A+B+C+D
    return Function_output

def Genetic_Algorithm(N, Pc=0.85, Pm=0.1, subset_size=15, dimensions=8):
    '''
    Parameters
    ----------
    N : Integer
        Population size.
    Pc : Float, optional
        Probability of parent chromosome crossover. The default is 0.85.
    Pm : Float, optional
        Probability of gene mutation occuring. The default is 0.1.
    subset_size : Integer, optional
        How many parents will be selected for best fitness selection. 
        The default is 15.
    dimensions : Interger, optional
        Dimensions of the problem at hand. The default is 8.

    Returns
    -------
    Population : List
        This is the coordinates of the all offsping population, which become 
        parents of the next generation.
    Best_solutions : List
        These are the best fitness values of each generation.
    Best_x_values : List
        The x-values corresponding the the best fitnesses of each generation.
    All_x_values_in_each_generation : List
        All the x-values of each generation (useful for debugging and plotting
                                             contour maps in 2D).

    '''
    Population = []
    initial_fitness =[]
    initial_xs = []
    All_x_values_in_each_generation = []
    ### Generate a population of N initial solutions ###
    for i in range(0,N):
        list2, fitness, x_values = Initial_value(dimensions)
        Population.append(list2)
        initial_fitness.append(fitness)
        initial_xs.append(x_values)
        
    best_initial_fitness, best_initial_index = min((val, idx) for idx, val \
                                                 in enumerate(initial_fitness))
    All_x_values_in_each_generation.append(initial_xs)

    ## New generation production ##
    Generations=0
    Best_solutions=[]
    Best_x_values= []
    Best_solutions.append(best_initial_fitness)
    Best_x_values.append(initial_xs[best_initial_index])
    tolerance=np.inf
    max_gen_allowed=10000/(N*2)-1
    co=0  # Set to 0 for two point crossover to occur, or 1 for one point 
          # crossover to occur (for the earlier generations)
    counter=0
    while Generations<max_gen_allowed and tolerance>1e-05:
        Generational_solutions = []
        Generational_x_values = []
        new_gen=0
        new_population=[]
        
        if Generations>0.7*max_gen_allowed:
            co=0 # set to 0 for double point or 1 for single point crossover
                 # to occur for the later generations.

        
        while new_gen<N:

            ### Parent selection ###
            binary_string_parent_one, binary_string_parent_two = \
                tournament_selection(Population, subset_size, N, dimensions)
            ### Crossover (breeding) ###
            if co==0: # two point crossover
                binary_string_child_one, binary_string_child_two = double_point_crossover(binary_string_parent_one, binary_string_parent_two, Pc, dimensions)
            
            else:   # one point crossover
                binary_string_child_one, binary_string_child_two = single_point_crossover(binary_string_parent_one, binary_string_parent_two, Pc, dimensions)

            new_gen+=2

            ### mutation ###
            binary_string_child_one, fitness, x_values = \
                mutation(binary_string_child_one, dimensions, Pm)
            Generational_solutions.append(fitness)
            Generational_x_values.append(x_values)
            
            binary_string_child_two, fitness, x_values = \
                mutation(binary_string_child_two, dimensions, Pm)
            Generational_solutions.append(fitness)
            Generational_x_values.append(x_values)
            
            children=[]
            children.append(binary_string_child_one)
            new_population.append(children)
            
            children=[]
            children.append(binary_string_child_two)
            new_population.append(children)

        best_, _index = min((val, idx) for idx, val in enumerate(Generational_solutions))
        Best_solutions.append(best_)
        counter+=1
        if Best_solutions[counter]==Best_solutions[counter-1]:
            Pm=0.05
        Best_x_values.append(Generational_x_values[_index])
        All_x_values_in_each_generation.append(Generational_x_values)
        
        Population=new_population

        Generations+=1

        #print(Generations)

    return Population, Best_solutions, Best_x_values, All_x_values_in_each_generation
GA, Best_solutions, Best_x_values, All_x_values_in_each_generation = Genetic_Algorithm(170, Pm=0.3, subset_size=40,dimensions=2)
'''
Example run through 2D. Proof of convergence
'''
generations = [0, 1, 2, len(Best_solutions)-1]

for q in generations:
    # Generating grid values for x1 and x2
    x1_values = np.linspace(0.1, 10, 100)  # Define a range for x1
    x2_values = np.linspace(0.1, 10, 100)  # Define a range for x2

    # Creating a grid of points using np.meshgrid for x1 and x2
    x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)

    # Reshaping x1_grid and x2_grid into column vectors and stacking them vertically
    stacked_x = np.vstack([x1_grid.reshape(-1), x2_grid.reshape(-1)]).T

    # Initialize z_values array
    z_values = np.zeros_like(x1_grid)
    x_values_to_plot = All_x_values_in_each_generation[q]
    # Calculate Objective_function values for each point in the grid
    for i, x_pair in enumerate(stacked_x):
        z_values[i // len(x1_values), i % len(x1_values)] = Objective_function(x_pair, 2)  # Assuming n = 3

    # Create a contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(x1_grid, x2_grid, z_values, cmap='viridis')
    for x_coord in x_values_to_plot:
        plt.scatter(x_coord[0], x_coord[1], color='red', marker='o')  # Plot each coordinate as a red point

    plt.colorbar(contour, label='Objective Function Value')
    plt.title("Contour Plot of Objective Function. Generation: {} ".format(q))
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()
plt.plot(Best_solutions, marker='o')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Best Fitness For Each Generation 2D')
plt.show()

results = []
for Q in range(0,5):
    GA, Best_solutions, Best_x_values, All_x_values_in_each_generation = Genetic_Algorithm(170, Pm=0.3, subset_size=40,dimensions=2)
    results.append(abs(Best_solutions[-1]))
print('Mean located optima of 100 random seed tests: ', sum(results)/len(results))
print('--------------------------------')

''' Example run through in 8D '''
GA, Best_solutions, Best_x_values, All_x_values_in_each_generation = Genetic_Algorithm(170)

plt.plot(Best_solutions, marker='o')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Best Fitness For Each Generation 8D')
plt.show()
        
        
# Outputs = []
# # Errors = []
# Z=15
# for s in range(Z):
#     GA, Best_solutions, Best_x_values, All_x_values_in_each_generation = Genetic_Algorithm(170)
#     Outputs.append(abs(Best_solutions[-1]))
#     #Errors.append(abs(Outputs-0.724))
# RMSE = round(m.sqrt(np.sum((np.array(Outputs)-0.724)**2)/len(Outputs)),4)
# mu = round(sum(Outputs)/len(Outputs),4)
# # print('Mean outputs: ', mu)
# # print('Standard deviation of outputs: ', round(np.std(Outputs),4))
# # print('Lowest energy output: ', round(min(Outputs),4))
# # print('Highest energy output: ', round(max(Outputs),4))
# # print('RMSE from 0.724 (max from literature): ', RMSE)

# x_values = range(len(Outputs))

# # Scatter plot
# plt.scatter(x_values, Outputs, label='Fitness Values', color='blue', marker='o')

# # Add a horizontal line at y=0.724
# plt.axhline(y=0.724, color='red', linestyle='--', label='Approx maxmimum (Literature)')
# plt.axhline(y=mu, color='green', linestyle='--', label='Mean Fitness output')

# # Set y-axis limits
# plt.ylim(0.0, max(Outputs) + 0.1)  # Adjust the upper limit as needed

# # Title and labels
# plt.title('{} random seed trials and the final identified maximums'.format(Z))
# plt.xlabel('Random seed trial')
# plt.ylabel('Final Fitness values identified')

# # Display the legend
# plt.legend()

# # Show the plot
# plt.show()

'''
 Mutation and cross over probabilites
'''
# The following tests various mutation and cross over probabilities.
# The results are exported and plotted in excel for a cleaner 3D plot, in 
# comparison to matplotlib features.
# '''
Pm_used = []
Pc_used = []
Means = []
RMSEs = []
Standard_devs = []

Z=15
for A in np.arange(0.3,1.0,0.1): # probability of crossover
    for B in np.arange(0.04,0.16, 0.02):# probability of mutation 
        Outputs = []
        for s in range(Z):
            GA, Best_solutions, Best_x_values, All_x_values_in_each_generation = Genetic_Algorithm(60, Pc=A, Pm=B)
            Outputs.append(abs(Best_solutions[-1]))
            #Errors.append(abs(Outputs-0.724))
        RMSE = round(m.sqrt(np.sum((np.array(Outputs)-0.724)**2)/len(Outputs)),4)
        RMSEs.append(RMSE)
        mu = round(sum(Outputs)/len(Outputs),4)
        Means.append(mu)
        Standard_devs.append(round(np.std(Outputs),4))
        # Pm_used.append(A)
        Pc_used.append(B)
    Pm_used.append(A)
    print('Mean outputs: ', mu)
    print('Standard deviation of outputs: ', round(np.std(Outputs),4))
    print('Lowest energy output: ', round(min(Outputs),4))
    print('Highest energy output: ', round(max(Outputs),4))
    print('RMSE from 0.724 (max from literature): ', RMSE)


# '''
# The following section tests the effects of different subset sizes which are 
# used for parent selection.

# '''
Subset_size_used = []
RMSEs_vs_size = []
Standard_devs_vs_size = []
Means_vs_size = []
Outputs = []
Z=15
for Size in range(5,55,5):
    Outputs = []
    for s in range(Z):
          GA, Best_solutions, Best_x_values, All_x_values_in_each_generation = Genetic_Algorithm(60, subset_size=Size)
          Outputs.append(abs(Best_solutions[-1]))
            #Errors.append(abs(Outputs-0.724))
    RMSE = round(m.sqrt(np.sum((np.array(Outputs)-0.724)**2)/len(Outputs)),4)
    RMSEs_vs_size.append(RMSE)
    mu = round(sum(Outputs)/len(Outputs),4)
    Means_vs_size.append(mu)
    Standard_devs_vs_size.append(round(np.std(Outputs),4))
    # Pm_used.append(A)
    Subset_size_used.append(Size)
plt.plot(Subset_size_used, Means_vs_size, label='Mean Optima')
plt.plot(Subset_size_used, Standard_devs_vs_size, label='Standard deviation')
plt.plot(Subset_size_used,RMSEs_vs_size, label='RMSE')
# Adding labels and title
plt.xlabel('Subset size')
plt.ylabel('Value')
plt.title('Effect of Subset size for parent selection on GA')
# Adding legend
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# Display the plot
plt.show()
        

# '''
# The following section tests the effects of different population sizes.
# '''
Population_sizes_tested = []
RMSEs_vs_pop_size = []
Standard_devs_vs_pop_size = []
Means_vs_pop_size = []
Z=15
for P in range(30,190,20):
    Outputs = []
    for s in range(Z):
          GA, Best_solutions, Best_x_values, All_x_values_in_each_generation = Genetic_Algorithm(P)
          Outputs.append(abs(Best_solutions[-1]))
            #Errors.append(abs(Outputs-0.724))
    RMSE = round(m.sqrt(np.sum((np.array(Outputs)-0.724)**2)/len(Outputs)),4)
    RMSEs_vs_pop_size.append(RMSE)
    mu = round(sum(Outputs)/len(Outputs),4)
    Means_vs_pop_size.append(mu)
    Standard_devs_vs_pop_size.append(round(np.std(Outputs),4))
    # Pm_used.append(A)
    Population_sizes_tested.append(P)
plt.plot(Population_sizes_tested, Means_vs_pop_size, label='Mean Optima')
plt.plot(Population_sizes_tested, Standard_devs_vs_pop_size, label='Standard deviation')
plt.plot(Population_sizes_tested,RMSEs_vs_pop_size, label='RMSE')
# Adding labels and title
plt.xlabel('Population size')
plt.ylabel('Value')
plt.title('Effect of population size on GA')
# Adding legend
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# Display the plot
plt.show()


