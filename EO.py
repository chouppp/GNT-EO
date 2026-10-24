import random
# The random module provides pseudorandom number generators.
import numpy as np  # NumPy supports scientific computing.
import pandas as pd  # pandas supports data processing and analysis.
import matplotlib.pyplot as plt  # Matplotlib is a popular data visualization library.
import re  # Regular expressions match, find, and replace text using patterns.
import rich.console  # Rich renders formatted text in the terminal.
from rich.table import Table  # This class works with Console to create and display tables.

# Define five variables and their domains.
domains = {
    'x1': [i for i in range(0, 16)],
    'x2': [i for i in range(0, 11)],
    'x3': [i for i in range(0, 26)],
    'x4': [i for i in range(0, 5)],
    'x5': [i for i in range(0, 31)],
}

# Define the constraints.
constraints = {

    ('x1', 'x2'): lambda x1, x2:  170*x1 <= 3800-310*x2,
    ('x2', 'x1'): lambda x2, x1:  3800-310*x2 >= 170*x1,

    ('x3', 'x4'): lambda x3, x4:  60*x3 <= 2800-101*x4,
    ('x4', 'x3'): lambda x4, x3:  2800-101*x4 >= 60*x3,

    ('x3', 'x5'): lambda x3, x5:  60*x3 <= 3500-11*x5,
    ('x5', 'x3'): lambda x5, x3:  3500-11*x5 >= 60*x3,

}

# revise updates the domain of x using the constraints between x and y.
# Check each value of x against all values of y to find a satisfying combination.
# Remove unsupported x values and return whether the domain changed.
def revise(x, y):
    revised = False  # Track whether the domain has been revised.
    x_domain = domains[x]
    y_domain = domains[y]  # Read x_domain and y_domain from domains for constraint checking and revision.
    all_constraints = [
        constraint for constraint in constraints if constraint[0] == x and constraint[1] == y]
    # Collect all constraints involving x and y in all_constraints.
    # Constraint keys are tuples of variable names identifying the relevant pair.
    for x_value in x_domain:  # Iterate over each x_value in the current domain of x.
        satisfies = False  # Initially assume no y value satisfies the constraints for this x_value.
        for y_value in y_domain:
            for constraint in all_constraints:  # Iterate over y values and all constraints involving x and y.
                constraint_func = constraints[constraint]
                # Retrieve the constraint function, defined earlier as a lambda.
                if constraint_func(x_value, y_value):  # Evaluate the function with x_value and y_value.
                    satisfies = True
        if not satisfies:  # If satisfies remains False, no y value supports the current x_value.
            x_domain.remove(x_value)
            revised = True  # Remove x_value from x_domain and mark revised as True.
    return revised  # Return whether the domain was revised.


# AC-3 maintains arc consistency in a constraint satisfaction problem.
# Repeatedly call revise on constraint arcs to update variable domains.
# Arc consistency means every x value has a supporting y value for each arc (x, y).
# Continue processing arcs as domains change until no work remains.
def ac3(arcs):
    queue = arcs[:]  # Copy all input arcs into the pending queue.
    while queue:
        (x, y) = queue.pop(0)
        revised = revise(x, y)
    # While the queue is nonempty, retrieve an arc and unpack its endpoints as x and y.
    # Call revise and record whether the domain of x changed.
        if revised:
            neighbors = [neighbor for neighbor in arcs if neighbor[1] == x]
            queue = queue + neighbors
        # If x changed, identify other arcs whose second variable is x.
        # Collect these arcs from the original arcs list into neighbors.
        # Queue affected arcs for further revision to maintain network-wide arc consistency.


# Define the constraint arcs and apply AC-3 to revise the variable domains.
# Print the domains of x1 through x5 after constraint propagation.
arcs = [
    ('x1', 'x2'), ('x2', 'x1'),
    ('x3', 'x4'), ('x4', 'x3'),
    ('x3', 'x5'), ('x5', 'x3')
]
# Define six directed arcs as pairs of variable names corresponding to constraints.
# Include both directions for x1/x2, x3/x4, and x3/x5 as AC-3 inputs.

ac3(arcs)
# Run AC-3, repeatedly revising domains according to the supplied arcs.
# Requeue affected arcs whenever a domain changes, until the queue is empty.
# Ensure every remaining value has supporting values in related variable domains.

print("x1: " + str(domains['x1']))
print("x2: " + str(domains['x2']))
print("x3: " + str(domains['x3']))
print("x4: " + str(domains['x4']))
print("x5: " + str(domains['x5']))
# Print the final domains of x1 through x5 to inspect the effect of constraint propagation.
# Domains may shrink when values have no supporting assignments; the output shows this.
# The resulting domains help assess feasibility and search for a complete solution.

# Problema defines domains, costs, budget limits, constraint checks, quality scores, and transformations.
# These methods support feasibility checks and solution evaluation in search or evolutionary algorithms.
class Problema:
    def __init__(self, epsilon):
        self.epsilon = epsilon  # Define the threshold epsilon.
        self.dimensiones = 5  # Define five variables.
        self.limites = {
            'x1': (0, 15),
            'x2': (0, 10),
            'x3': (0, 25),
            'x4': (0, 4),
            'x5': (0, 30)
        }  # Define the domains of the five variables.

        self.costos = {
            'x1': 170,
            'x2': 310,
            'x3': 60,
            'x4': 101,
            'x5': 11
        }  # Define the cost coefficients for the five variables.

        self.presMaxTv = 3800  # Define budget limits; why is presMaxDyR assigned twice?
        self.presMaxDyR = 2800
        self.presMaxDyR = 3500

    # Check whether the five-dimensional solution x satisfies every constraint.
    def check(self, x):
        # Calculate costs for different combinations.
        costo_tv = self.costos['x1'] * x[0] + self.costos['x2'] * x[1]
        costo_print = self.costos['x3'] * x[2] + self.costos['x4'] * x[3]
        costo_combined = self.costos['x3'] * x[2] + self.costos['x5'] * x[4]

        # Return False if the budget is exceeded.
        if costo_tv > self.presMaxTv:
            return False
        if costo_print > self.presMaxDyR:
            return False
        if costo_combined > self.presMaxDyR:
            return False

        # Enumerate each variable's index and bounds c_min and c_max.
        # Return False if any component of x lies outside its allowed interval.
        for i, (c_min, c_max) in enumerate(self.limites.values()):
            if not (c_min <= x[i] <= c_max):
                return False

        # Sum variable values times cost coefficients; reject total costs above epsilon.
        # Return True only if all constraints hold.
        totalCosto = 170 * x[0] + 310 * x[1] + 60 * x[2] + 101 * x[3] + 11 * x[4]
        if totalCosto > self.epsilon:
            return False

        return True

    # Return total cost as the sum of variable values times coefficients from self.costos.
    def checkCostos(self, x):
        totalCosto = 170 * x[0] + 310 * x[1] + 60 * x[2] + 101 * x[3] + 11 * x[4]
        return totalCosto

    # Compute total quality as a weighted sum of variable values for solution evaluation.
    # For example, evaluate quality and cost together in multi-objective optimization.
    def checkQuality(self, x):
        totalQuality = 70 * x[0] + 91 * x[1] + 50 * x[2] + 61 * x[3] + 21 * x[4]
        return totalQuality

    # Evaluate fitness using the same calculation as checkQuality.
    # Fitness measures solution quality and is maximized or minimized according to the problem.
    def eval(self, x):  # Calculate fitness.
        return 70 * x[0] + 91 * x[1] + 50 * x[2] + 61 * x[3] + 21 * x[4]

    # Sigmoid is a nonlinear map to (0, 1), useful for smoothing, probabilities, and activation functions.
    # Later methods in this class build on this transformation.
    def sigmoide(self, x, alpha, x0):
        return 1 / (1 + np.exp(-alpha * (x - x0)))

    # Find the interval index containing y, typically a sigmoid output in [0, 1].
    def find_y_interval(self, y, intervals):
        interval_width = 1 / intervals  # Compute each interval's width as 1 / intervals.
        for i in range(intervals):  # Find the interval containing y and return its index.
            if i * interval_width <= y < (i + 1) * interval_width:
                return i  # Map y == 1 to the last interval, intervals - 1.
        return intervals - 1 if y == 1 else None  # Return None for values outside the valid range.

    # Process the five-element arreglo using the predefined alpha, x0, and intervals parameters.
    # Apply sigmoide, then map the continuous result to a discrete interval.
    def master_sigmoide(self, arreglo):
        parameters = [
            (0.6, 7.5, 16),
            (1, 5, 11),
            (0.35, 12.5, 26),
            (2.5, 2, 5),
            (0.3, 15, 31)
        ]

        results_list = []
        for x, (alpha, x0, intervals) in zip(arreglo, parameters):
            y = self.sigmoide(x, alpha, x0)
            if intervals == 5:
                # For five intervals, use bins of width 0.2 over [0, 1] and append the index.
                # For example, [0, 0.2) maps to 0 and [0.2, 0.4) maps to 1.
                if 0 <= y < 0.2:
                    results_list.append(0)
                elif 0.2 <= y < 0.4:
                    results_list.append(1)
                elif 0.4 <= y < 0.6:
                    results_list.append(2)
                elif 0.6 <= y < 0.8:
                    results_list.append(3)
                elif 0.8 <= y <= 1:
                    results_list.append(4)
                else:
                    results_list.append(None)
            else:  # Otherwise use find_y_interval, then return the result as a NumPy array.
                results_list.append(self.find_y_interval(y, intervals))

        return np.array(results_list)

# Particula represents a particle associated with a Problema instance and a position self.x.
# Methods initialize positions, check feasibility, compare fitness, and format particle information.
class Particula:
  def __init__(self, problema):
    self.problema = problema
    # Initialize self.x as a zero vector of length problema.dimensiones.
    # Use this vector as the initial position representation.
    self.x = np.zeros(problema.dimensiones)
    # Randomly initialize each position component within its allowed bounds.
    self.inicializacion()

  # Initialize a particle position within the problem's valid variable ranges.
  def inicializacion(self):
    for j in range(self.problema.dimensiones):
    # Iterate over dimensions and retrieve the lower and upper bounds c_min and c_max.
    # Read each bounds tuple from problema.limites using index j.
      c_min, c_max = list(self.problema.limites.values())[j]
    # Sample uniformly as c_min + random.random() * (c_max - c_min).
      self.x[j] = (c_min + random.random() * (c_max - c_min))

  # Check whether self.x satisfies all constraints in the associated Problema instance.
  def esFactible(self, x):
    return self.problema.check(self.x)

  # Compare this particle with comp to determine which better meets the optimization objective.
  def esMejorQue(self, comp):
    return self.fit() > comp.fit()
  # Evaluate current-position fitness to guide particle comparison and updates.
  def fit(self):
    return self.problema.eval(self.x)

  # Override __str__ to display particle fitness and position for inspection and debugging.
  def __str__(self):
    # String representation of the particle.
    return f"fit:{self.fit()} x:{self.x}"


# An adapted Equilibrium Optimizer for the problem represented by the Problema instance.
# Manage the swarm and equilibrium candidates through initialization, pool construction, and evolution.
# Search for improved solutions and display their particle information.
class EquilibriumOptimizer2:
    def __init__(self, problema, n, MAX_ITER, a1, a2, GP):
        self.problema = problema
        self.nParticulas = n  # Number of particles in the population.
        self.maxIter = MAX_ITER  # Maximum iteration count, limiting the optimization loop.
        self.a1 = a1  # a1 controls exploration amplitude; larger values permit greater distances from equilibrium candidates.
        self.a2 = a2  # a2 controls exploitation of promising solutions.
        self.GP = GP  # GP controls generation-term participation: GP=1 disables it, favoring exploration; GP=0 always enables it, increasing the risk of local stagnation.
        self.V = 1
        self.enjambre = []  # Initialize the particle list, populated later by inicializarPoblacion.
        self.eq_candidatos = [Particula(problema) for _ in range(4)]
        # Create four equilibrium candidates that will be updated as better particles are found.

        self.lower_band = [self.problema.limites[f"x{i + 1}"][0] for i in range(len(self.problema.limites.keys()))]
        self.upper_band = [self.problema.limites[f"x{i + 1}"][-1] for i in range(len(self.problema.limites.keys()))]
        # Extract lower and upper bounds from each tuple in problema.limites.
        # Use these bounds in subsequent position-validity checks.

    # Create self.nParticulas feasible particles and add them to self.enjambre.
    def inicializarPoblacion(self):
        print("Particulas creation: \n")  # Start creating a particle.
        for _ in range(self.nParticulas):
            while True:
                particula = Particula(self.problema)  # Create a Particula instance.
                # Transform the new position with problema.master_sigmoide, for example by discretization or normalization.
                # The exact transformation is defined by Problema.master_sigmoide.
                particula.x = self.problema.master_sigmoide(particula.x)
                print(particula.x)
                if particula.esFactible(particula.x):  # Use esFactible to check whether the particle satisfies all constraints.
                    self.enjambre.append(particula)
                    # Append a feasible particle and exit the inner loop to create the next one.
                    # Continue until the required number of feasible particles has been created.
                    break

    # Replace equilibrium candidates with better swarm particles as they are found.
    def updateCandidatosEq(self):
        for particula in self.enjambre:  # Iterate over particles in the swarm.
            for i in range(len(self.eq_candidatos)):  # Iterate over equilibrium candidate indices.
                # Use esMejorQue to compare the particle with the current equilibrium candidate.
                # Replace a worse candidate, then break and continue with the next swarm particle.
                if particula.esMejorQue(self.eq_candidatos[i]):
                    self.eq_candidatos[i] = particula
                    break

    # Build an equilibrium pool from the candidates and a transformed particle at their mean position.
    def construirEqPool(self):
        eqPromedio = Particula(self.problema)  # Create the mean particle eqPromedio.
        # Average candidate positions along axis 0 and store the result as a list in eqPromedio.x.
        eqPromedio.x = np.mean([candidato.x for candidato in self.eq_candidatos], axis=0).tolist()
        # Transform the mean position using problema.master_sigmoide.
        eqPromedio.x = self.problema.master_sigmoide(eqPromedio.x)
        # Return the candidates together with the transformed mean particle.
        return self.eq_candidatos + [eqPromedio]


    # Evolve positions for self.maxIter iterations while maintaining feasibility.
    def evolucion(self):

        for iter in range(1, self.maxIter + 1):  # Loop over optimization iterations.

            self.updateCandidatosEq()  # Update equilibrium candidates with updateCandidatosEq.
            eq_pool = self.construirEqPool()  # Build the equilibrium pool with construirEqPool for this iteration's position updates.

            # Compute t using Eq. (9); t decreases as the iteration count increases.
            # Iter and Max_iter denote current and maximum iterations; a2 controls exploitation.
            t = (1 - iter / self.maxIter) ** (self.a2 * iter / self.maxIter)

            for particula in self.enjambre:  # Iterate over all particles.

                while True:  # For each particle, repeat position updates until the constraints are satisfied.

                    eq_candidato = random.choice(eq_pool)  # Randomly choose eq_candidato from the equilibrium pool.
                    # Generate vectorLambda and vectorR, with one uniform [0, 1] value per dimension.
                    # Use these random vectors and t in the position-update equations.
                    vectorLambda = np.random.rand(self.problema.dimensiones)
                    vectorR = np.random.rand(self.problema.dimensiones)

                    # Compute F using Eq. (11), controlling the magnitude and direction of position updates.
                    F = self.a1 * np.sign(vectorR - 0.5) * (np.exp(-vectorLambda * t) - 1)

                    # Compute the generation control parameter GCP using Eq. (15).
                    # Based on random draws and GP, each component is zero or a value between 0 and 0.5.
                    GCP = np.where(np.random.rand(self.problema.dimensiones) >= self.GP, 0.5 * random.random(), 0)

                    # Compute G0 using Eq. (14), combining GCP with the particle-candidate position difference.
                    G0 = GCP * (eq_candidato.x - vectorLambda * particula.x)

                    # Compute the generation rate G using Eq. (13).
                    G = G0 * F

                    # Update particula.x using Eq. (16), combining current and candidate positions with F and G.
                    # The first term is equilibrium concentration; the second and third terms describe concentration changes.
                    # The second term supports global exploration through large concentration differences.
                    # The third term refines solutions through small changes controlled by the generation rate in Eq. (13).
                    particula.x = eq_candidato.x + ((particula.x - eq_candidato.x) * F) + (G / vectorLambda) * (1 - F)
                    # Transform the updated position with problema.master_sigmoide for the required representation.
                    particula.x = self.problema.master_sigmoide(particula.x)
                    # Check esFactible and retry until the updated position satisfies the constraints.
                    # Repeat for all particles and subsequent iterations until the iteration limit is reached.
                    if (particula.esFactible(particula.x)):
                        break

    # Initialize the swarm, evolve it, update equilibrium candidates, and report the resulting solutions.
    # Display solution fitness and position using Particula.__str__.
    def solve(self):
        self.inicializarPoblacion()
        self.evolucion()
        self.updateCandidatosEq()
        mejoresParticulas = self.eq_candidatos
        print("\n\nMejores particulas: ")
        for particula in mejoresParticulas:
            print(particula)


import os

exp = 10        # Repeat the experiment 10 times.
n = 30          # Use 30 particles.
MAX_ITER = 5  # Run five iterations.
a1 = 2
a2 = 1.5
GP = 0.5
epsilon = 3000  # {3000!, 4000, 5000!}: 4000 does not work; the reason is not yet known.
results = []    # Collect experiment results in a list for later conversion to a DataFrame.
for i in range(exp):
    problema = Problema(epsilon)  # Create Problema with epsilon to initialize its constraints.
    print(f"Parametros = a1:{a1}, a2:{a2}, epsilon:{epsilon}, maxIter: {MAX_ITER}")
    optimizer = EquilibriumOptimizer2(problema, n, MAX_ITER, a1=a1, a2=a2, GP=GP)  # Create an EquilibriumOptimizer2 instance.
    optimizer.solve()  # Call solve to initialize, evolve, and obtain equilibrium candidates.
    optimizer.updateCandidatosEq()  # Refresh equilibrium candidates before collecting results, as particle states may have changed.
    # Retrieve the particles currently considered best.

    for particula in optimizer.eq_candidatos:  # Iterate over the final equilibrium candidates.
        if particula.esFactible(particula.x) == False:
            # Check feasibility through esFactible and the associated Problema.check method.
            # Stop collecting candidates when an infeasible particle is encountered.
            break
        results.append({  # Collect each feasible particle's information in a dictionary and append it to results.
            'x': particula.x.tolist(),
            # Convert the NumPy position vector to a list for DataFrame storage and processing.
            # The position vector represents a possible solution in the problem space.
            'fit': particula.fit(),
            # Retrieve fitness through the particle's fit method and Problema.eval.
            # Fitness measures solution quality and is the quantity being optimized.
            'Maximizacion': particula.problema.checkQuality(particula.x),  # Pass the position to problema.checkQuality.
            # Retrieve the solution's maximization objective value.
            'Minimizacion': particula.problema.checkCostos(particula.x),
            # Retrieve solution cost through problema.checkCostos.
            # Compute total cost as the sum of values times cost coefficients.
            'Factibilidad': particula.esFactible(particula.x),  # Record feasibility again, although it was already checked before collecting the result.
            # The redundant field makes result validity explicit for later inspection.
            'a1': a1,
            'a2': a2,
            'epsilonConstraint': epsilon,  # Record experiment parameters to compare performance across parameter combinations.
        })

df = pd.DataFrame(results)  # Convert the list of particle-result dictionaries into a pandas DataFrame.
directory = f'experimentos_e_{epsilon}'  # Construct the experiment output directory name from epsilon using an f-string.
# For epsilon = 3000, use experimentos_e_3000.
if not os.path.exists(directory):  # Check whether the output directory already exists.
    os.makedirs(directory)  # Create the directory if needed before saving results.
filename = f'{directory}/resultadosExp_{a1}_a2_{a2}.csv'  # Construct the output filename from the directory and a1/a2 values.
# For a1 = 2 and a2 = 1, an example is experimentos_e_3000/resultadosExp_2_a2_1.csv.
df.to_csv(filename, index = False)  # Save the DataFrame in CSV format.
# index=False omits the default index column and saves only the data.
print(f"Resultados guardados en {filename}")  # Report the output filename after saving.