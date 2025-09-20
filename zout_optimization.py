import numpy as np
import pandas as pd

import time
from typing import Callable, Tuple, List, Union


def constrained_optimization(objective_function: Callable) -> Tuple[np.ndarray, float, int]:
    """
    Constrained optimization using the active set method with simplex-based direction finding.
    
    This function solves constrained optimization problems of the form:
        minimize f(x)
        subject to g(x) <= 0
                  lower_bounds <= x <= upper_bounds
    
    Args:
        objective_function: Function that returns [f, g] where:
                          - f is the objective function value (scalar)
                          - g is array of constraint values (g <= 0 for feasibility)
        
    Returns:
        optimal_point: The optimal solution vector
        optimal_value: The optimal objective function value
        iterations: Number of iterations performed
        
    Example:
        def my_function(x):
            f = (x[0] - 3)**2 + (x[1] - 4)**2  # objective
            g = np.array([x[0] + x[1] - 5])     # constraint x1 + x2 <= 5
            return [f, g]
        
        x_opt, f_opt, iters = constrained_optimization(my_function)
    """
    
    # ===== INITIALIZATION PARAMETERS =====
    initial_point = np.array([0.5, 0.5])  # Starting guess
    lower_bounds = np.array([0.0, 0.0])   # Lower bounds for variables
    upper_bounds = np.array([5.0, 5.0])   # Upper bounds for variables
    
    # Gradient functions (problem-specific - should be customized for your problem)
    def objective_gradient(x: np.ndarray) -> np.ndarray:
        """Gradient of the objective function f(x) = (x1-3)^2 + (x2-4)^2"""
        return np.array([2*x[0] - 6, 2*x[1] - 8])
    
    def constraint_gradient(x: np.ndarray) -> np.ndarray:
        """Gradient of constraints g(x) = [x1 + x2 - 5]"""
        return np.array([1, 1])  # Gradient of x1 + x2 - 5
    
    # ===== ALGORITHM PARAMETERS =====
    num_active_constraints = 0      # Number of currently active constraints
    num_constraint_functions = 1    # Total number of constraint functions
    convergence_tolerance = 1e-4    # Main convergence criterion
    max_iterations = int(1e3)       # Maximum allowed iterations
    constraint_tolerance = 2e-3     # Tolerance for constraint activity
    gradient_tolerance = 1e-4       # Gradient-based convergence tolerance
    max_line_search_steps = 30      # Maximum steps in line search
    num_variables = len(initial_point)
    
    # ===== INITIALIZATION =====
    current_point = initial_point.copy()
    function_gradient_values = objective_function(current_point)
    current_objective = function_gradient_values[0]
    current_constraints = function_gradient_values[1] if len(function_gradient_values) > 1 else np.array([])
    
    # Convergence tracking variables
    convergence_counter = 0
    previous_objective = current_objective
    iteration_count = 0
    initial_objective = current_objective
    
    # Results storage for displaying iteration history
    iteration_results = []
    
    print("Starting constrained optimization using active set method...")
    start_time = time.time()
    
    # ===== MAIN OPTIMIZATION LOOP =====
    while True:
        iteration_count += 1
        
        # Step 1: Identify active constraints at current point
        num_active_constraints, active_constraint_indices = identify_active_constraints(
            num_constraint_functions, convergence_tolerance, current_constraints
        )
        
        # Check for maximum iterations
        if iteration_count > max_iterations:
            print('Maximum iterations exceeded. Optimization terminated.')
            break
            
        # Step 2: Compute feasible search direction using simplex method
        search_direction, direction_norm, simplex_beta = compute_search_direction(
            objective_gradient, constraint_gradient, convergence_tolerance, num_variables,
            current_point, num_active_constraints, lower_bounds, upper_bounds, constraint_tolerance
        )
        
        # Step 3: Check convergence criteria
        if abs(direction_norm) < convergence_tolerance or abs(simplex_beta) < convergence_tolerance:
            print(f"Convergence achieved: direction_norm={direction_norm:.2e}, beta={simplex_beta:.2e}")
            break
            
        # Step 4: Normalize search direction
        search_direction = search_direction / direction_norm
        
        # Step 5: Perform line search to find optimal step size
        step_size = perform_line_search(
            objective_function, objective_gradient, num_variables, num_constraint_functions,
            current_point, num_active_constraints, active_constraint_indices, lower_bounds,
            upper_bounds, search_direction, initial_point, max_line_search_steps,
            gradient_tolerance, convergence_tolerance
        )
        
        # Step 6: Update current point
        current_point = initial_point + step_size * search_direction
        function_gradient_values = objective_function(current_point)
        current_objective = function_gradient_values[0]
        initial_point = current_point.copy()  # Update reference point for next iteration
        
        # Step 7: Check for objective function convergence
        if abs(current_objective - previous_objective) < convergence_tolerance:
            convergence_counter += 1
            if convergence_counter == 2:  # Require two consecutive small changes
                print(f"Objective convergence achieved after {iteration_count} iterations")
                break
        else:
            convergence_counter = 0
            
        previous_objective = current_objective
        
        # Store iteration results for display
        iteration_results.append([iteration_count, current_point[0], current_point[1], current_objective])
    
    # ===== FINAL EVALUATION =====
    function_gradient_values = objective_function(current_point)
    final_objective = function_gradient_values[0]
    final_constraints = function_gradient_values[1] if len(function_gradient_values) > 1 else np.array([])
    num_active_constraints, active_constraint_indices = identify_active_constraints(
        num_constraint_functions, convergence_tolerance, final_constraints
    )
    
    optimal_point = current_point
    optimal_value = final_objective
    
    # Display results table
    results_dataframe = pd.DataFrame(
        iteration_results, 
        columns=['Iteration', 'x1', 'x2', 'Objective_Value']
    )
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(results_dataframe.to_string(index=False, float_format='%.6f'))
    
    end_time = time.time()
    print(f"\nOptimization completed in {end_time - start_time:.4f} seconds")
    print(f"Active constraints at solution: {num_active_constraints}")
    
    return optimal_point, optimal_value, iteration_count


def identify_active_constraints(num_constraints: int, tolerance: float, 
                               constraint_values: np.ndarray) -> Tuple[int, List[int]]:
    """
    Identify which constraints are active (nearly violated) at the current point.
    
    A constraint g_i(x) <= 0 is considered active if g_i(x) >= -tolerance.
    This function also handles the reordering of constraint indices to put
    active constraints first in the list.
    
    Args:
        num_constraints: Total number of constraint functions
        tolerance: Tolerance for determining constraint activity
        constraint_values: Current values of all constraints
        
    Returns:
        num_active: Number of active constraints
        active_indices: List of constraint indices (active ones first)
    """
    # Initialize all constraint indices (using 1-based indexing as in original MATLAB)
    active_indices = list(range(1, num_constraints + 1))
    num_active = 0
    
    # Check each constraint for activity and reorder indices
    for constraint_idx in range(num_constraints):
        if len(constraint_values) > constraint_idx and constraint_values[constraint_idx] > -tolerance:
            num_active += 1
            # Swap current constraint to the front of active list
            temp_index = active_indices[constraint_idx]
            active_indices[constraint_idx] = active_indices[num_active - 1]
            active_indices[num_active - 1] = temp_index
            
    return num_active, active_indices


def compute_search_direction(objective_grad_func: Callable, constraint_grad_func: Callable,
                           tolerance: float, num_variables: int, current_point: np.ndarray,
                           num_active_constraints: int, lower_bounds: np.ndarray,
                           upper_bounds: np.ndarray, constraint_tolerance: float) -> Tuple[np.ndarray, float, float]:
    """
    Compute a feasible search direction using the active set method.
    
    This function finds a direction that:
    1. Decreases the objective function (if possible)
    2. Maintains feasibility with respect to active constraints
    3. Respects variable bounds
    
    Args:
        objective_grad_func: Function that computes objective gradient
        constraint_grad_func: Function that computes constraint gradients
        tolerance: Numerical tolerance
        num_variables: Number of optimization variables
        current_point: Current iterate
        num_active_constraints: Number of active constraints
        lower_bounds: Lower bounds on variables
        upper_bounds: Upper bounds on variables
        constraint_tolerance: Tolerance for bound constraints
        
    Returns:
        search_direction: Computed search direction
        direction_norm: Norm of the search direction
        beta_value: Value from simplex method (optimality indicator)
    """
    # Compute and normalize objective gradient
    objective_gradient = objective_grad_func(current_point).reshape(1, -1)  # Row vector
    
    # Compute constraint gradients if there are active constraints
    if num_active_constraints > 0:
        constraint_gradient_matrix = constraint_grad_func(current_point).reshape(1, -1)
    
    # Normalize objective gradient
    objective_gradient = objective_gradient / np.linalg.norm(objective_gradient)
    
    # Normalize constraint gradients
    if num_active_constraints > 0:
        for constraint_idx in range(num_active_constraints):
            if constraint_idx < constraint_gradient_matrix.shape[0]:
                gradient_norm = np.sqrt(np.dot(constraint_gradient_matrix[constraint_idx, :], 
                                              constraint_gradient_matrix[constraint_idx, :]))
                constraint_gradient_matrix[constraint_idx, :] = constraint_gradient_matrix[constraint_idx, :] / gradient_norm
    
    # Add active bound constraints to the constraint matrix
    bound_constraint_rows = []
    
    # Check lower bounds
    for var_idx in range(num_variables):
        if lower_bounds[var_idx] - current_point[var_idx] + constraint_tolerance >= 0:
            num_active_constraints += 1
            bound_row = np.zeros(num_variables)
            bound_row[var_idx] = -1  # Lower bound: x >= lower_bound => -x <= -lower_bound
            bound_constraint_rows.append(bound_row)
    
    # Check upper bounds  
    for var_idx in range(num_variables):
        if current_point[var_idx] - upper_bounds[var_idx] + constraint_tolerance >= 0:
            num_active_constraints += 1
            bound_row = np.zeros(num_variables)
            bound_row[var_idx] = 1   # Upper bound: x <= upper_bound
            bound_constraint_rows.append(bound_row)
    
    # Handle case with no active constraints - use steepest descent
    if num_active_constraints == 0:
        beta_value = 1.0
        search_direction = -objective_gradient.flatten()
        direction_norm = np.linalg.norm(search_direction)
        return search_direction, direction_norm, beta_value
    
    # Combine all constraint gradients
    if bound_constraint_rows:
        if 'constraint_gradient_matrix' in locals() and constraint_gradient_matrix.shape[0] > 0:
            # Combine function constraints and bound constraints
            all_constraints = np.vstack([constraint_gradient_matrix, np.array(bound_constraint_rows)])
        else:
            # Only bound constraints
            all_constraints = np.array(bound_constraint_rows)
    else:
        # Only function constraints
        all_constraints = constraint_gradient_matrix
    
    # Solve for feasible direction using simplex method
    beta_value, search_direction = solve_direction_subproblem(
        num_active_constraints, num_variables, objective_gradient.flatten(), all_constraints
    )
    direction_norm = np.sqrt(np.dot(search_direction, search_direction))
    
    return search_direction, direction_norm, beta_value


def perform_line_search(objective_func: Callable, objective_grad_func: Callable,
                       num_variables: int, num_constraints: int, current_point: np.ndarray,
                       num_active: int, active_indices: List[int], lower_bounds: np.ndarray,
                       upper_bounds: np.ndarray, search_direction: np.ndarray,
                       reference_point: np.ndarray, max_steps: int,
                       gradient_tolerance: float, convergence_tolerance: float) -> float:
    """
    Perform line search to find the optimal step size along the search direction.
    
    This function finds the step size alpha such that:
    1. The new point remains feasible
    2. The objective function is minimized along the direction
    3. The gradient condition is satisfied
    
    Args:
        objective_func: Objective function
        objective_grad_func: Objective gradient function  
        num_variables: Number of variables
        num_constraints: Number of constraint functions
        current_point: Current iterate
        num_active: Number of active constraints
        active_indices: Indices of active constraints
        lower_bounds: Variable lower bounds
        upper_bounds: Variable upper bounds
        search_direction: Search direction vector
        reference_point: Reference point for line search
        max_steps: Maximum line search iterations
        gradient_tolerance: Gradient-based tolerance
        convergence_tolerance: General convergence tolerance
        
    Returns:
        optimal_step_size: The computed step size
    """
    # Find maximum feasible step size based on variable bounds
    max_feasible_step = 1e40
    bound_range = np.max(np.abs(upper_bounds - lower_bounds))
    
    for var_idx in range(num_variables):
        if abs(search_direction[var_idx]) * max_feasible_step > bound_range:
            if search_direction[var_idx] < 0:
                # Moving towards lower bound
                candidate_step = (lower_bounds[var_idx] - current_point[var_idx]) / search_direction[var_idx]
                if candidate_step < max_feasible_step:
                    max_feasible_step = candidate_step
            else:
                # Moving towards upper bound
                candidate_step = (upper_bounds[var_idx] - current_point[var_idx]) / search_direction[var_idx]
                if candidate_step < max_feasible_step:
                    max_feasible_step = candidate_step
    
    # Test point at maximum feasible step
    boundary_step = max_feasible_step
    test_point = reference_point + boundary_step * search_direction
    function_values = objective_func(test_point)
    test_objective = function_values[0]
    test_constraints = function_values[1] if len(function_values) > 1 else np.array([0])
    
    # Check constraint feasibility
    max_constraint_violation = np.max(test_constraints) if len(test_constraints) > 0 else 0
    constraint_satisfied = max_constraint_violation <= 0
    
    if constraint_satisfied:
        max_step_size = boundary_step
    else:
        # Find feasible step size using constraint satisfaction
        lower_step = 0
        feasible_step, feasible_objective = find_feasible_step(
            objective_func, lower_step, boundary_step, test_point, search_direction,
            reference_point, max_steps, convergence_tolerance
        )
        max_step_size = feasible_step
    
    # Check gradient condition at maximum step
    test_point = reference_point + max_step_size * search_direction
    objective_gradient = objective_grad_func(test_point)
    directional_derivative = np.dot(objective_gradient, search_direction)
    
    # If gradient condition satisfied, return maximum step
    if directional_derivative <= 0:
        return max_step_size
    
    # Otherwise, perform bisection search for optimal step size
    left_step = 0
    right_step = max_step_size
    step_range = right_step - left_step
    
    while (right_step - left_step) > convergence_tolerance * step_range:
        middle_step = (left_step + right_step) / 2
        test_point = reference_point + middle_step * search_direction
        objective_gradient = objective_grad_func(test_point)
        directional_derivative = np.dot(objective_gradient, search_direction)
        
        if directional_derivative == 0:
            break
        elif directional_derivative < 0:
            left_step = middle_step
        else:  # directional_derivative > 0
            right_step = middle_step
    
    return left_step


def find_feasible_step(objective_func: Callable, step_lower: float, step_upper: float,
                      test_point: np.ndarray, direction: np.ndarray, reference_point: np.ndarray,
                      max_iterations: int, tolerance: float) -> Tuple[float, float]:
    """
    Find a feasible step size using bisection method on constraint violations.
    
    This function finds the largest step size such that all constraints remain satisfied.
    
    Args:
        objective_func: Objective function
        step_lower: Lower bound on step size
        step_upper: Upper bound on step size  
        test_point: Current test point
        direction: Search direction
        reference_point: Reference point for stepping
        max_iterations: Maximum bisection iterations
        tolerance: Tolerance for constraint satisfaction
        
    Returns:
        feasible_step: Largest feasible step size
        feasible_objective: Objective value at feasible step
    """
    iteration_count = 0
    lower_objective = 0
    
    while True:
        middle_step = (step_lower + step_upper) / 2
        iteration_count += 1
        
        # Check iteration limit
        if iteration_count > max_iterations:
            return step_lower, lower_objective
        
        # Evaluate at middle step
        test_point = reference_point + middle_step * direction
        function_values = objective_func(test_point)
        middle_objective = function_values[0]
        constraint_values = function_values[1] if len(function_values) > 1 else np.array([0])
        
        max_constraint_violation = np.max(constraint_values) if len(constraint_values) > 0 else 0
        
        # Check if constraints are satisfied within tolerance
        if max_constraint_violation <= 0 and max_constraint_violation >= -tolerance:
            return middle_step, middle_objective
        
        # Update search interval based on feasibility
        if max_constraint_violation < 0:
            # Feasible - try larger step
            step_lower = middle_step
            lower_objective = middle_objective
        else:
            # Infeasible - try smaller step
            step_upper = middle_step


def solve_direction_subproblem(num_constraints: int, num_variables: int,
                             objective_gradient: np.ndarray, constraint_matrix: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Solve the direction-finding subproblem using the simplex method.
    
    This function solves the linear program:
        minimize c^T * d
        subject to A * d = 0  (active constraints)
                  |d_i| <= 1  (direction bounds)
    
    where c is the objective gradient and A contains the active constraint gradients.
    
    Args:
        num_constraints: Number of active constraints
        num_variables: Number of optimization variables
        objective_gradient: Gradient of objective function
        constraint_matrix: Matrix of active constraint gradients
        
    Returns:
        beta_value: Optimal value of the linear program
        direction_vector: Optimal direction vector
    """
    # Simplex method parameters
    big_m_parameter = 1e2  # Big M parameter for artificial variables
    num_tableau_rows = num_constraints + num_variables + 2
    num_simplex_variables = num_constraints + num_variables + 1
    
    # Initialize right-hand-side vector
    rhs_vector = np.zeros(num_tableau_rows)
    rhs_vector[0] = np.sum(objective_gradient)
    
    # Set up constraint right-hand sides
    for constraint_idx in range(num_constraints):
        rhs_vector[constraint_idx + 1] = 0
        for var_idx in range(num_variables):
            if constraint_idx < constraint_matrix.shape[0]:
                rhs_vector[constraint_idx + 1] += constraint_matrix[constraint_idx, var_idx]
    
    # Set bounds for direction variables
    for var_idx in range(num_constraints + 1, num_tableau_rows - 1):
        rhs_vector[var_idx] = 2  # |d_i| <= 1 becomes -1 <= d_i <= 1, or d_i + s_i = 1, d_i - s_i = -1
    
    # Initialize basis variables
    basis_variables = np.arange(num_variables + 2, num_variables + num_simplex_variables + 2)
    num_tableau_cols = num_variables + num_simplex_variables + 1
    
    # Add artificial variables for negative RHS values
    for var_idx in range(num_simplex_variables):
        if rhs_vector[var_idx] < 0:
            num_tableau_cols += 1
            basis_variables[var_idx] = -num_tableau_cols
    
    # Initialize simplex tableau
    simplex_tableau = np.zeros((num_tableau_rows, num_tableau_cols))
    
    # Set up objective row
    simplex_tableau[0, :num_variables] = objective_gradient
    simplex_tableau[0, num_variables] = 1
    
    # Set up constraint rows
    for constraint_idx in range(num_constraints):
        if constraint_idx < constraint_matrix.shape[0]:
            simplex_tableau[constraint_idx + 1, :num_variables] = constraint_matrix[constraint_idx, :]
        simplex_tableau[constraint_idx + 1, num_variables] = 1
    
    # Set up identity matrix for slack variables
    slack_column_index = 0
    for row_idx in range(num_constraints + 1, num_tableau_rows - 1):
        simplex_tableau[row_idx, slack_column_index] = 1
        slack_column_index += 1
    
    # Set up artificial variable column
    simplex_tableau[num_tableau_rows - 1, num_variables] = -1
    
    # Add slack variables
    for var_idx in range(num_simplex_variables):
        simplex_tableau[var_idx, num_variables + var_idx + 1] = 1
    
    # Handle artificial variables
    artificial_var_count = num_variables + num_simplex_variables + 1
    for var_idx in range(num_simplex_variables):
        if rhs_vector[var_idx] < 0:
            artificial_var_count += 1
            rhs_vector[var_idx] = -rhs_vector[var_idx]
            # Flip signs in row for artificial variable
            simplex_tableau[var_idx, :] = -simplex_tableau[var_idx, :]
            if artificial_var_count <= simplex_tableau.shape[1]:
                simplex_tableau[var_idx, artificial_var_count - 1] = 1
                simplex_tableau[num_tableau_rows - 1, artificial_var_count - 1] = big_m_parameter
    
    # Initialize artificial variables in objective
    for var_idx in range(num_simplex_variables):
        if basis_variables[var_idx] < 0:
            basis_variables[var_idx] = -basis_variables[var_idx]
            # Update objective row to eliminate artificial variables
            simplex_tableau[num_tableau_rows - 1, :] = (simplex_tableau[num_tableau_rows - 1, :] - 
                                                       big_m_parameter * simplex_tableau[var_idx, :])
            rhs_vector[num_tableau_rows - 1] = (rhs_vector[num_tableau_rows - 1] - 
                                               big_m_parameter * rhs_vector[var_idx])
    
    # ===== MAIN SIMPLEX ITERATIONS =====
    while True:
        # Check optimality: all reduced costs non-negative
        entering_variable_found = False
        for col_idx in range(num_tableau_cols):
            if simplex_tableau[num_tableau_rows - 1, col_idx] < 0:
                entering_variable_found = True
                break
        
        if not entering_variable_found:
            break  # Optimal solution found
        
        # Find entering variable (most negative reduced cost)
        most_negative_cost = big_m_parameter
        entering_variable_col = 0
        for col_idx in range(num_tableau_cols):
            if simplex_tableau[num_tableau_rows - 1, col_idx] < most_negative_cost:
                most_negative_cost = simplex_tableau[num_tableau_rows - 1, col_idx]
                entering_variable_col = col_idx
        
        # Find leaving variable (minimum ratio test)
        leaving_variable_found = False
        positive_pivot_count = 0
        leaving_variable_row = 0
        min_ratio = 0
        
        for row_idx in range(num_tableau_rows - 1):
            if simplex_tableau[row_idx, entering_variable_col] > 0:
                positive_pivot_count += 1
                current_ratio = rhs_vector[row_idx] / (simplex_tableau[row_idx, entering_variable_col] + 1e-10)
                
                if positive_pivot_count == 1:
                    min_ratio = current_ratio
                    leaving_variable_row = row_idx
                else:
                    if current_ratio < min_ratio:
                        min_ratio = current_ratio
                        leaving_variable_row = row_idx
                        
                leaving_variable_found = True
        
        # Update basis
        if leaving_variable_row < len(basis_variables):
            basis_variables[leaving_variable_row] = entering_variable_col
        
        if not leaving_variable_found:
            print('Unbounded objective function in direction subproblem.')
            break
        
        # Perform pivot operation
        pivot_element = simplex_tableau[leaving_variable_row, entering_variable_col]
        pivot_inverse = 1.0 / pivot_element
        
        # Scale pivot row
        rhs_vector[leaving_variable_row] = pivot_inverse * rhs_vector[leaving_variable_row]
        simplex_tableau[leaving_variable_row, :] = pivot_inverse * simplex_tableau[leaving_variable_row, :]
        
        # Eliminate entering variable from other rows
        for row_idx in range(num_tableau_rows):
            if row_idx != leaving_variable_row:
                elimination_multiplier = simplex_tableau[row_idx, entering_variable_col]
                simplex_tableau[row_idx, :] = (simplex_tableau[row_idx, :] - 
                                              elimination_multiplier * simplex_tableau[leaving_variable_row, :])
                rhs_vector[row_idx] = (rhs_vector[row_idx] - 
                                      elimination_multiplier * rhs_vector[leaving_variable_row])
    
    # ===== EXTRACT SOLUTION =====
    direction_vector = np.full(num_variables, -1.0)  # Default direction
    
    # Read solution from basis
    for basis_idx in range(min(num_simplex_variables, len(basis_variables))):
        for var_idx in range(num_variables):
            # Convert from 1-based to 0-based indexing
            if var_idx + 1 == basis_variables[basis_idx]:
                direction_vector[var_idx] = rhs_vector[basis_idx] - 1
    
    beta_value = rhs_vector[num_tableau_rows - 1]
    
    return beta_value, direction_vector


# ===== EXAMPLE USAGE AND TEST FUNCTIONS =====

def example_quadratic_function(x: np.ndarray) -> List[Union[float, np.ndarray]]:
    """
    Example quadratic optimization problem.
    
    Minimize: f(x) = (x1 - 3)^2 + (x2 - 4)^2
    Subject to: g(x) = x1 + x2 - 5 <= 0
    
    Args:
        x: Variable vector [x1, x2]
        
    Returns:
        [objective_value, constraint_values]
    """
    objective_value = (x[0] - 3)**2 + (x[1] - 4)**2
    constraint_values = np.array([x[0] + x[1] - 5])  # x1 + x2 <= 5
    return [objective_value, constraint_values]


def rosenbrock_constrained(x: np.ndarray) -> List[Union[float, np.ndarray]]:
    """
    Constrained Rosenbrock function example.
    
    Minimize: f(x) = 100*(x2 - x1^2)^2 + (1 - x1)^2
    Subject to: g1(x) = x1^2 + x2^2 - 2 <= 0
               g2(x) = x1 + x2 - 1 <= 0
    
    Args:
        x: Variable vector [x1, x2]
        
    Returns:
        [objective_value, constraint_values]
    """
    objective_value = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    constraint_values = np.array([
        x[0]**2 + x[1]**2 - 2,  # Circle constraint
        x[0] + x[1] - 1         # Linear constraint
    ])
    return [objective_value, constraint_values]


if __name__ == "__main__":
    print("="*80)
    print("CONSTRAINED OPTIMIZATION USING ACTIVE SET METHOD")
    print("="*80)
    
    # Test 1: Simple quadratic problem
    print("\nTest 1: Quadratic function with linear constraint")
    print("Minimize: f(x) = (x1-3)² + (x2-4)²")
    print("Subject to: x1 + x2 ≤ 5")
    
    optimal_point, optimal_value, iterations = constrained_optimization(example_quadratic_function)
    
    print(f"\nResults:")
    print(f"Optimal point: x* = [{optimal_point[0]:.6f}, {optimal_point[1]:.6f}]")
    print(f"Optimal value: f* = {optimal_value:.6f}")
    print(f"Iterations: {iterations}")
    
    # Verify constraint satisfaction
    constraint_check = optimal_point[0] + optimal_point[1] - 5
    print(f"Constraint check (should be ≤ 0): g* = {constraint_check:.6f}")
    
    # Test 2: Constrained Rosenbrock problem
    print("\n" + "="*60)
    print("\nTest 2: Constrained Rosenbrock function")
    print("Minimize: f(x) = 100*(x2-x1²)² + (1-x1)²")
    print("Subject to: x1² + x2² ≤ 2")
    print("           x1 + x2 ≤ 1")
    
    try:
        optimal_point_2, optimal_value_2, iterations_2 = constrained_optimization(rosenbrock_constrained)
        
        print(f"\nResults:")
        print(f"Optimal point: x* = [{optimal_point_2[0]:.6f}, {optimal_point_2[1]:.6f}]")
        print(f"Optimal value: f* = {optimal_value_2:.6f}")
        print(f"Iterations: {iterations_2}")
        
        # Verify constraint satisfaction
        g1 = optimal_point_2[0]**2 + optimal_point_2[1]**2 - 2
        g2 = optimal_point_2[0] + optimal_point_2[1] - 1
        print(f"Constraint 1 check: g1* = {g1:.6f}")
        print(f"Constraint 2 check: g2* = {g2:.6f}")
        
    except Exception as e:
        print(f"Error in Test 2: {e}")
        print("This may indicate the need for problem-specific gradient functions.")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print("\nNotes:")
    print("- The gradient functions (objective_gradient and constraint_gradient)")
    print("  need to be customized for each specific optimization problem.")
    print("- The algorithm uses the active set method with simplex-based direction finding.")
    print("- Convergence is based on both direction norm and objective function changes.")
    print("- For best results, ensure your problem is well-scaled and gradients are accurate.")