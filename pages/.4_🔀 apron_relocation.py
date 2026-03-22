import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations, product
import hashlib
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time, date
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import importlib.util
from utils.universal_masterplan_new import MasterplanInput

# Import apply_css from 4_🔀 Relocation_under.py
relocation_under_path = os.path.join(os.path.dirname(__file__), "3_🔀 Airline_Relocation.py")
spec = importlib.util.spec_from_file_location("relocation_new", relocation_under_path)
relocation_under_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relocation_under_module)
apply_css = relocation_under_module.apply_css

# Global random seed setting
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ============================================================================
# Automatic mooring assignment system
# ============================================================================

# ============================================================================
# Modular goal and constraint system
# ============================================================================

class Objective:
    """Base class that defines the target function"""
    def __init__(self, name, weight=1.0, maximize=True):
        self.name = name
        self.weight = weight
        self.maximize = maximize  # True: maximize, False: minimize
    
    def calculate(self, aprons, flights_df, assignments, num_used_aprons):
        """
        Calculate objective function value
        Returns: (score, metadata)
        - score: score value
        - metadata: Additional information (KPI etc.)
        """
        raise NotImplementedError("Subclass must implement calculate method")
    
    def get_score(self, aprons, flights_df, assignments, num_used_aprons):
        """Returns the weighted final score"""
        score, metadata = self.calculate(aprons, flights_df, assignments, num_used_aprons)
        weighted_score = score * self.weight
        if not self.maximize:
            weighted_score = -weighted_score  # Minimize converts to negative
        return weighted_score, metadata


class MinimizeApronsObjective(Objective):
    """Goal of minimizing the number of landing pads"""
    def __init__(self, weight=1.0):
        super().__init__("minimize_aprons", weight, maximize=False)
    
    def calculate(self, aprons, flights_df, assignments, num_used_aprons):
        return -num_used_aprons, {"num_used_aprons": num_used_aprons}


class AirlineGroupingObjective(Objective):
    """Goal of maximizing airline grouping"""
    def __init__(self, weight=0.5):
        super().__init__("airline_grouping", weight, maximize=True)
    
    def calculate(self, aprons, flights_df, assignments, num_used_aprons):
        score = 0
        total_pairs = 0
        
        for apron_id, flights in aprons.items():
            if len(flights) < 2:
                continue
            
            for i in range(len(flights) - 1):
                current_flight_id = flights[i][2]
                next_flight_id = flights[i+1][2]
                
                current_flight = flights_df[flights_df['flight_id'] == current_flight_id].iloc[0]
                next_flight = flights_df[flights_df['flight_id'] == next_flight_id].iloc[0]
                
                if current_flight['airline'] == next_flight['airline']:
                    score += 1
                total_pairs += 1
        
        return score, {"airline_pairs": score, "total_pairs": total_pairs}


class AvoidSpecificApronsObjective(Objective):
    """Specific approach plane avoidance goal"""
    def __init__(self, avoid_aprons, weight=1.0):
        """
        Args:
            avoid_aprons: Approach plane to avoid ID list (0-indexed)
            weight: weight
        """
        super().__init__("avoid_specific_aprons", weight, maximize=False)
        self.avoid_aprons = set(avoid_aprons)
    
    def calculate(self, aprons, flights_df, assignments, num_used_aprons):
        penalty = 0
        for apron_id in self.avoid_aprons:
            penalty += len(aprons.get(apron_id, []))
        return -penalty, {"avoided_aprons": list(self.avoid_aprons), "penalty": penalty}


class PreferSpecificApronsObjective(Objective):
    """Preferred target for specific approach plane"""
    def __init__(self, prefer_aprons, weight=0.5):
        """
        Args:
            prefer_aprons: Preferred landing gear ID list (0-indexed)
            weight: weight
        """
        super().__init__("prefer_specific_aprons", weight, maximize=True)
        self.prefer_aprons = set(prefer_aprons)
    
    def calculate(self, aprons, flights_df, assignments, num_used_aprons):
        score = 0
        for apron_id in self.prefer_aprons:
            score += len(aprons.get(apron_id, []))
        return score, {"preferred_aprons": list(self.prefer_aprons), "score": score}


class HardConstraint:
    """hard constraints (must be satisfied)"""
    def __init__(self, name):
        self.name = name
    
    def is_valid(self, aprons, flights_df, assignments):
        """
        Check whether constraints are satisfied
        Returns: True if valid, False otherwise
        """
        raise NotImplementedError("Subclass must implement is_valid method")
    
    def filter_valid_aprons(self, flight_id, aprons, flights_df, num_aprons):
        """
        Returns a list of valid landing pads for a specific flight
        Returns: list of valid apron IDs
        """
        return list(range(num_aprons))  # Basically, all landing gears are allowed


class AircraftTypeConstraint(HardConstraint):
    """Approach length restrictions depending on aircraft type"""
    def __init__(self, aircraft_apron_map):
        """
        Args:
            aircraft_apron_map: {aircraft_type: [allowed_apron_ids]}
        """
        super().__init__("aircraft_type_constraint")
        self.aircraft_apron_map = aircraft_apron_map
    
    def filter_valid_aprons(self, flight_id, aprons, flights_df, num_aprons):
        flight = flights_df[flights_df['flight_id'] == flight_id].iloc[0]
        aircraft_type = flight['aircraft_type']
        allowed = self.aircraft_apron_map.get(aircraft_type, list(range(num_aprons)))
        return allowed


class ScoreCalculator:
    """A class that integrates all goals and constraints to calculate the score."""
    def __init__(self, objectives=None, constraints=None):
        """
        Args:
            objectives: Objective object list
            constraints: HardConstraint object list
        """
        self.objectives = objectives or []
        self.constraints = constraints or []
    
    def add_objective(self, objective):
        """Add goal"""
        self.objectives.append(objective)
    
    def add_constraint(self, constraint):
        """Add constraints"""
        self.constraints.append(constraint)
    
    def calculate_total_score(self, aprons, flights_df, assignments, num_used_aprons):
        """
        Total score calculation
        Returns: (total_score, metadata_dict)
        """
        total_score = 0
        metadata = {}
        
        for objective in self.objectives:
            score, obj_metadata = objective.get_score(aprons, flights_df, assignments, num_used_aprons)
            total_score += score
            metadata[objective.name] = obj_metadata
        
        return total_score, metadata
    
    def check_constraints(self, aprons, flights_df, assignments):
        """Check whether all constraints are satisfied"""
        for constraint in self.constraints:
            if not constraint.is_valid(aprons, flights_df, assignments):
                return False, constraint.name
        return True, None
    
    def get_valid_aprons(self, flight_id, aprons, flights_df, num_aprons):
        """Returns a list of landing pads that satisfy all constraints for a specific flight"""
        valid_aprons = set(range(num_aprons))
        
        for constraint in self.constraints:
            constraint_valid = constraint.filter_valid_aprons(flight_id, aprons, flights_df, num_aprons)
            valid_aprons = valid_aprons.intersection(set(constraint_valid))
        
        return list(valid_aprons)

def generate_flight_dummy_data(num_flights=20):
    """Generate flight dummy data"""
    flights = []
    airlines = ['KE', 'OZ', '7C', 'LJ', 'TW', 'BX', 'ZE']
    
    # Divide the day into 24 hours and distribute them by time zone
    base_time = datetime(2024, 1, 18, 6, 0)  # Starts at 6 AM
    
    for i in range(num_flights):
        # Arrival time: 6 o'clock~22Random between poems
        arrival_hour = np.random.randint(6, 22)
        arrival_minute = np.random.choice([0, 15, 30, 45])
        arrival_time = base_time.replace(hour=arrival_hour, minute=arrival_minute)
        
        # Duration of stay: 1 hour~4hour
        stay_duration = np.random.choice([40, 50, 60, 70, 90])  # minute by minute
        departure_time = arrival_time + pd.Timedelta(minutes=stay_duration)
        
        # Adjust the departure time so that it does not exceed midnight
        if departure_time.hour >= 24:
            departure_time = departure_time.replace(hour=23, minute=59)
        
        airline = np.random.choice(airlines)
        flight_id = f"FL{i+1:03d}"
        
        flight = {
            'flight_id': flight_id,
            'airline': airline,
            'flight_code': f"{airline}-{flight_id}",  # airlineclass flight_idA column combining
            'arrival_time': arrival_time,
            'departure_time': departure_time,
            'aircraft_type': np.random.choice(['A320', 'A321', 'B737', 'B777', 'A330']),
        }
        flights.append(flight)
    
    df = pd.DataFrame(flights)
    # Sort by time
    df = df.sort_values('arrival_time').reset_index(drop=True)
    return df


def initial_assignment(flights_df, num_aprons=5):
    """Draft Assignment: Chronologically First-Fit Algorithm usage"""
    # Stores the usage time of each landing terminal
    aprons = {i: [] for i in range(num_aprons)}  # {apron_id: [(start, end), ...]}
    assignments = {}  # {flight_id: apron_id}
    failed_assignments = []  # Flight assignment failed ID list
    
    for idx, flight in flights_df.iterrows():
        flight_id = flight['flight_id']
        arrival = flight['arrival_time']
        departure = flight['departure_time']
        
        # Find the first available landing pad
        assigned = False
        for apron_id in range(num_aprons):
            # Check time overlap
            conflict = False
            for existing_start, existing_end in aprons[apron_id]:
                if not (departure <= existing_start or arrival >= existing_end):
                    conflict = True
                    break
            
            if not conflict:
                aprons[apron_id].append((arrival, departure))
                assignments[flight_id] = apron_id
                assigned = True
                break
        
        if not assigned:
            # If all landing pads are in use, assignment is treated as failure.
            failed_assignments.append(flight_id)
    
    return assignments, aprons, failed_assignments


def calculate_airline_grouping_score(aprons, flights_df):
    """
    Same airline adjacent assignment score calculation (Maintain backwards compatibility)
    Returns the number of pairs of identical airlines that are adjacent to each other in chronological order at each landing site.
    """
    score = 0
    total_pairs = 0
    
    for apron_id, flights in aprons.items():
        if len(flights) < 2:
            continue
        
        # Flights sorted chronologically (Already sorted)
        for i in range(len(flights) - 1):
            current_flight_id = flights[i][2]
            next_flight_id = flights[i+1][2]
            
            current_flight = flights_df[flights_df['flight_id'] == current_flight_id].iloc[0]
            next_flight = flights_df[flights_df['flight_id'] == next_flight_id].iloc[0]
            
            # If they are close together front to back and on the same plane, your score increases.
            if current_flight['airline'] == next_flight['airline']:
                score += 1
            total_pairs += 1
    
    return score, total_pairs


def create_default_score_calculator(weight_apron=1.0, weight_airline=0.5, 
                                     avoid_aprons=None, prefer_aprons=None,
                                     aircraft_constraints=None):
    """
    basic ScoreCalculator generation (Maintain backwards compatibility)
    
    Args:
        weight_apron: Abutment field minimization weight
        weight_airline: Airline grouping weight
        avoid_aprons: List of approach planes to avoid (0-indexed)
        prefer_aprons: List of preferred landing gears (0-indexed)
        aircraft_constraints: Aircraft type constraints {aircraft_type: [allowed_aprons]}
    """
    calculator = ScoreCalculator()
    
    # Add default goal
    calculator.add_objective(MinimizeApronsObjective(weight=weight_apron))
    calculator.add_objective(AirlineGroupingObjective(weight=weight_airline))
    
    # Additional Goals
    if avoid_aprons:
        calculator.add_objective(AvoidSpecificApronsObjective(avoid_aprons, weight=1.0))
    
    if prefer_aprons:
        calculator.add_objective(PreferSpecificApronsObjective(prefer_aprons, weight=0.5))
    
    # Add constraints
    if aircraft_constraints:
        calculator.add_constraint(AircraftTypeConstraint(aircraft_constraints))
    
    return calculator


# ============================================================================
# Example usage: Helper function that converts natural language conditions into code
# ============================================================================

def parse_natural_language_constraint(natural_language_text, num_aprons=5):
    """
    By parsing natural language conditions ScoreCalculatorConvert it to a form that can be added to
    
    example:
        "Apron4, Apron5I want to allocate as little as possible to"
        → {"type": "avoid_aprons", "aprons": [3, 4], "weight": 1.0}
        
        "KE The airline Apron1, Apron2to Assign it줘"
        → {"type": "prefer_airline_aprons", "airline": "KE", "aprons": [0, 1], "weight": 0.8}
    
    Args:
        natural_language_text: Natural language conditional text
        num_aprons: Number of landing planes
    
    Returns:
        dict: Parsed condition information
    """
    # actually LLM APIParse it by calling
    # Only a simple example is provided here.
    
    text_lower = natural_language_text.lower()
    
    # Apron avoidance condition
    if "less" in text_lower or "evasion" in text_lower or "avoid" in text_lower:
        # Apron number extraction (1-indexedcast 0-indexedconvert to)
        import re
        apron_numbers = re.findall(r'apron\s*(\d+)', text_lower, re.IGNORECASE)
        if apron_numbers:
            aprons = [int(n) - 1 for n in apron_numbers]  # 1-indexed → 0-indexed
            return {
                "type": "avoid_aprons",
                "aprons": aprons,
                "weight": 1.0
            }
    
    # Apron Preferred Conditions
    if "preference" in text_lower or "Assign it" in text_lower:
        import re
        apron_numbers = re.findall(r'apron\s*(\d+)', text_lower, re.IGNORECASE)
        if apron_numbers:
            aprons = [int(n) - 1 for n in apron_numbers]
            return {
                "type": "prefer_aprons",
                "aprons": aprons,
                "weight": 0.5
            }
    
    return None


def add_constraint_from_natural_language(score_calculator, natural_language_text, num_aprons=5):
    """
    natural language conditions ScoreCalculatoradd to
    
    Args:
        score_calculator: ScoreCalculator object
        natural_language_text: Natural language conditional text
        num_aprons: Number of landing planes
    
    Returns:
        bool: Success or failure
    """
    constraint_info = parse_natural_language_constraint(natural_language_text, num_aprons)
    
    if constraint_info is None:
        return False
    
    if constraint_info["type"] == "avoid_aprons":
        objective = AvoidSpecificApronsObjective(
            avoid_aprons=constraint_info["aprons"],
            weight=constraint_info["weight"]
        )
        score_calculator.add_objective(objective)
        return True
    
    elif constraint_info["type"] == "prefer_aprons":
        objective = PreferSpecificApronsObjective(
            prefer_aprons=constraint_info["aprons"],
            weight=constraint_info["weight"]
        )
        score_calculator.add_objective(objective)
        return True
    
    return False


# ============================================================================
# Example code of use (annotation)
# ============================================================================

"""
# Example 1: Basic use (Maintain backwards compatibility)____________________________________________________
result = greedy_optimization(
    flights_df, initial_assignments, failed_assignments, 
    num_aprons=5, weight_apron=1.0, weight_airline=0.5
)

# example 2: Apron Add avoidance condition____________________________________________________
result = greedy_optimization(
    flights_df, initial_assignments, failed_assignments,
    num_aprons=5, avoid_aprons=[3, 4]  # Apron 4, 5 evasion
)

# Example 3: Custom ScoreCalculator use____________________________________________________ 
calculator = ScoreCalculator()
calculator.add_objective(MinimizeApronsObjective(weight=1.0))
calculator.add_objective(AirlineGroupingObjective(weight=0.5))
calculator.add_objective(AvoidSpecificApronsObjective([3, 4], weight=1.0))
calculator.add_constraint(AircraftTypeConstraint({
    'B777': [0, 1],  # B777silver Apron 1, 2Only possible
    'A330': [2, 3]   # A330silver Apron 3, 4Only possible
}))


result = greedy_optimization(
    flights_df, initial_assignments, failed_assignments,
    num_aprons=5, score_calculator=calculator
    ) # score_calculatorvalue greedy_optimizationIf you receive this variable, there is no need to enter additional conditions.








# Example 4: Adding natural language conditions____________________________________________________
calculator = create_default_score_calculator(weight_apron=1.0, weight_airline=0.5)
add_constraint_from_natural_language(
    calculator, 
    "Apron4, Apron5I want to allocate as little as possible to",
    num_aprons=5
)

result = greedy_optimization(
    flights_df, initial_assignments, failed_assignments,
    num_aprons=5, score_calculator=calculator
)


# Example 5: Adding a new objective function (expansion)____________________________________________________
class MinimizeDistanceObjective(Objective):
    \"\"\"The goal is to minimize the distance between landing pads\"\"\"
    def __init__(self, distance_matrix, weight=0.3):
        super().__init__("minimize_distance", weight, maximize=False)
        self.distance_matrix = distance_matrix
    
    def calculate(self, aprons, flights_df, assignments, num_used_aprons):
        total_distance = 0
        for flight_id, apron_id in assignments.items():
            # Distance calculation logic
            pass
        return -total_distance, {"total_distance": total_distance}

calculator.add_objective(MinimizeDistanceObjective(distance_matrix, weight=0.3))
"""


def greedy_optimization(flights_df, initial_assignments, failed_assignments, num_aprons=5, 
                        weight_apron=1.0, weight_airline=0.5, score_calculator=None,
                        avoid_aprons=None, prefer_aprons=None, aircraft_constraints=None):
    """
    Greedy Multi-objective optimization with algorithms (Modular version)
    
    Args:
        flights_df: Flight dataframe
        initial_assignments: Initial Placement Results
        failed_assignments: List of failed flights
        num_aprons: Number of landing planes
        weight_apron: Abutment field minimization weight (backwards compatibility)
        weight_airline: Airline grouping weight (backwards compatibility)
        score_calculator: ScoreCalculator object (NoneIf this is the default creation)
        avoid_aprons: List of approach planes to avoid
        prefer_aprons: List of preferred landing gears
        aircraft_constraints: Aircraft type constraints
    
    Returns:
        (optimized_assignments, optimized_failed, final_num_used, initial_num_used, 
         final_airline_score, total_pairs, metadata)
    """
    # ScoreCalculator generation (If not, create default)
    if score_calculator is None:
        score_calculator = create_default_score_calculator(
            weight_apron=weight_apron,
            weight_airline=weight_airline,
            avoid_aprons=avoid_aprons,
            prefer_aprons=prefer_aprons,
            aircraft_constraints=aircraft_constraints
        )
    # Check the currently used landing gear
    used_aprons = set(initial_assignments.values())
    num_used = len(used_aprons)
    
    # Reconstruct the usage time of each landing terminal
    aprons = {i: [] for i in range(num_aprons)}
    for flight_id, apron_id in initial_assignments.items():
        flight = flights_df[flights_df['flight_id'] == flight_id].iloc[0]
        aprons[apron_id].append((flight['arrival_time'], flight['departure_time'], flight_id))
    
    # Sort the time zone of each landing station
    for apron_id in aprons:
        aprons[apron_id].sort(key=lambda x: x[0])
    
    # Optimization: Minimize the number of taxi stands in use and maximize airline grouping.
    optimized_assignments = initial_assignments.copy()
    optimized_failed = failed_assignments.copy()
    
    # Retry Assignment Failed Flight (Consider airline grouping)
    retry_flights = []
    for flight_id in optimized_failed:
        flight = flights_df[flights_df['flight_id'] == flight_id].iloc[0]
        arrival = flight['arrival_time']
        departure = flight['departure_time']
        airline = flight['airline']
        
        # Find available landing gear (Priority consideration for airline grouping)
        assigned = False
        best_apron = None
        best_score = -float('inf')
        
        # Check only the landing gear that satisfies the constraints
        valid_aprons = score_calculator.get_valid_aprons(flight_id, aprons, flights_df, num_aprons)
        
        for apron_id in valid_aprons:
            conflict = False
            for existing_start, existing_end, _ in aprons[apron_id]:
                if not (departure <= existing_start or arrival >= existing_end):
                    conflict = True
                    break
            
            if not conflict:
                # Airline grouping score calculation (Temporarily add to check score)
                temp_aprons = {k: v.copy() for k, v in aprons.items()}
                temp_aprons[apron_id].append((arrival, departure, flight_id))
                temp_aprons[apron_id].sort(key=lambda x: x[0])
                
                # ScoreCalculatorCalculate score using
                temp_assignments = optimized_assignments.copy()
                temp_assignments[flight_id] = apron_id
                temp_used = len([a for a in apron_usage.values() if a > 0])
                if apron_id not in [a for a in range(num_aprons) if apron_usage[a] > 0]:
                    temp_used += 1
                
                temp_score, _ = score_calculator.calculate_total_score(
                    temp_aprons, flights_df, temp_assignments, temp_used
                )
                
                # Choose if your score is better
                if temp_score > best_score:
                    best_score = temp_score
                    best_apron = apron_id
        
        if best_apron is not None:
            aprons[best_apron].append((arrival, departure, flight_id))
            aprons[best_apron].sort(key=lambda x: x[0])
            optimized_assignments[flight_id] = best_apron
            retry_flights.append(flight_id)
            assigned = True
    
    # Remove flights with successful retry
    for flight_id in retry_flights:
        if flight_id and flight_id in optimized_failed:
            optimized_failed.remove(flight_id)
    
    # Attempting to move from a less frequently used landing pad to another landing pad
    apron_usage = {i: len(aprons[i]) for i in range(num_aprons)}
    sorted_aprons = sorted(range(num_aprons), key=lambda x: apron_usage[x])
    
    # Perform optimization in multiple rounds
    max_iterations = 10
    for iteration in range(max_iterations):
        changed = False
        
        # Attempt reassignment starting with a low usage approach terminal.
        for source_apron in sorted_aprons:
            if apron_usage[source_apron] == 0:
                continue
            
            # An attempt was made to move flights from this landing pad to another landing pad.
            flights_to_reassign = aprons[source_apron].copy()
            
            for arrival, departure, flight_id in flights_to_reassign:
                # Priority is given to approaches that are already in use.
                # Check the heavily used landing gear first. (If already in use, more likely to be moved)
                target_candidates = sorted(
                    [i for i in range(num_aprons) if i != source_apron],
                    key=lambda x: apron_usage[x],
                    reverse=True
                )
                
                for target_apron in target_candidates:
                    # Check constraints
                    valid_aprons = score_calculator.get_valid_aprons(flight_id, aprons, flights_df, num_aprons)
                    if target_apron not in valid_aprons:
                        continue
                    
                    # Check time overlap
                    conflict = False
                    for existing_start, existing_end, _ in aprons[target_apron]:
                        if not (departure <= existing_start or arrival >= existing_end):
                            conflict = True
                            break
                    
                    if not conflict:
                        # Calculate your score before moving
                        current_used = len([a for a in apron_usage.values() if a > 0])
                        current_score, _ = score_calculator.calculate_total_score(
                            aprons, flights_df, optimized_assignments, current_used
                        )
                        
                        # movement simulation
                        temp_aprons = {k: [f for f in v] for k, v in aprons.items()}
                        temp_aprons[source_apron].remove((arrival, departure, flight_id))
                        temp_aprons[target_apron].append((arrival, departure, flight_id))
                        temp_aprons[target_apron].sort(key=lambda x: x[0])
                        
                        # Calculate usage after moving
                        temp_usage = {i: len(temp_aprons[i]) for i in range(num_aprons)}
                        new_used = len([a for a in temp_usage.values() if a > 0])
                        
                        # Update assignments after move
                        temp_assignments = optimized_assignments.copy()
                        temp_assignments[flight_id] = target_apron
                        
                        # Score after move
                        new_score, _ = score_calculator.calculate_total_score(
                            temp_aprons, flights_df, temp_assignments, new_used
                        )
                        
                        # Go if score improves or stays the same (Multi-objective optimization)
                        if new_score >= current_score:
                            # perform movement
                            aprons[source_apron].remove((arrival, departure, flight_id))
                            aprons[target_apron].append((arrival, departure, flight_id))
                            aprons[target_apron].sort(key=lambda x: x[0])
                            optimized_assignments[flight_id] = target_apron
                            apron_usage[source_apron] -= 1
                            apron_usage[target_apron] += 1
                            changed = True
                            break
        
        # Quit when no more changes are made
        if not changed:
            break
    
    # Calculation of the number of terminal terminals in use
    final_used_aprons = {apron_id for apron_id in optimized_assignments.values() if apron_usage[apron_id] > 0}
    final_num_used = len(final_used_aprons)
    
    # Final score and metadata calculation
    final_score, final_metadata = score_calculator.calculate_total_score(
        aprons, flights_df, optimized_assignments, final_num_used
    )
    
    # Airline grouping scores for backward compatibility
    final_airline_score = final_metadata.get('airline_grouping', {}).get('airline_pairs', 0)
    total_pairs = final_metadata.get('airline_grouping', {}).get('total_pairs', 0)
    
    return optimized_assignments, optimized_failed, final_num_used, num_used, final_airline_score, total_pairs, final_metadata


def visualize_assignments(flights_df, assignments, num_aprons=5):
    """Visualize assignment results"""
    # Gantt Visualize with chart styles
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3[:num_aprons]
    
    # Standard time setting (Start time of first flight arrival time)
    base_time = flights_df['arrival_time'].min().replace(hour=6, minute=0)
    
    for apron_id in range(num_aprons):
        apron_flights = flights_df[flights_df['flight_id'].isin(
            [fid for fid, aid in assignments.items() if aid == apron_id]
        )].copy()
        
        if len(apron_flights) == 0:
            continue
        
        # convert time to number (minute by minute, base_time standard)
        apron_flights = apron_flights.sort_values('arrival_time')
        for idx, flight in apron_flights.iterrows():
            # base_time Calculate elapsed time based on (minute)
            start_min = (flight['arrival_time'] - base_time).total_seconds() / 60
            duration = (flight['departure_time'] - flight['arrival_time']).total_seconds() / 60
            
            # Create time string
            arrival_str = flight['arrival_time'].strftime('%H:%M')
            departure_str = flight['departure_time'].strftime('%H:%M')
            
            fig.add_trace(go.Bar(
                x=[duration],
                y=[apron_id],
                base=[start_min],
                orientation='h',
                name=f"Apron {apron_id+1}",
                marker_color=colors[apron_id],
                text=f"{flight['flight_code']}<br>{arrival_str}-{departure_str}",
                textposition='inside',
                textfont=dict(size=9),
                hovertemplate=f"<b>{flight['flight_code']}</b><br>" +
                            f"Airline: {flight['airline']}<br>" +
                            f"Aircraft: {flight['aircraft_type']}<br>" +
                            f"Arrival: {arrival_str}<br>" +
                            f"Departure: {departure_str}<br>" +
                            f"<extra></extra>"
            ))
    
    # xCreate axis time labels (1time interval)
    max_time = (flights_df['departure_time'].max() - base_time).total_seconds() / 60
    hour_ticks = list(range(0, int(max_time) + 60, 60))
    hour_labels = [(base_time + pd.Timedelta(minutes=t)).strftime('%H:%M') for t in hour_ticks]
    
    fig.update_layout(
        title='Marina allocation results (Gantt Chart)',
        xaxis_title='hour',
        yaxis_title='landing port',
        xaxis=dict(
            tickmode='array',
            tickvals=hour_ticks,
            ticktext=hour_labels
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,
            ticktext=[f"Apron {i+1}" for i in range(num_aprons)],
            tickvals=list(range(num_aprons))
        ),
        height=max(400, num_aprons * 80),
        barmode='overlay',
        showlegend=False
    )
    
    return fig


def render_apron_assignment_system():
    """Automatic mooring assignment system UI rendering"""
    st.markdown("## 🛫 Automatic mooring assignment system")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        num_flights = st.number_input("number of flights", min_value=5, max_value=50, value=20, step=1)
        num_aprons = st.number_input("Number of landing planes", min_value=3, max_value=10, value=5, step=1)
    
    with col2:
        if st.button("🔄 Generate dummy data", use_container_width=True):
            flights_df = generate_flight_dummy_data(num_flights)
            st.session_state['apron_flights_df'] = flights_df
            st.session_state['apron_assignments'] = None
            st.success(f"✅ {num_flights}Flight data has been generated!")
    
    if 'apron_flights_df' in st.session_state:
        flights_df = st.session_state['apron_flights_df']
        
        st.markdown("### 📋 flight data")
        st.dataframe(flights_df[['flight_code', 'airline', 'aircraft_type', 
                                 'arrival_time', 'departure_time']].style.format({
            'arrival_time': lambda x: x.strftime('%H:%M'),
            'departure_time': lambda x: x.strftime('%H:%M')
        }), use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📝 Run Draft Assignment", use_container_width=True):
                with st.spinner("Assigning draft..."):
                    initial_assignments, aprons, failed_assignments = initial_assignment(flights_df, num_aprons)
                    st.session_state['apron_assignments'] = initial_assignments
                    st.session_state['apron_failed'] = failed_assignments
                    st.session_state['apron_initial'] = True
                    st.session_state['apron_optimized'] = False
                    
                    used_aprons = len(set(initial_assignments.values()))
                    success_count = len(initial_assignments)
                    failed_count = len(failed_assignments)
                    
                    # Calculating airline grouping scores for draft assignments
                    aprons_for_kpi = {i: [] for i in range(num_aprons)}
                    for flight_id, apron_id in initial_assignments.items():
                        flight = flights_df[flights_df['flight_id'] == flight_id].iloc[0]
                        aprons_for_kpi[apron_id].append((flight['arrival_time'], flight['departure_time'], flight_id))
                    for apron_id in aprons_for_kpi:
                        aprons_for_kpi[apron_id].sort(key=lambda x: x[0])
                    
                    initial_airline_score, initial_total_pairs = calculate_airline_grouping_score(aprons_for_kpi, flights_df)
                    st.session_state['apron_airline_score'] = initial_airline_score
                    st.session_state['apron_total_pairs'] = initial_total_pairs
                    
                    messages = []
                    if failed_count > 0:
                        messages.append(f"success: {success_count}dog, failure: {failed_count}dog")
                    messages.append(f"The landing gear used: {used_aprons}dog")
                    if initial_airline_score > 0:
                        messages.append(f"Airline grouping: {initial_airline_score}dog")
                    
                    if failed_count > 0:
                        st.warning(f"⚠️ Draft assignment completed! {' | '.join(messages)}")
                    else:
                        st.success(f"✅ Draft assignment completed! {' | '.join(messages)}")
        
        with col2:
            weight_col1, weight_col2 = st.columns(2)
            with weight_col1:
                weight_apron = st.slider("Abutment field minimization weight", 0.0, 2.0, 1.0, 0.1, key="weight_apron")
            with weight_col2:
                weight_airline = st.slider("Airline grouping weight", 0.0, 2.0, 0.5, 0.1, key="weight_airline")
            
            if st.button("⚡ Greedy Optimization run", use_container_width=True, 
                        disabled='apron_assignments' not in st.session_state):
                with st.spinner("Optimizing..."):
                    if 'apron_assignments' in st.session_state:
                        failed = st.session_state.get('apron_failed', [])
                        optimized_assignments, optimized_failed, final_used, initial_used, airline_score, total_pairs, metadata = greedy_optimization(
                            flights_df, st.session_state['apron_assignments'], failed, num_aprons,
                            weight_apron, weight_airline
                        )
                        st.session_state['apron_assignments'] = optimized_assignments
                        st.session_state['apron_failed'] = optimized_failed
                        st.session_state['apron_optimized'] = True
                        st.session_state['apron_airline_score'] = airline_score
                        st.session_state['apron_total_pairs'] = total_pairs
                        
                        improvement = initial_used - final_used
                        success_count = len(optimized_assignments)
                        failed_count = len(optimized_failed)
                        
                        messages = []
                        if improvement > 0:
                            messages.append(f"landing port {improvement}dog saving ({initial_used}dog → {final_used}dog)")
                        if failed_count > 0:
                            messages.append(f"Assignment failed: {failed_count}dog")
                        if airline_score > 0:
                            messages.append(f"Airline grouping: {airline_score}dog")
                        
                        if messages:
                            st.warning(f"⚠️ Optimized! {' | '.join(messages)}")
                        else:
                            st.success(f"✅ Optimized! All flight assignments successful (Used landing gear: {final_used}dog)")
        
        if 'apron_assignments' in st.session_state:
            assignments = st.session_state['apron_assignments']
            failed_assignments = st.session_state.get('apron_failed', [])
            
            # statistical information
            st.markdown("### 📊 Placement Statistics")
            used_aprons = len(set(assignments.values()))
            apron_counts = pd.Series(list(assignments.values())).value_counts().sort_index()
            total_flights = len(assignments) + len(failed_assignments)
            
            # Airline grouping KPI calculate
            aprons_for_kpi = {i: [] for i in range(num_aprons)}
            for flight_id, apron_id in assignments.items():
                flight = flights_df[flights_df['flight_id'] == flight_id].iloc[0]
                aprons_for_kpi[apron_id].append((flight['arrival_time'], flight['departure_time'], flight_id))
            for apron_id in aprons_for_kpi:
                aprons_for_kpi[apron_id].sort(key=lambda x: x[0])
            
            airline_score, total_pairs = calculate_airline_grouping_score(aprons_for_kpi, flights_df)
            airline_grouping_rate = (airline_score / total_pairs * 100) if total_pairs > 0 else 0
            
            stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
            with stat_col1:
                st.metric("Number of landing gear used", f"{used_aprons}dog")
            with stat_col2:
                st.metric("Assignment Success", f"{len(assignments)}dog", delta=f"{len(assignments)/total_flights*100:.1f}%" if total_flights > 0 else "0%")
            with stat_col3:
                st.metric("Assignment failed", f"{len(failed_assignments)}dog", 
                         delta=f"-{len(failed_assignments)/total_flights*100:.1f}%" if total_flights > 0 else "0%",
                         delta_color="inverse")
            with stat_col4:
                avg_per_apron = len(assignments) / used_aprons if used_aprons > 0 else 0
                st.metric("Average flights per landing station", f"{avg_per_apron:.1f}dog")
            with stat_col5:
                st.metric("Airline grouping KPI", f"{airline_score}dog", 
                         delta=f"{airline_grouping_rate:.1f}%" if total_pairs > 0 else "0%",
                         help=f"Assignment adjacent to same airline: {airline_score}/{total_pairs} pair")
            
            # Airline grouping details
            if airline_score > 0 or total_pairs > 0:
                st.markdown("#### 🎯 Airline grouping KPI particular")
                kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
                with kpi_col1:
                    st.metric("Same-airline adjacent pairs", f"{airline_score}dog")
                with kpi_col2:
                    st.metric("full adjacent pair", f"{total_pairs}dog")
                with kpi_col3:
                    st.metric("grouping ratio", f"{airline_grouping_rate:.1f}%")
            
            # Show failed assignment flights
            if len(failed_assignments) > 0:
                st.markdown("### ❌ Flight assignment failed")
                failed_df = flights_df[flights_df['flight_id'].isin(failed_assignments)].copy()
                failed_df['arrival_time'] = failed_df['arrival_time'].dt.strftime('%H:%M')
                failed_df['departure_time'] = failed_df['departure_time'].dt.strftime('%H:%M')
                failed_df['status'] = 'Assignment failed'
                failed_df = failed_df[['flight_code', 'airline', 'aircraft_type', 
                                      'arrival_time', 'departure_time', 'status']]
                st.dataframe(failed_df, use_container_width=True, hide_index=True)
            
            # Allocation status by landing site
            st.markdown("### 🎯 Allocation status by landing site")
            assignment_df = flights_df.copy()
            assignment_df['apron'] = assignment_df['flight_id'].map(assignments)
            assignment_df['apron'] = assignment_df['apron'].apply(lambda x: f"Apron {x+1}")
            
            for apron_num in range(num_aprons):
                apron_name = f"Apron {apron_num+1}"
                apron_flights = assignment_df[assignment_df['apron'] == apron_name]
                
                if len(apron_flights) > 0:
                    with st.expander(f"**{apron_name}** ({len(apron_flights)}dog flights)", expanded=False):
                        display_df = apron_flights[['flight_code', 'airline', 'aircraft_type', 
                                                    'arrival_time', 'departure_time']].copy()
                        display_df['arrival_time'] = display_df['arrival_time'].dt.strftime('%H:%M')
                        display_df['departure_time'] = display_df['departure_time'].dt.strftime('%H:%M')
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # visualization
            st.markdown("### 📈 visualization")
            fig = visualize_assignments(flights_df, assignments, num_aprons)
            st.plotly_chart(fig, use_container_width=True)
            
            # Assignment Result Table
            st.markdown("### 📋 Assignment result details")
            result_df = flights_df.copy()
            result_df['apron'] = result_df['flight_id'].map(assignments).apply(
                lambda x: f"Apron {x+1}" if pd.notna(x) else "Assignment failed"
            )
            result_df['status'] = result_df['flight_id'].apply(
                lambda x: 'Assignment Success' if x in assignments else 'Assignment failed'
            )
            result_df['arrival_time'] = result_df['arrival_time'].dt.strftime('%H:%M')
            result_df['departure_time'] = result_df['departure_time'].dt.strftime('%H:%M')
            result_df = result_df[['flight_code', 'airline', 'aircraft_type', 
                                   'arrival_time', 'departure_time', 'apron', 'status']]
            st.dataframe(result_df, use_container_width=True, hide_index=True)


# ============================================================================
# main function
# ============================================================================

def main_apron_assignment():
    """Main function of automatic mooring assignment system"""
    apply_css()
    render_apron_assignment_system()


if __name__ == "__main__":
    st.set_page_config(page_title="Apron Relocation", layout="wide", initial_sidebar_state="collapsed")
    if not st.session_state.get("authenticated", False):
        st.warning("Please log in from the Home page to access this section.")
        st.stop()
    st.title("🛫 Automatic mooring assignment system")
    main_apron_assignment()
