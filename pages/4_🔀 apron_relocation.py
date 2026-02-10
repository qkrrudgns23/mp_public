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

# 전역 랜덤 시드 설정
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ============================================================================
# 계류장 자동배정 시스템
# ============================================================================

# ============================================================================
# 모듈화된 목표 및 제약조건 시스템
# ============================================================================

class Objective:
    """목표 함수를 정의하는 기본 클래스"""
    def __init__(self, name, weight=1.0, maximize=True):
        self.name = name
        self.weight = weight
        self.maximize = maximize  # True: 최대화, False: 최소화
    
    def calculate(self, aprons, flights_df, assignments, num_used_aprons):
        """
        목표 함수 값 계산
        Returns: (score, metadata)
        - score: 점수 값
        - metadata: 추가 정보 (KPI 등)
        """
        raise NotImplementedError("Subclass must implement calculate method")
    
    def get_score(self, aprons, flights_df, assignments, num_used_aprons):
        """가중치를 적용한 최종 점수 반환"""
        score, metadata = self.calculate(aprons, flights_df, assignments, num_used_aprons)
        weighted_score = score * self.weight
        if not self.maximize:
            weighted_score = -weighted_score  # 최소화는 음수로 변환
        return weighted_score, metadata


class MinimizeApronsObjective(Objective):
    """접현주기장 수 최소화 목표"""
    def __init__(self, weight=1.0):
        super().__init__("minimize_aprons", weight, maximize=False)
    
    def calculate(self, aprons, flights_df, assignments, num_used_aprons):
        return -num_used_aprons, {"num_used_aprons": num_used_aprons}


class AirlineGroupingObjective(Objective):
    """항공사 그룹핑 최대화 목표"""
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
    """특정 접현주기장 회피 목표"""
    def __init__(self, avoid_aprons, weight=1.0):
        """
        Args:
            avoid_aprons: 회피할 접현주기장 ID 리스트 (0-indexed)
            weight: 가중치
        """
        super().__init__("avoid_specific_aprons", weight, maximize=False)
        self.avoid_aprons = set(avoid_aprons)
    
    def calculate(self, aprons, flights_df, assignments, num_used_aprons):
        penalty = 0
        for apron_id in self.avoid_aprons:
            penalty += len(aprons.get(apron_id, []))
        return -penalty, {"avoided_aprons": list(self.avoid_aprons), "penalty": penalty}


class PreferSpecificApronsObjective(Objective):
    """특정 접현주기장 선호 목표"""
    def __init__(self, prefer_aprons, weight=0.5):
        """
        Args:
            prefer_aprons: 선호할 접현주기장 ID 리스트 (0-indexed)
            weight: 가중치
        """
        super().__init__("prefer_specific_aprons", weight, maximize=True)
        self.prefer_aprons = set(prefer_aprons)
    
    def calculate(self, aprons, flights_df, assignments, num_used_aprons):
        score = 0
        for apron_id in self.prefer_aprons:
            score += len(aprons.get(apron_id, []))
        return score, {"preferred_aprons": list(self.prefer_aprons), "score": score}


class HardConstraint:
    """하드 제약조건 (반드시 만족해야 함)"""
    def __init__(self, name):
        self.name = name
    
    def is_valid(self, aprons, flights_df, assignments):
        """
        제약조건 만족 여부 확인
        Returns: True if valid, False otherwise
        """
        raise NotImplementedError("Subclass must implement is_valid method")
    
    def filter_valid_aprons(self, flight_id, aprons, flights_df, num_aprons):
        """
        특정 항공편에 대해 유효한 접현주기장 리스트 반환
        Returns: list of valid apron IDs
        """
        return list(range(num_aprons))  # 기본적으로 모든 접현주기장 허용


class AircraftTypeConstraint(HardConstraint):
    """항공기종에 따른 접현주기장 제한"""
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
    """모든 목표와 제약조건을 통합하여 점수를 계산하는 클래스"""
    def __init__(self, objectives=None, constraints=None):
        """
        Args:
            objectives: Objective 객체 리스트
            constraints: HardConstraint 객체 리스트
        """
        self.objectives = objectives or []
        self.constraints = constraints or []
    
    def add_objective(self, objective):
        """목표 추가"""
        self.objectives.append(objective)
    
    def add_constraint(self, constraint):
        """제약조건 추가"""
        self.constraints.append(constraint)
    
    def calculate_total_score(self, aprons, flights_df, assignments, num_used_aprons):
        """
        총 점수 계산
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
        """모든 제약조건 만족 여부 확인"""
        for constraint in self.constraints:
            if not constraint.is_valid(aprons, flights_df, assignments):
                return False, constraint.name
        return True, None
    
    def get_valid_aprons(self, flight_id, aprons, flights_df, num_aprons):
        """특정 항공편에 대해 모든 제약조건을 만족하는 접현주기장 리스트 반환"""
        valid_aprons = set(range(num_aprons))
        
        for constraint in self.constraints:
            constraint_valid = constraint.filter_valid_aprons(flight_id, aprons, flights_df, num_aprons)
            valid_aprons = valid_aprons.intersection(set(constraint_valid))
        
        return list(valid_aprons)

def generate_flight_dummy_data(num_flights=20):
    """항공편 더미데이터 생성"""
    flights = []
    airlines = ['KE', 'OZ', '7C', 'LJ', 'TW', 'BX', 'ZE']
    
    # 하루를 24시간으로 나누어 시간대별로 분산
    base_time = datetime(2024, 1, 18, 6, 0)  # 오전 6시부터 시작
    
    for i in range(num_flights):
        # 도착시간: 6시~22시 사이 랜덤
        arrival_hour = np.random.randint(6, 22)
        arrival_minute = np.random.choice([0, 15, 30, 45])
        arrival_time = base_time.replace(hour=arrival_hour, minute=arrival_minute)
        
        # 체류시간: 1시간~4시간
        stay_duration = np.random.choice([40, 50, 60, 70, 90])  # 분 단위
        departure_time = arrival_time + pd.Timedelta(minutes=stay_duration)
        
        # 출발시간이 자정을 넘지 않도록 조정
        if departure_time.hour >= 24:
            departure_time = departure_time.replace(hour=23, minute=59)
        
        airline = np.random.choice(airlines)
        flight_id = f"FL{i+1:03d}"
        
        flight = {
            'flight_id': flight_id,
            'airline': airline,
            'flight_code': f"{airline}-{flight_id}",  # airline과 flight_id를 합친 컬럼
            'arrival_time': arrival_time,
            'departure_time': departure_time,
            'aircraft_type': np.random.choice(['A320', 'A321', 'B737', 'B777', 'A330']),
        }
        flights.append(flight)
    
    df = pd.DataFrame(flights)
    # 시간순으로 정렬
    df = df.sort_values('arrival_time').reset_index(drop=True)
    return df


def initial_assignment(flights_df, num_aprons=5):
    """초안 배정: 시간순으로 First-Fit 알고리즘 사용"""
    # 각 접현주기장의 사용 시간대를 저장
    aprons = {i: [] for i in range(num_aprons)}  # {apron_id: [(start, end), ...]}
    assignments = {}  # {flight_id: apron_id}
    failed_assignments = []  # 배정 실패한 항공편 ID 리스트
    
    for idx, flight in flights_df.iterrows():
        flight_id = flight['flight_id']
        arrival = flight['arrival_time']
        departure = flight['departure_time']
        
        # 첫 번째로 사용 가능한 접현주기장 찾기
        assigned = False
        for apron_id in range(num_aprons):
            # 시간 겹침 확인
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
            # 모든 접현주기장이 사용 중이면 배정 실패로 처리
            failed_assignments.append(flight_id)
    
    return assignments, aprons, failed_assignments


def calculate_airline_grouping_score(aprons, flights_df):
    """
    동일 항공사 인접 배정 점수 계산 (하위 호환성 유지)
    각 접현주기장에서 시간순으로 앞뒤로 붙어있는 동일 항공사 쌍의 개수를 반환
    """
    score = 0
    total_pairs = 0
    
    for apron_id, flights in aprons.items():
        if len(flights) < 2:
            continue
        
        # 시간순으로 정렬된 항공편들 (이미 정렬되어 있음)
        for i in range(len(flights) - 1):
            current_flight_id = flights[i][2]
            next_flight_id = flights[i+1][2]
            
            current_flight = flights_df[flights_df['flight_id'] == current_flight_id].iloc[0]
            next_flight = flights_df[flights_df['flight_id'] == next_flight_id].iloc[0]
            
            # 앞뒤로 붙어있고 동일 항공사면 점수 증가
            if current_flight['airline'] == next_flight['airline']:
                score += 1
            total_pairs += 1
    
    return score, total_pairs


def create_default_score_calculator(weight_apron=1.0, weight_airline=0.5, 
                                     avoid_aprons=None, prefer_aprons=None,
                                     aircraft_constraints=None):
    """
    기본 ScoreCalculator 생성 (하위 호환성 유지)
    
    Args:
        weight_apron: 접현주기장 최소화 가중치
        weight_airline: 항공사 그룹핑 가중치
        avoid_aprons: 회피할 접현주기장 리스트 (0-indexed)
        prefer_aprons: 선호할 접현주기장 리스트 (0-indexed)
        aircraft_constraints: 항공기종 제약조건 {aircraft_type: [allowed_aprons]}
    """
    calculator = ScoreCalculator()
    
    # 기본 목표 추가
    calculator.add_objective(MinimizeApronsObjective(weight=weight_apron))
    calculator.add_objective(AirlineGroupingObjective(weight=weight_airline))
    
    # 추가 목표
    if avoid_aprons:
        calculator.add_objective(AvoidSpecificApronsObjective(avoid_aprons, weight=1.0))
    
    if prefer_aprons:
        calculator.add_objective(PreferSpecificApronsObjective(prefer_aprons, weight=0.5))
    
    # 제약조건 추가
    if aircraft_constraints:
        calculator.add_constraint(AircraftTypeConstraint(aircraft_constraints))
    
    return calculator


# ============================================================================
# 사용 예시: 자연어 조건을 코드로 변환하는 헬퍼 함수
# ============================================================================

def parse_natural_language_constraint(natural_language_text, num_aprons=5):
    """
    자연어 조건을 파싱하여 ScoreCalculator에 추가할 수 있는 형태로 변환
    
    예시:
        "Apron4, Apron5에 최대한 적게 배정하고 싶어"
        → {"type": "avoid_aprons", "aprons": [3, 4], "weight": 1.0}
        
        "KE 항공사는 Apron1, Apron2에 배정해줘"
        → {"type": "prefer_airline_aprons", "airline": "KE", "aprons": [0, 1], "weight": 0.8}
    
    Args:
        natural_language_text: 자연어 조건 텍스트
        num_aprons: 접현주기장 수
    
    Returns:
        dict: 파싱된 조건 정보
    """
    # 실제로는 LLM API를 호출하여 파싱
    # 여기서는 간단한 예시만 제공
    
    text_lower = natural_language_text.lower()
    
    # Apron 회피 조건
    if "적게" in text_lower or "회피" in text_lower or "피하고" in text_lower:
        # Apron 번호 추출 (1-indexed를 0-indexed로 변환)
        import re
        apron_numbers = re.findall(r'apron\s*(\d+)', text_lower, re.IGNORECASE)
        if apron_numbers:
            aprons = [int(n) - 1 for n in apron_numbers]  # 1-indexed → 0-indexed
            return {
                "type": "avoid_aprons",
                "aprons": aprons,
                "weight": 1.0
            }
    
    # Apron 선호 조건
    if "선호" in text_lower or "배정해" in text_lower:
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
    자연어 조건을 ScoreCalculator에 추가
    
    Args:
        score_calculator: ScoreCalculator 객체
        natural_language_text: 자연어 조건 텍스트
        num_aprons: 접현주기장 수
    
    Returns:
        bool: 성공 여부
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
# 사용 예시 코드 (주석)
# ============================================================================

"""
# 예시 1: 기본 사용 (하위 호환성 유지)____________________________________________________
result = greedy_optimization(
    flights_df, initial_assignments, failed_assignments, 
    num_aprons=5, weight_apron=1.0, weight_airline=0.5
)

# 예시 2: Apron 회피 조건 추가____________________________________________________
result = greedy_optimization(
    flights_df, initial_assignments, failed_assignments,
    num_aprons=5, avoid_aprons=[3, 4]  # Apron 4, 5 회피
)

# 예시 3: 커스텀 ScoreCalculator 사용____________________________________________________ 
calculator = ScoreCalculator()
calculator.add_objective(MinimizeApronsObjective(weight=1.0))
calculator.add_objective(AirlineGroupingObjective(weight=0.5))
calculator.add_objective(AvoidSpecificApronsObjective([3, 4], weight=1.0))
calculator.add_constraint(AircraftTypeConstraint({
    'B777': [0, 1],  # B777은 Apron 1, 2만 가능
    'A330': [2, 3]   # A330은 Apron 3, 4만 가능
}))


result = greedy_optimization(
    flights_df, initial_assignments, failed_assignments,
    num_aprons=5, score_calculator=calculator
    ) # score_calculator값을 greedy_optimization이 변수로 받으면 따로 조건 중복입력 불필요








# 예시 4: 자연어 조건 추가____________________________________________________
calculator = create_default_score_calculator(weight_apron=1.0, weight_airline=0.5)
add_constraint_from_natural_language(
    calculator, 
    "Apron4, Apron5에 최대한 적게 배정하고 싶어",
    num_aprons=5
)

result = greedy_optimization(
    flights_df, initial_assignments, failed_assignments,
    num_aprons=5, score_calculator=calculator
)


# 예시 5: 새로운 목표 함수 추가 (확장)____________________________________________________
class MinimizeDistanceObjective(Objective):
    \"\"\"접현주기장 간 거리 최소화 목표\"\"\"
    def __init__(self, distance_matrix, weight=0.3):
        super().__init__("minimize_distance", weight, maximize=False)
        self.distance_matrix = distance_matrix
    
    def calculate(self, aprons, flights_df, assignments, num_used_aprons):
        total_distance = 0
        for flight_id, apron_id in assignments.items():
            # 거리 계산 로직
            pass
        return -total_distance, {"total_distance": total_distance}

calculator.add_objective(MinimizeDistanceObjective(distance_matrix, weight=0.3))
"""


def greedy_optimization(flights_df, initial_assignments, failed_assignments, num_aprons=5, 
                        weight_apron=1.0, weight_airline=0.5, score_calculator=None,
                        avoid_aprons=None, prefer_aprons=None, aircraft_constraints=None):
    """
    Greedy 알고리즘으로 다중 목표 최적화 (모듈화된 버전)
    
    Args:
        flights_df: 항공편 데이터프레임
        initial_assignments: 초기 배정 결과
        failed_assignments: 배정 실패 항공편 리스트
        num_aprons: 접현주기장 수
        weight_apron: 접현주기장 최소화 가중치 (하위 호환성)
        weight_airline: 항공사 그룹핑 가중치 (하위 호환성)
        score_calculator: ScoreCalculator 객체 (None이면 기본 생성)
        avoid_aprons: 회피할 접현주기장 리스트
        prefer_aprons: 선호할 접현주기장 리스트
        aircraft_constraints: 항공기종 제약조건
    
    Returns:
        (optimized_assignments, optimized_failed, final_num_used, initial_num_used, 
         final_airline_score, total_pairs, metadata)
    """
    # ScoreCalculator 생성 (없으면 기본 생성)
    if score_calculator is None:
        score_calculator = create_default_score_calculator(
            weight_apron=weight_apron,
            weight_airline=weight_airline,
            avoid_aprons=avoid_aprons,
            prefer_aprons=prefer_aprons,
            aircraft_constraints=aircraft_constraints
        )
    # 현재 사용 중인 접현주기장 확인
    used_aprons = set(initial_assignments.values())
    num_used = len(used_aprons)
    
    # 각 접현주기장의 사용 시간대 재구성
    aprons = {i: [] for i in range(num_aprons)}
    for flight_id, apron_id in initial_assignments.items():
        flight = flights_df[flights_df['flight_id'] == flight_id].iloc[0]
        aprons[apron_id].append((flight['arrival_time'], flight['departure_time'], flight_id))
    
    # 각 접현주기장의 시간대를 정렬
    for apron_id in aprons:
        aprons[apron_id].sort(key=lambda x: x[0])
    
    # 최적화: 사용 중인 접현주기장을 최소화하고 항공사 그룹핑 최대화
    optimized_assignments = initial_assignments.copy()
    optimized_failed = failed_assignments.copy()
    
    # 배정 실패 항공편 재시도 (항공사 그룹핑을 고려)
    retry_flights = []
    for flight_id in optimized_failed:
        flight = flights_df[flights_df['flight_id'] == flight_id].iloc[0]
        arrival = flight['arrival_time']
        departure = flight['departure_time']
        airline = flight['airline']
        
        # 사용 가능한 접현주기장 찾기 (항공사 그룹핑 우선 고려)
        assigned = False
        best_apron = None
        best_score = -float('inf')
        
        # 제약조건을 만족하는 접현주기장만 확인
        valid_aprons = score_calculator.get_valid_aprons(flight_id, aprons, flights_df, num_aprons)
        
        for apron_id in valid_aprons:
            conflict = False
            for existing_start, existing_end, _ in aprons[apron_id]:
                if not (departure <= existing_start or arrival >= existing_end):
                    conflict = True
                    break
            
            if not conflict:
                # 항공사 그룹핑 점수 계산 (임시로 추가해서 점수 확인)
                temp_aprons = {k: v.copy() for k, v in aprons.items()}
                temp_aprons[apron_id].append((arrival, departure, flight_id))
                temp_aprons[apron_id].sort(key=lambda x: x[0])
                
                # ScoreCalculator를 사용하여 점수 계산
                temp_assignments = optimized_assignments.copy()
                temp_assignments[flight_id] = apron_id
                temp_used = len([a for a in apron_usage.values() if a > 0])
                if apron_id not in [a for a in range(num_aprons) if apron_usage[a] > 0]:
                    temp_used += 1
                
                temp_score, _ = score_calculator.calculate_total_score(
                    temp_aprons, flights_df, temp_assignments, temp_used
                )
                
                # 점수가 더 좋으면 선택
                if temp_score > best_score:
                    best_score = temp_score
                    best_apron = apron_id
        
        if best_apron is not None:
            aprons[best_apron].append((arrival, departure, flight_id))
            aprons[best_apron].sort(key=lambda x: x[0])
            optimized_assignments[flight_id] = best_apron
            retry_flights.append(flight_id)
            assigned = True
    
    # 재시도 성공한 항공편 제거
    for flight_id in retry_flights:
        if flight_id and flight_id in optimized_failed:
            optimized_failed.remove(flight_id)
    
    # 사용 빈도가 낮은 접현주기장부터 다른 접현주기장으로 이동 시도
    apron_usage = {i: len(aprons[i]) for i in range(num_aprons)}
    sorted_aprons = sorted(range(num_aprons), key=lambda x: apron_usage[x])
    
    # 여러 라운드로 최적화 수행
    max_iterations = 10
    for iteration in range(max_iterations):
        changed = False
        
        # 낮은 사용량의 접현주기장부터 시작하여 재배정 시도
        for source_apron in sorted_aprons:
            if apron_usage[source_apron] == 0:
                continue
            
            # 이 접현주기장의 항공편들을 다른 접현주기장으로 이동 시도
            flights_to_reassign = aprons[source_apron].copy()
            
            for arrival, departure, flight_id in flights_to_reassign:
                # 이미 사용 중인 접현주기장을 우선적으로 고려
                # 사용량이 많은 접현주기장부터 확인 (이미 사용 중이면 이동 가능성 높음)
                target_candidates = sorted(
                    [i for i in range(num_aprons) if i != source_apron],
                    key=lambda x: apron_usage[x],
                    reverse=True
                )
                
                for target_apron in target_candidates:
                    # 제약조건 확인
                    valid_aprons = score_calculator.get_valid_aprons(flight_id, aprons, flights_df, num_aprons)
                    if target_apron not in valid_aprons:
                        continue
                    
                    # 시간 겹침 확인
                    conflict = False
                    for existing_start, existing_end, _ in aprons[target_apron]:
                        if not (departure <= existing_start or arrival >= existing_end):
                            conflict = True
                            break
                    
                    if not conflict:
                        # 이동 전 점수 계산
                        current_used = len([a for a in apron_usage.values() if a > 0])
                        current_score, _ = score_calculator.calculate_total_score(
                            aprons, flights_df, optimized_assignments, current_used
                        )
                        
                        # 이동 시뮬레이션
                        temp_aprons = {k: [f for f in v] for k, v in aprons.items()}
                        temp_aprons[source_apron].remove((arrival, departure, flight_id))
                        temp_aprons[target_apron].append((arrival, departure, flight_id))
                        temp_aprons[target_apron].sort(key=lambda x: x[0])
                        
                        # 이동 후 사용량 계산
                        temp_usage = {i: len(temp_aprons[i]) for i in range(num_aprons)}
                        new_used = len([a for a in temp_usage.values() if a > 0])
                        
                        # 이동 후 배정 업데이트
                        temp_assignments = optimized_assignments.copy()
                        temp_assignments[flight_id] = target_apron
                        
                        # 이동 후 점수 계산
                        new_score, _ = score_calculator.calculate_total_score(
                            temp_aprons, flights_df, temp_assignments, new_used
                        )
                        
                        # 점수가 개선되거나 동일하면 이동 (다중 목표 최적화)
                        if new_score >= current_score:
                            # 이동 수행
                            aprons[source_apron].remove((arrival, departure, flight_id))
                            aprons[target_apron].append((arrival, departure, flight_id))
                            aprons[target_apron].sort(key=lambda x: x[0])
                            optimized_assignments[flight_id] = target_apron
                            apron_usage[source_apron] -= 1
                            apron_usage[target_apron] += 1
                            changed = True
                            break
        
        # 더 이상 변경이 없으면 종료
        if not changed:
            break
    
    # 최종 사용 중인 접현주기장 수 계산
    final_used_aprons = {apron_id for apron_id in optimized_assignments.values() if apron_usage[apron_id] > 0}
    final_num_used = len(final_used_aprons)
    
    # 최종 점수 및 메타데이터 계산
    final_score, final_metadata = score_calculator.calculate_total_score(
        aprons, flights_df, optimized_assignments, final_num_used
    )
    
    # 하위 호환성을 위한 항공사 그룹핑 점수
    final_airline_score = final_metadata.get('airline_grouping', {}).get('airline_pairs', 0)
    total_pairs = final_metadata.get('airline_grouping', {}).get('total_pairs', 0)
    
    return optimized_assignments, optimized_failed, final_num_used, num_used, final_airline_score, total_pairs, final_metadata


def visualize_assignments(flights_df, assignments, num_aprons=5):
    """배정 결과를 시각화"""
    # Gantt 차트 스타일로 시각화
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3[:num_aprons]
    
    # 기준 시간 설정 (첫 항공편 도착 시간의 시작 시간)
    base_time = flights_df['arrival_time'].min().replace(hour=6, minute=0)
    
    for apron_id in range(num_aprons):
        apron_flights = flights_df[flights_df['flight_id'].isin(
            [fid for fid, aid in assignments.items() if aid == apron_id]
        )].copy()
        
        if len(apron_flights) == 0:
            continue
        
        # 시간을 숫자로 변환 (분 단위, base_time 기준)
        apron_flights = apron_flights.sort_values('arrival_time')
        for idx, flight in apron_flights.iterrows():
            # base_time 기준으로 경과 시간 계산 (분)
            start_min = (flight['arrival_time'] - base_time).total_seconds() / 60
            duration = (flight['departure_time'] - flight['arrival_time']).total_seconds() / 60
            
            # 시간 문자열 생성
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
    
    # x축 시간 레이블 생성 (1시간 간격)
    max_time = (flights_df['departure_time'].max() - base_time).total_seconds() / 60
    hour_ticks = list(range(0, int(max_time) + 60, 60))
    hour_labels = [(base_time + pd.Timedelta(minutes=t)).strftime('%H:%M') for t in hour_ticks]
    
    fig.update_layout(
        title='계류장 배정 결과 (Gantt Chart)',
        xaxis_title='시간',
        yaxis_title='접현주기장',
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
    """계류장 자동배정 시스템 UI 렌더링"""
    st.markdown("## 🛫 계류장 자동배정 시스템")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        num_flights = st.number_input("항공편 수", min_value=5, max_value=50, value=20, step=1)
        num_aprons = st.number_input("접현주기장 수", min_value=3, max_value=10, value=5, step=1)
    
    with col2:
        if st.button("🔄 더미데이터 생성", use_container_width=True):
            flights_df = generate_flight_dummy_data(num_flights)
            st.session_state['apron_flights_df'] = flights_df
            st.session_state['apron_assignments'] = None
            st.success(f"✅ {num_flights}개의 항공편 데이터가 생성되었습니다!")
    
    if 'apron_flights_df' in st.session_state:
        flights_df = st.session_state['apron_flights_df']
        
        st.markdown("### 📋 항공편 데이터")
        st.dataframe(flights_df[['flight_code', 'airline', 'aircraft_type', 
                                 'arrival_time', 'departure_time']].style.format({
            'arrival_time': lambda x: x.strftime('%H:%M'),
            'departure_time': lambda x: x.strftime('%H:%M')
        }), use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📝 초안 배정 실행", use_container_width=True):
                with st.spinner("초안 배정 중..."):
                    initial_assignments, aprons, failed_assignments = initial_assignment(flights_df, num_aprons)
                    st.session_state['apron_assignments'] = initial_assignments
                    st.session_state['apron_failed'] = failed_assignments
                    st.session_state['apron_initial'] = True
                    st.session_state['apron_optimized'] = False
                    
                    used_aprons = len(set(initial_assignments.values()))
                    success_count = len(initial_assignments)
                    failed_count = len(failed_assignments)
                    
                    # 초안 배정의 항공사 그룹핑 점수 계산
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
                        messages.append(f"성공: {success_count}개, 실패: {failed_count}개")
                    messages.append(f"사용된 접현주기장: {used_aprons}개")
                    if initial_airline_score > 0:
                        messages.append(f"항공사 그룹핑: {initial_airline_score}개")
                    
                    if failed_count > 0:
                        st.warning(f"⚠️ 초안 배정 완료! {' | '.join(messages)}")
                    else:
                        st.success(f"✅ 초안 배정 완료! {' | '.join(messages)}")
        
        with col2:
            weight_col1, weight_col2 = st.columns(2)
            with weight_col1:
                weight_apron = st.slider("접현주기장 최소화 가중치", 0.0, 2.0, 1.0, 0.1, key="weight_apron")
            with weight_col2:
                weight_airline = st.slider("항공사 그룹핑 가중치", 0.0, 2.0, 0.5, 0.1, key="weight_airline")
            
            if st.button("⚡ Greedy 최적화 실행", use_container_width=True, 
                        disabled='apron_assignments' not in st.session_state):
                with st.spinner("최적화 중..."):
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
                            messages.append(f"접현주기장 {improvement}개 절약 ({initial_used}개 → {final_used}개)")
                        if failed_count > 0:
                            messages.append(f"배정 실패: {failed_count}개")
                        if airline_score > 0:
                            messages.append(f"항공사 그룹핑: {airline_score}개")
                        
                        if messages:
                            st.warning(f"⚠️ 최적화 완료! {' | '.join(messages)}")
                        else:
                            st.success(f"✅ 최적화 완료! 모든 항공편 배정 성공 (사용 접현주기장: {final_used}개)")
        
        if 'apron_assignments' in st.session_state:
            assignments = st.session_state['apron_assignments']
            failed_assignments = st.session_state.get('apron_failed', [])
            
            # 통계 정보
            st.markdown("### 📊 배정 통계")
            used_aprons = len(set(assignments.values()))
            apron_counts = pd.Series(list(assignments.values())).value_counts().sort_index()
            total_flights = len(assignments) + len(failed_assignments)
            
            # 항공사 그룹핑 KPI 계산
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
                st.metric("사용된 접현주기장 수", f"{used_aprons}개")
            with stat_col2:
                st.metric("배정 성공", f"{len(assignments)}개", delta=f"{len(assignments)/total_flights*100:.1f}%" if total_flights > 0 else "0%")
            with stat_col3:
                st.metric("배정 실패", f"{len(failed_assignments)}개", 
                         delta=f"-{len(failed_assignments)/total_flights*100:.1f}%" if total_flights > 0 else "0%",
                         delta_color="inverse")
            with stat_col4:
                avg_per_apron = len(assignments) / used_aprons if used_aprons > 0 else 0
                st.metric("접현주기장당 평균 항공편", f"{avg_per_apron:.1f}개")
            with stat_col5:
                st.metric("항공사 그룹핑 KPI", f"{airline_score}개", 
                         delta=f"{airline_grouping_rate:.1f}%" if total_pairs > 0 else "0%",
                         help=f"동일 항공사 인접 배정: {airline_score}/{total_pairs} 쌍")
            
            # 항공사 그룹핑 상세 정보
            if airline_score > 0 or total_pairs > 0:
                st.markdown("#### 🎯 항공사 그룹핑 KPI 상세")
                kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
                with kpi_col1:
                    st.metric("동일 항공사 인접 쌍", f"{airline_score}개")
                with kpi_col2:
                    st.metric("전체 인접 쌍", f"{total_pairs}개")
                with kpi_col3:
                    st.metric("그룹핑 비율", f"{airline_grouping_rate:.1f}%")
            
            # 배정 실패 항공편 표시
            if len(failed_assignments) > 0:
                st.markdown("### ❌ 배정 실패 항공편")
                failed_df = flights_df[flights_df['flight_id'].isin(failed_assignments)].copy()
                failed_df['arrival_time'] = failed_df['arrival_time'].dt.strftime('%H:%M')
                failed_df['departure_time'] = failed_df['departure_time'].dt.strftime('%H:%M')
                failed_df['status'] = '배정 실패'
                failed_df = failed_df[['flight_code', 'airline', 'aircraft_type', 
                                      'arrival_time', 'departure_time', 'status']]
                st.dataframe(failed_df, use_container_width=True, hide_index=True)
            
            # 접현주기장별 배정 현황
            st.markdown("### 🎯 접현주기장별 배정 현황")
            assignment_df = flights_df.copy()
            assignment_df['apron'] = assignment_df['flight_id'].map(assignments)
            assignment_df['apron'] = assignment_df['apron'].apply(lambda x: f"Apron {x+1}")
            
            for apron_num in range(num_aprons):
                apron_name = f"Apron {apron_num+1}"
                apron_flights = assignment_df[assignment_df['apron'] == apron_name]
                
                if len(apron_flights) > 0:
                    with st.expander(f"**{apron_name}** ({len(apron_flights)}개 항공편)", expanded=False):
                        display_df = apron_flights[['flight_code', 'airline', 'aircraft_type', 
                                                    'arrival_time', 'departure_time']].copy()
                        display_df['arrival_time'] = display_df['arrival_time'].dt.strftime('%H:%M')
                        display_df['departure_time'] = display_df['departure_time'].dt.strftime('%H:%M')
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # 시각화
            st.markdown("### 📈 시각화")
            fig = visualize_assignments(flights_df, assignments, num_aprons)
            st.plotly_chart(fig, use_container_width=True)
            
            # 배정 결과 테이블
            st.markdown("### 📋 배정 결과 상세")
            result_df = flights_df.copy()
            result_df['apron'] = result_df['flight_id'].map(assignments).apply(
                lambda x: f"Apron {x+1}" if pd.notna(x) else "배정 실패"
            )
            result_df['status'] = result_df['flight_id'].apply(
                lambda x: '배정 성공' if x in assignments else '배정 실패'
            )
            result_df['arrival_time'] = result_df['arrival_time'].dt.strftime('%H:%M')
            result_df['departure_time'] = result_df['departure_time'].dt.strftime('%H:%M')
            result_df = result_df[['flight_code', 'airline', 'aircraft_type', 
                                   'arrival_time', 'departure_time', 'apron', 'status']]
            st.dataframe(result_df, use_container_width=True, hide_index=True)


# ============================================================================
# 메인 함수
# ============================================================================

def main_apron_assignment():
    """계류장 자동배정 시스템 메인 함수"""
    apply_css()
    render_apron_assignment_system()


if __name__ == "__main__":
    st.title("🛫 계류장 자동배정 시스템")
    main_apron_assignment()
