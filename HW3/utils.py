from collections import Counter, defaultdict
from datetime import datetime
import heapq
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LogisticRegression


def filter_tournaments_data(
    results: Dict[int, List], tournaments: Dict[int, Dict], years: Tuple[int]
) -> Tuple[Dict[int, List], Dict[int, Dict]]:
    """
    Filter out tournament results by:
        - certain years
        - results have at least one team both filled
            - question answering mask
            - members list
        (save only these teams results)
    """
    new_results = defaultdict(list)
    new_tournaments = {}
    years = set(years)
    for tournament_id, tournament_results_data in results.items():
        tournament_year = datetime.strptime(
            tournaments[tournament_id]["dateStart"], '%Y-%m-%dT%H:%M:%S%z'
        ).year
        
        if tournament_year not in years:
            continue
        
        for team_result in tournament_results_data:
            mask = team_result.get("mask", None)
            team_members = team_result.get("teamMembers", [])
            if mask and team_members:
                new_results[tournament_id].append(team_result)
    new_tournaments = {k: tournaments[k] for k, v in new_results.items()}
    return dict(new_results), new_tournaments


def split_data_on_train_and_test(
    results: Dict[int, List], tournaments: Dict[int, Dict], train_year: int=2019, test_year: int=2020
) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    result_data = {train_year: {}, test_year: {}}
    for tournament_id, tournament_results_data in results.items():
        tournament_year = datetime.strptime(
            tournaments[tournament_id]["dateStart"], '%Y-%m-%dT%H:%M:%S%z'
        ).year
        
        result_data[tournament_year][tournament_id] = {"tournament_name": tournaments[tournament_id]["name"]}
        team_results = []
        for team_result in tournament_results_data:
            team_results.append({
                "team_id": team_result["team"]["id"],
                "mask": team_result["mask"],
                "position": team_result["position"],
                "teamMembers": [team_member["player"]["id"] for team_member in team_result['teamMembers']],
            })
        result_data[tournament_year][tournament_id]["tournament_result"] = team_results
    return result_data[train_year], result_data[test_year]


def replace_rare_players_ids_inplace(
    data: Dict[int, Dict], q_min: int=100, replace_id: int=-1
) -> None:
    questions_per_player_counter = Counter()
    for tournament_data in data.values():
        for team_result in tournament_data["tournament_result"]:
            questions_num = len(team_result["mask"])
            questions_per_player_counter += Counter({
                player_id: questions_num for player_id in team_result["teamMembers"]
            })
    
    rare_players = set()
    for player_id, questions_num in questions_per_player_counter.most_common():
        if questions_num <= q_min:
            rare_players.add(player_id)
    
    for tournament_data in data.values():
        for team_result in tournament_data["tournament_result"]:
            team_result["teamMembers"] = set([
                replace_id if player_id in rare_players else player_id for player_id in team_result["teamMembers"]
            ])
    return None


def replace_unseen_on_train_players_inplace(
    train_data: Dict[int, Dict], test_data: Dict[int, Dict], replace_id: int=-1
) -> None:
    train_players = set()
    for tournament_data in train_data.values():
        for team_result in tournament_data["tournament_result"]:
            train_players.update(team_result["teamMembers"])
            
    for tournament_data in test_data.values():
        for team_result in tournament_data["tournament_result"]:
            team_result["teamMembers"] = set([
                replace_id if player_id not in train_players else player_id for player_id in team_result["teamMembers"]
            ])
    return None


def get_feature_mapper(data: Dict[int, Dict]) -> Dict[Union[str, int], int]:
    all_features = set()
    for tournament_id, tournament_data in data.items():
        for team_result in tournament_data["tournament_result"]:
            all_features.update(team_result["teamMembers"])
            questions_num = len(team_result["mask"])
            question_ids = (f"{tournament_id}_{question_num}" for question_num in range(questions_num))
            all_features.update(question_ids)
    mapper = {feature: i for i, feature in enumerate(all_features)}
    return mapper


def get_sparse_data_for_sklearn_api(
    data: Dict[int, Dict], mapper: Dict[Union[str, int], int], regime: str
) -> Tuple[coo_matrix, np.ndarray]:
    rows = []
    cols = []
    is_answered = []
    tournament_ids = []
    team_ids = []
    player_ids = []
    tournament_question_nums = []
    current_row = 0
    for tournament_id, tournament_data in data.items():
        for team_result in tournament_data["tournament_result"]:
            team_id = team_result["team_id"]
            players_num = len(team_result["teamMembers"])
            for tournament_question_num, answer in enumerate(team_result["mask"]):
                try:
                    is_answered.extend([int(answer)] * players_num)
                    tournament_ids.extend([tournament_id] * players_num)
                    team_ids.extend([team_id] * players_num)
                    tournament_question_nums.extend([tournament_question_num] * players_num)
                except ValueError:
                    continue
                for player_id in team_result["teamMembers"]:
                    player_ids.append(player_id)
                    rows.append(current_row)
                    cols.append(mapper[player_id])
                    if regime != "test":    
                        rows.append(current_row)
                        cols.append(mapper[f"{tournament_id}_{tournament_question_num}"])
                    current_row += 1
    
    num_rows = len(rows)
    num_cols = len(cols)
    num_answers = len(is_answered)
            
    assert num_rows == num_cols, f"different number of row and col indices: {num_rows} vs {num_cols}"
    
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    data = np.ones(len(rows))
    is_answered = np.asarray(is_answered, dtype=np.int8)
    
    tournament_ids = np.asarray(tournament_ids, dtype=np.int32)
    team_ids = np.asarray(team_ids, dtype=np.int32)
    tournament_question_nums = np.asarray(tournament_question_nums, dtype=np.int32)
    
    X = coo_matrix((data, (rows, cols)), shape=(num_answers, len(mapper)))
    return X, is_answered, tournament_ids, team_ids, player_ids, tournament_question_nums


def get_true_positions(data: Dict[int, Dict]) -> pd.DataFrame:
    tournament_ids = []
    team_ids = []
    positions = []
    for tournament_id, tournament_data in data.items():
        for team_result in tournament_data["tournament_result"]:
            tournament_ids.append(tournament_id)
            team_ids.append(team_result["team_id"])
            positions.append(team_result["position"])
    
    result = pd.DataFrame.from_dict({
        'tournament_id': tournament_ids,
        'team_id': team_ids,
        'position_true': positions,
    })
    return result


def get_predicted_positions(
    tournament_ids: np.ndarray, 
    team_ids: np.ndarray, 
    proba_to_answer: np.ndarray
) -> pd.DataFrame:
    predicted_test_results = pd.DataFrame.from_dict({
        'tournament_id': tournament_ids,
        'team_id': team_ids,
        'proba_to_not_answer': 1 - proba_to_answer,
    })
        
    predicted_test_results = predicted_test_results.groupby(
        ["tournament_id", "team_id"]
    ).agg("prod").reset_index()
    predicted_test_results["position_pred"] = predicted_test_results.groupby(
        "tournament_id"
    )["proba_to_not_answer"].rank("dense")
    return predicted_test_results


def get_kendall_and_spearman_corr_values(
    true_positions: pd.DataFrame, predicted_positions: pd.DataFrame
) -> Tuple[float, float]:
    comparison_df = pd.merge(predicted_positions, true_positions, on=["tournament_id", "team_id"])
    
    kendalls = []
    spearmans = []
    for _, df_cur in comparison_df.groupby("tournament_id"):
        kendalls += [kendalltau(df_cur["position_pred"], df_cur["position_true"]).correlation]
        spearmans += [spearmanr(df_cur["position_pred"], df_cur["position_true"]).correlation]
    
    kendalls = np.asarray(kendalls)
    spearmans = np.asarray(spearmans)
    
    kendalls[np.isnan(kendalls)] = 0.0
    spearmans[np.isnan(spearmans)] = 0.0
    
    return np.mean(kendalls), np.mean(spearmans)


def estimate_latent_vars_expectaion(
    tournament_ids: np.ndarray,
    team_ids: np.ndarray,
    player_ids: np.ndarray,
    tournament_question_nums: np.ndarray,
    team_answered_correctly: np.ndarray,
    individual_probas_to_answer: np.ndarray,
) -> np.ndarray:
    df_em = pd.DataFrame.from_dict({
        'tournament_id': tournament_ids,
        'team_id': team_ids,
        'player_id': player_ids,
        'tournament_question_num': tournament_question_nums,
        'proba_to_answer': individual_probas_to_answer,
    })
    df_em["proba_to_not_answer"] = 1 - df_em["proba_to_answer"]
    
    df_team = df_em.drop(columns=["player_id", "proba_to_answer"]).groupby(
        ["tournament_id", "team_id", "tournament_question_num"]
    ).agg("prod").reset_index()
    df_team["team_proba_to_answer"] = 1 - df_team["proba_to_not_answer"]
    
    df_em = pd.merge(
        df_em.drop(columns="proba_to_not_answer"),
        df_team.drop(columns="proba_to_not_answer"),
        on=["tournament_id", "team_id", "tournament_question_num"]
    )
    df_em["z_expectation"] = df_em["proba_to_answer"] / df_em["team_proba_to_answer"]
    
    z_expectation = df_em["z_expectation"].values
    z_expectation = np.where(team_answered_correctly == 0, 0, z_expectation)
    z_expectation = np.clip(z_expectation, 1e-6, 1 - 1e-6)
    return z_expectation


def get_top_n_most_and_least_difficult_tournament_names(
    model: LogisticRegression,
    mapper: Dict[Union[str, int], int],
    data: Dict[int, Dict],
    n: int
) -> Tuple[List[str], List[str]]:
    tournament_inv_mapper = {
        v: int(k.split("_")[0]) for k, v in mapper.items() if isinstance(k, str)
    }
    # store tournament questions weights sum, questions_num, mean weight
    tournaments_questions_difficulty = defaultdict(lambda: [0, 0, 0])
    
    for weight_index, weight in enumerate(model.coef_):
        try:
            tournament_id = tournament_inv_mapper[weight_index]
        except KeyError:
            continue
        tournaments_questions_difficulty[tournament_id][0] += weight
        tournaments_questions_difficulty[tournament_id][1] += 1
    
    # calculate mean question weight
    for tournament_id, questions_difficulty in tournaments_questions_difficulty.items():
        questions_difficulty[2] = questions_difficulty[0] / questions_difficulty[1]
    
    most_difficult_tournaments_names = [
        data[key]['tournament_name'] for key, value in tournaments_questions_difficulty.items()
        if -value[2] in heapq.nlargest(n, [-diff[2] for diff in tournaments_questions_difficulty.values()])
    ]
    least_difficult_tournaments_names = [
        data[key]['tournament_name'] for key, value in tournaments_questions_difficulty.items()
        if value[2] in heapq.nlargest(n, [diff[2] for diff in tournaments_questions_difficulty.values()])
    ]
    return most_difficult_tournaments_names, least_difficult_tournaments_names
