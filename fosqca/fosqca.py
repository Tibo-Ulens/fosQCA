# -*- coding: utf-8 -*-

import sys
import os
import argparse
import itertools
import copy
import logging
from collections import defaultdict

import pandas as pd


logger = logging.getLogger("fosQCA")


class FosQca:
    def __init__(
        self,
        sets: dict[str, pd.DataFrame],
        variables: list[str],
        outcome_col: str,
        outcome_value: int,
        consistency_threshold: float,
        coverage_threshold: float,
    ):
        self.sets = sets
        self.variables = variables
        self.outcome_col = outcome_col
        self.outcome_value = int(outcome_value)
        self.consistency_threshold = consistency_threshold
        self.coverage_threshold = coverage_threshold

    def generate_query(self, abstract_query: list):
        """
        Generate a textual query from an abstract one
        """

        query = ""

        for i in range(len(abstract_query)):
            if abstract_query[i] is None:
                continue

            query += f"`{self.variables[i]}`" + "==" + str(abstract_query[i]) + " & "

        return query[:-3]

    @staticmethod
    def query_len(query: str) -> int:
        products = query.split(" | ")

        return len(
            list(itertools.chain.from_iterable(map(lambda p: p.split(" & "), products)))
        )

    def generate_rules(self) -> dict[str, pd.DataFrame]:
        """
        Generate all possible variable rules for all input sets
        """

        rules = dict()

        for filename, dataset in self.sets.items():
            dataset_rules = []

            # get all the unique values each variable can assume
            per_variable_unique_values = []
            for i in range(len(self.variables)):
                per_variable_unique_values.append(
                    list(dataset[self.variables[i]].unique())
                )

            # generate every permutation of these unique values
            unique_values_permutations = [
                p for p in itertools.product(*per_variable_unique_values)
            ]

            for value_permutation in unique_values_permutations:
                query = self.generate_query(value_permutation)

                # Get the rows matching the query
                result = dataset.query(query)

                if result.empty:
                    row = [query, value_permutation, 0, 0, -1.0]
                    dataset_rules.append(row)

                    continue

                # Get the relative frequencies of the values of the outcome column
                p = result[self.outcome_col].value_counts(normalize=True, dropna=False)

                # consistency = (# cases with condition and outcome) / (# cases with condition)
                # which is the same as the relative frequency of a 'correct' outcome in the result
                # column
                if p.get(self.outcome_value, 0.0) >= self.consistency_threshold:
                    row = [
                        query,
                        value_permutation,
                        len(result[result[self.outcome_col] == self.outcome_value]),
                        len(result[result[self.outcome_col] != self.outcome_value]),
                        p.get(self.outcome_value, 0.0),
                    ]

                    dataset_rules.append(row)

            dataset_rules = pd.DataFrame(
                dataset_rules,
                columns=[
                    "rule",
                    "values",
                    "positive_cases",
                    "negative_cases",
                    "consistency",
                ],
            )

            dataset_rules = dataset_rules.drop_duplicates()

            rules[filename] = dataset_rules

        return rules

    @staticmethod
    def can_be_merged(rulea: list, ruleb: list) -> (bool, int):
        distance = 0
        idx = 0

        for i, (a, b) in enumerate(zip(rulea, ruleb)):
            if a != b:
                distance += 1
                idx = i

        if distance != 1:
            return (False, idx)

        return (True, idx)

    def merge_rule_list(
        self, rules: pd.DataFrame, dataset_name: str
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Attempt to merge rules in a given list, returns the new list of rules
        and a list of the unmerged rules that should be retained
        """

        merged_rules = set()
        new_rules = []

        for i in range(len(rules)):
            rulea = rules.iloc[i]

            for j in range(i + 1, len(rules)):
                ruleb = rules.iloc[j]

                cond_distance = self.can_be_merged(
                    rulea.get("values"), ruleb.get("values")
                )
                if not cond_distance[0]:
                    continue

                cons = (rulea.get("consistency"), ruleb.get("consistency"))

                match cons:
                    case (
                        x,
                        y,
                    ) if x >= self.consistency_threshold and y >= self.consistency_threshold:
                        pass
                    case (x, -1.0) if x >= self.consistency_threshold:
                        pass
                    case (-1.0, x) if x >= self.consistency_threshold:
                        pass
                    case (-1.0, -1.0):
                        pass
                    case _:
                        continue

                new_rule_values = copy.deepcopy(rulea.get("values"))
                new_rule_values = list(new_rule_values)
                new_rule_values[cond_distance[1]] = None
                new_rule_values = tuple(new_rule_values)

                new_query = self.generate_query(new_rule_values)

                rulea_values = list(
                    map(
                        lambda x: int(x) if x is not None else None,
                        list(rulea.get("values")),
                    )
                )
                ruleb_values = list(
                    map(
                        lambda x: int(x) if x is not None else None,
                        list(ruleb.get("values")),
                    )
                )
                new_values_pretty = list(
                    map(
                        lambda x: int(x) if x is not None else None,
                        list(new_rule_values),
                    )
                )

                logger.debug(
                    f"merged queries {rulea_values} {rulea.get("consistency")} + {ruleb_values} {ruleb.get("consistency")} -> {new_values_pretty}\n"
                )

                merged_rules.add(rulea.get("rule"))
                merged_rules.add(ruleb.get("rule"))

                # Get the rows matching the query
                # result = self.sets[0].query(new_query)
                result = self.sets[dataset_name].query(new_query)

                if result.empty:
                    row = [new_query, new_rule_values, 0, 0, -1.0]
                    new_rules.append(row)

                    continue

                # Get the relative frequencies of the values of the outcome column
                p = result[self.outcome_col].value_counts(normalize=True, dropna=False)

                # consistency = (# cases with condition and outcome) / (# cases with condition)
                # which is the same as the relative frequency of a 'correct' outcome in the result
                # column
                if p.get(self.outcome_value, 0.0) >= self.consistency_threshold:
                    row = [
                        new_query,
                        new_rule_values,
                        len(result[result[self.outcome_col] == self.outcome_value]),
                        len(result[result[self.outcome_col] != self.outcome_value]),
                        p.get(self.outcome_value, 0.0),
                    ]

                    new_rules.append(row)

        new_rules = pd.DataFrame(
            new_rules,
            columns=[
                "rule",
                "values",
                "positive_cases",
                "negative_cases",
                "consistency",
            ],
        )

        unmerged_rules = rules[
            rules.apply(lambda r: r.get("rule") not in merged_rules, axis=1)
        ]

        return (new_rules, unmerged_rules)

    def merge_rules(self, rules: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        new_rules, unmerged_rules = self.merge_rule_list(rules, dataset_name)
        rules = pd.concat([new_rules, unmerged_rules])

        while True:
            new_rules, unmerged_rules = self.merge_rule_list(rules, dataset_name)

            if new_rules.empty:
                break

            rules = pd.concat([new_rules, unmerged_rules])
            rules = rules.drop_duplicates()

            logger.debug(f"new rules:\n{rules}\n")

        return rules

    def evaluate_query(self, query: str) -> dict[str, tuple[int, int]]:
        rule_stats = dict()

        for filename, dataset in self.sets.items():
            result = dataset.query(query)

            cases_with_condition_and_outcome = len(
                result[result[self.outcome_col] == self.outcome_value]
            )
            cases_with_condition = len(result.values)
            cases_with_outcome = len(
                dataset[dataset[self.outcome_col] == self.outcome_value]
            )

            consistency = cases_with_condition_and_outcome / cases_with_condition
            coverage = cases_with_condition_and_outcome / cases_with_outcome

            rule_stats[filename] = (consistency, coverage)

        avg_cons = sum(map(lambda x: x[0], rule_stats.values())) / len(
            rule_stats.values()
        )
        avg_cov = sum(map(lambda x: x[1], rule_stats.values())) / len(
            rule_stats.values()
        )

        rule_stats["average"] = (avg_cons, avg_cov)

        return rule_stats

    def get_sufficient_rules(
        self, rules: pd.DataFrame
    ) -> list[tuple[str, dict[str, tuple[int, int]]]]:
        """
        Get the set of rules that reach the consistency and coverage thresholds
        on every input set

        This is almost certainly horrendously inefficient, but if inefficient
        works then inefficient it shall be
        """

        # Generate all combinations of the possible rules

        rule_combinations = []
        for r in range(len(rules) + 1):
            rule_combinations.extend(itertools.combinations(rules["rule"].values, r))

        rule_combinations = list(
            map(
                lambda rc: " | ".join(map(lambda r: f"({r})", list(rc))),
                rule_combinations,
            )
        )

        # Apply each combination to every dataset and calculate consistency and coverage

        rule_stats = defaultdict(dict)
        for rule in rule_combinations:
            if rule == "":
                continue

            for filename, dataset in self.sets.items():
                result = dataset.query(rule)

                cases_with_condition_and_outcome = len(
                    result[result[self.outcome_col] == self.outcome_value]
                )
                cases_with_condition = len(result.values)
                cases_with_outcome = len(
                    dataset[dataset[self.outcome_col] == self.outcome_value]
                )

                consistency = cases_with_condition_and_outcome / cases_with_condition
                coverage = cases_with_condition_and_outcome / cases_with_outcome

                if (
                    consistency < self.consistency_threshold
                    or coverage < self.coverage_threshold
                ):
                    if rule in rule_stats:
                        del rule_stats[rule]

                    break

                rule_stats[rule][filename] = (consistency, coverage)

        # Average out the consistency and coverage

        for rule, stats in rule_stats.items():
            avg_cons = sum(map(lambda x: x[0], stats.values())) / len(stats.values())
            avg_cov = sum(map(lambda x: x[1], stats.values())) / len(stats.values())

            rule_stats[rule]["average"] = (avg_cons, avg_cov)

        # yarr harr fiddle dee dee

        return list(rule_stats.items())


def main():
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--outcome-col", default="outcome")
    parser.add_argument("-o", "--outcome-value", default=1, type=float)
    parser.add_argument("-i", "--ignore-col", action="append")
    parser.add_argument("--verbose", action="count", default=0)
    parser.add_argument("-c", "--consistency", default=0.8, type=float)
    parser.add_argument("-v", "--coverage", default=0.8, type=float)
    parser.add_argument("-f", "--formula")
    parser.add_argument("sets", nargs="*")

    args = parser.parse_args()

    match args.verbose:
        case 0:
            verbosity = logging.INFO
        case _:
            verbosity = logging.DEBUG

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    logger.setLevel(verbosity)
    logger.addHandler(handler)

    if len(args.sets) < 1:
        print("at least 1 dataset is required")
        sys.exit(1)

    for set_file in args.sets:
        if not os.path.isfile(set_file):
            print(f"'{set_file}' is not a file")
            sys.exit(1)

    sets = dict()
    cols = None
    for set_file in args.sets:
        data = pd.read_csv(set_file)
        data.attrs = {"file": set_file}

        sets[set_file] = data
        cols = data.columns

    for filename, s in sets.items():
        if (s.columns != cols).any():
            print(f"{filename} has invalid headers, expected {list(cols)}")

    cols = list(cols)

    if args.outcome_col in cols:
        cols.remove(args.outcome_col)

    for ignored in args.ignore_col:
        if ignored in cols:
            cols.remove(ignored)

    qca = FosQca(
        sets,
        variables=cols,
        outcome_col=args.outcome_col,
        outcome_value=args.outcome_value,
        consistency_threshold=args.consistency,
        coverage_threshold=args.coverage,
    )

    if args.formula:
        print(f"evaluating {args.formula} on all input sets...\n")

        result = qca.evaluate_query(args.formula)

        formatted_info = "\n".join(
            map(
                lambda x: f"{x[0]} :: consistency: {x[1][0]} coverage: {x[1][1]}",
                result.items(),
            )
        )

        print(formatted_info)

        sys.exit(0)

    rules = qca.generate_rules()

    for filename, file_rules in rules.items():
        logger.info(f"possible rules for {filename}:\n{file_rules}\n")

    merged_rules = dict()
    for filename, file_rules in rules.items():
        merged_rules[filename] = qca.merge_rules(file_rules, filename)

        logger.info(f"merged rules for {filename}:\n{merged_rules[filename]}\n")

    total_merged_rules = list()
    for file_rules in merged_rules.values():
        total_merged_rules.extend(file_rules["rule"].values)

    total_merged_rules = pd.DataFrame(total_merged_rules, columns=["rule"])
    total_merged_rules = total_merged_rules.drop_duplicates()

    print(f"final list of possible rules:\n{total_merged_rules}\n")

    sufficient_rules = qca.get_sufficient_rules(total_merged_rules)
    sufficient_rules.sort(key=lambda r: qca.query_len(r[0]), reverse=True)
    sufficient_rules.sort(key=lambda r: r[1]["average"])

    for r in sufficient_rules:
        formatted_rule = "\n".join(r[0].split(" | "))
        formatted_info = "\n".join(
            map(
                lambda x: f"{x[0]} :: consistency: {x[1][0]} coverage: {x[1][1]}",
                r[1].items(),
            )
        )

        print(
            f"solution with {qca.query_len(r[0])} terms:\n{formatted_rule}\n\n{formatted_info}\n\n"
        )
